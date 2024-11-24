import boto3
import mujoco
import numpy as np
import pandas as pd

from typing import Union, Optional
from xml.etree import ElementTree as ET

from . import utils as ut


def calc_dist_btwn(
    pd_object: Union[pd.Series, pd.DataFrame], begin_joint: str, end_joint: str
) -> float:
    """
    Calculate the distance between two joints

    :param pd_object: the pandas dataframe or series with the data
    :param begin_joint: the beginning joint in the calculation
    :param end_joint: the ending joint in the calculation
    :return: the distance between the joints, either the median of a series or the distance for a single value
    """

    dist = np.sqrt(
        (pd_object[f"{end_joint}_X"] - pd_object[f"{begin_joint}_X"]) ** 2
        + (pd_object[f"{end_joint}_Y"] - pd_object[f"{begin_joint}_Y"]) ** 2
        + (pd_object[f"{end_joint}_Z"] - pd_object[f"{begin_joint}_Z"]) ** 2
    )

    if isinstance(dist, pd.Series):
        return dist.median()

    return dist


def get_bone_lengths(pd_object: Union[pd.Series, pd.DataFrame]):
    """
    Get all the bone lengths in an IK dataframe or series

    :param pd_object: the IK DataFrame or Series
    :return: dict of bone lengths
    """
    return {
        "shoulders": calc_dist_btwn(pd_object, "LSJC", "RSJC"),
        "hips": calc_dist_btwn(pd_object, "LHJC", "RHJC"),
        "torso_length": calc_dist_btwn(pd_object, "neck", "pelvis"),
        "left_upper_arm": calc_dist_btwn(pd_object, "LSJC", "LEJC"),
        "left_lower_arm": calc_dist_btwn(pd_object, "LEJC", "LWJC"),
        "right_upper_arm": calc_dist_btwn(pd_object, "RSJC", "REJC"),
        "right_lower_arm": calc_dist_btwn(pd_object, "REJC", "RWJC"),
        "left_upper_leg": calc_dist_btwn(pd_object, "LHJC", "LKJC"),
        "left_lower_leg": calc_dist_btwn(pd_object, "LKJC", "LAJC"),
        "right_upper_leg": calc_dist_btwn(pd_object, "RHJC", "RKJC"),
        "right_lower_leg": calc_dist_btwn(pd_object, "RKJC", "RAJC"),
    }


def scale_human_xml(
    ik_df: pd.DataFrame,
    desired_mass: float,
    movement_type: str = "baseball-pitching",
    boto3_session: Optional[boto3.Session] = None,
    verbose: bool = True,
) -> str:
    """
    Retrieve a scaled humanoid model from AWS, scaled in both length and mass

    :param ik_df: the inverse kinematics data frame to use for calculating scaling factors
    :param desired_mass: the desired mass to scale to
    :param movement_type: the movement type intended for the mujoco simulation
    :param boto3_session: the boto3 session for accessing AWS
    :param verbose: whether to print status info
    :return: the scaled MuJoCo XML model string
    """
    if boto3_session is None:
        raise ValueError("Please input a boto3 session")

    args = {
        "bone_length_dict": get_bone_lengths(ik_df),
        "desired_mass": desired_mass,
        "movement_type": movement_type,
    }

    payload = {"function_name": "scale_human_xml", "args": args}

    return ut.handle_lambda_invocation(boto3_session, payload, verbose=verbose)


def get_model_info(
    human_model_xml_str: str, element_type: str, return_names: bool = False
) -> Union[list, dict]:
    """
    Get the body or joint info for a MuJoCo XML model

    :param human_model_xml_str: the overall model XML string
    :param element_type: the element type to inspect (body or joint)
    :param return_names: whether to return a list of names or a dict with all info
    :return:
    """
    if element_type not in {"body", "joint"}:
        raise ValueError("element_type must be 'body' or 'joint'")

    xml_tree = ET.ElementTree(ET.fromstring(human_model_xml_str))

    model_dict = {
        element.attrib["name"]: element.attrib
        for element in xml_tree.getroot().iter(element_type)
        if "name" in element.attrib
    }

    if return_names:
        return list(model_dict.keys())

    return model_dict


def reorder_joint_angle_df_like_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik_df: pd.DataFrame,
    joint_angle_names: list[str],
) -> pd.DataFrame:
    """
    Reorder the columns of an IK df to match the order of joints in MuJoCo so its easier to simulate

    :param model: the MuJoCo model
    :param data: the MuJoCo data element
    :param ik_df: the IK dataframe
    :param joint_angle_names: the joint angle names to order
    :return: the reordered dataframe
    """

    name_order = [None] * data.qpos.shape[0]

    for joint_angle_name in joint_angle_names:

        q_pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_angle_name)

        if q_pos_id >= 0:
            name_order[model.jnt_qposadr[q_pos_id]] = joint_angle_name

    return ik_df[name_order]


def inverse_dynamics(mj_model, mj_joint_names, joint_angle_df, dom_hand, col_suffix="invdyn"):

    mj_data = mujoco.MjData(mj_model)

    dt = joint_angle_df["time"].diff().median()
    mj_model.opt.timestep = dt

    positions_df = reorder_joint_angle_df_like_model(
        mj_model, mj_data, joint_angle_df.copy(), mj_joint_names
    )

    # for lefties, revert the post-processing negation that was done based on handedness
    if dom_hand.lower().startswith("l"):
        positions_df["pelvis_rot"] = -(positions_df["pelvis_rot"] - 180.0)
        positions_df["pelvis_side"] = -positions_df["pelvis_side"]

        positions_df["torso_rot"] = -positions_df["torso_rot"]
        positions_df["torso_side"] = -positions_df["torso_side"]

    angle_cols = [col for col in list(positions_df) if not col.endswith("translation")]
    positions_df[angle_cols] = positions_df[angle_cols].apply(np.radians)

    velocities_df = positions_df.copy().apply(np.gradient, raw=True) / dt
    accel_df = velocities_df.copy().apply(np.gradient, raw=True) / dt

    sim_results = []

    for i, row_pos in positions_df.iterrows():

        row_vel = velocities_df.iloc[i]
        row_acc = accel_df.iloc[i]

        mj_data.qpos = row_pos.to_numpy()
        mj_data.qvel = row_vel.to_numpy()
        mj_data.qacc = row_acc.to_numpy()

        mujoco.mj_inverse(mj_model, mj_data)

        sim_results.append(mj_data.qfrc_inverse.copy())

    return pd.DataFrame(
        data=sim_results, columns=[f"{jn}_{col_suffix}" for jn in mj_joint_names]
    )
