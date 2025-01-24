from __future__ import annotations

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


def get_segment_lengths(pd_object: Union[pd.Series, pd.DataFrame]):
    """
    Get all the segment lengths in an IK dataframe or series.

    :param pd_object: the IK DataFrame or Series
    :return: dict of segment lengths
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
        boto3_session = boto3.session.Session()

    args = {
        "bone_length_dict": get_segment_lengths(ik_df),
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
    input_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reorder the columns of an IK df to match the order of joints in MuJoCo so its easier to simulate.

    :param model: the MuJoCo model
    :param input_df: the dataframe to reorder according to the model joint order
    :return: the reordered dataframe
    """
    if model.nq != model.njnt:
        raise NotImplementedError(
            "model n generalized coordinates not equal to n joints "
            "- possible free joint in mujoco model, which is not supported yet"
        )

    name_order = [None] * model.nq

    for idx in range(model.njnt):

        joint_angle_name = model.joint(idx).name

        q_pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_angle_name)

        if q_pos_id < 0:
            raise ValueError(
                f"unrecognized joint_angle_name in mujoco model: {joint_angle_name}"
            )

        name_order[model.jnt_qposadr[q_pos_id]] = joint_angle_name

    return input_df[name_order]


def inverse_dynamics(
    mj_model: mujoco.MjModel,
    joint_angle_df: pd.DataFrame,
    dom_hand: str,
    col_suffix: str = "invdyn",
) -> pd.DataFrame:
    """
    Run inverse dynamics simulation with MuJoCo model and a dataframe of joint angles.

    :param mj_model: the MuJoCo model
    :param joint_angle_df: dataframe of joint angles to use for the inverse dynamics simulation
    :param dom_hand: the dominant hand used in the movement
    :param col_suffix: suffix to add to column names to distinguish inverse dynamics from inverse kinematics
    :return:
    """

    mj_data = mujoco.MjData(mj_model)

    dt = joint_angle_df["time"].diff().median()
    mj_model.opt.timestep = dt

    positions_df = reorder_joint_angle_df_like_model(mj_model, joint_angle_df.copy())

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
        data=sim_results, columns=[f"{jn}_{col_suffix}" for jn in list(positions_df)]
    )


def inverse_dynamics_multiple(
    player_df: pd.DataFrame,
    movement_type: str,
    player_mass: int | float,
    dom_hand: str,
    suppress_model_creation_error: bool = False,
):
    """
    Run inverse dynamics simulation for multiple movements for the same player.

    :param player_df: dataframe containing multiple movements for the same player
    :param movement_type: the movement type intended for the mujoco simulation
    :param player_mass: the mass of the player in kg
    :param dom_hand: the dominant hand used in the movement
    :param suppress_model_creation_error: whether to return an empty df upon an error in model creation
    :return: dataframe of inverse dynamics simulation results for multiple movements
    """
    model_xml_str = scale_human_xml(
        player_df,
        player_mass,
        movement_type,
        verbose=False,
    )

    try:
        model = mujoco.MjModel.from_xml_string(model_xml_str)

    except ValueError:
        if suppress_model_creation_error:
            return pd.DataFrame()
        raise

    inv_dyn_dfs = []

    for org_movement_id, org_movement_df in player_df.groupby("org_movement_id"):
        id_df = inverse_dynamics(
            model, org_movement_df.reset_index(drop=True), dom_hand
        )
        id_df["org_movement_id"] = org_movement_id
        inv_dyn_dfs.append(id_df)

    return pd.concat(inv_dyn_dfs).reset_index().rename(columns={"index": "frame"})
