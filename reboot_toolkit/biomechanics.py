import mujoco
import numpy as np
import pandas as pd

from xml.etree import ElementTree as ET

from . import utils as ut


def add_ik_joints(df, coords):

    for coord in coords:
        if 'lEar_X' in df.columns:
            df[f'SKULL_{coord}'] = (df[f'lEar_{coord}'] + df[f'rEar_{coord}']) / 2.

        df[f'neck_{coord}'] = (df[f'LSJC_{coord}'] + df[f'RSJC_{coord}']) / 2.

        df[f'pelvis_{coord}'] = (df[f'LHJC_{coord}'] + df[f'RHJC_{coord}']) / 2.

        df[f'torso_{coord}'] = df[f'pelvis_{coord}']


def calc_dist_btwn(pd_object, begin_joint, end_joint):

    dist = np.sqrt(
        (pd_object[f"{end_joint}_X"] - pd_object[f"{begin_joint}_X"]) ** 2 +
        (pd_object[f"{end_joint}_Y"] - pd_object[f"{begin_joint}_Y"]) ** 2 +
        (pd_object[f"{end_joint}_Z"] - pd_object[f"{begin_joint}_Z"]) ** 2
    )

    if isinstance(dist, pd.Series):
        return dist.median()

    return dist


def get_bone_lengths(pd_object):
    return {
        'shoulders': calc_dist_btwn(pd_object, 'LSJC', 'RSJC'),
        'hips': calc_dist_btwn(pd_object, 'LHJC', 'RHJC'),
        'torso_length': calc_dist_btwn(pd_object, 'neck', 'pelvis'),
        'left_upper_arm': calc_dist_btwn(pd_object, 'LSJC', 'LEJC'),
        'left_lower_arm': calc_dist_btwn(pd_object, 'LEJC', 'LWJC'),
        'right_upper_arm': calc_dist_btwn(pd_object, 'RSJC', 'REJC'),
        'right_lower_arm': calc_dist_btwn(pd_object, 'REJC', 'RWJC'),
        'left_upper_leg': calc_dist_btwn(pd_object, 'LHJC', 'LKJC'),
        'left_lower_leg': calc_dist_btwn(pd_object, 'LKJC', 'LAJC'),
        'right_upper_leg': calc_dist_btwn(pd_object, 'RHJC', 'RKJC'),
        'right_lower_leg': calc_dist_btwn(pd_object, 'RKJC', 'RAJC')
    }


def scale_human_xml(ik_df, desired_mass, boto3_session):

    args = {
        "bone_length_dict": get_bone_lengths(ik_df),
        "desired_mass": desired_mass
    }

    payload = {"function_name": "scale_human_xml", "args": args}

    # lambda_client = boto3.session.Session().client("lambda")
    #
    # resp = lambda_client.invoke(
    #     FunctionName='reboot_toolkit_backend_dev',
    #     InvocationType="RequestResponse",
    #     Payload=json.dumps(payload),
    # )
    #
    # return json.loads(resp['Payload'].read())
    return ut.handle_lambda_invocation(boto3_session, payload)


def get_model_info(human_model_xml_str, element_type, return_names=False):

    if element_type not in {'body', 'joint'}:
        raise ValueError("element_type must be 'body' or 'joint'")

    xml_tree = ET.ElementTree(ET.fromstring(human_model_xml_str))

    model_dict = {
        element.attrib['name']: element.attrib
        for element in xml_tree.getroot().iter(element_type)
        if 'name' in element.attrib
    }

    if return_names:
        return list(model_dict.keys())

    return model_dict


def reorder_joint_angle_df_like_model(model, data, joint_angle_df, joint_angle_names):

    name_order = [None] * data.qpos.shape[0]

    for joint_angle_name in joint_angle_names:

        q_pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_angle_name)

        if q_pos_id >= 0:
            name_order[
                model.jnt_qposadr[q_pos_id]
            ] = joint_angle_name

    return joint_angle_df[name_order]


def read_inverse_kinematics_internal(ik_file_path):
    df = pd.read_csv(ik_file_path, index_col=[0]).apply(pd.to_numeric, errors='ignore')

    df[['org_movement_id', 'movement_id']] = df[['org_movement_id', 'movement_id']].astype('string')

    for coord in ('X', 'Y', 'Z'):
        add_ik_joints(df, (coord,))

        df[f'{coord.lower()}_translation'] = df[f'pelvis_{coord}']

    return df


def main():
    import matplotlib.pyplot as plt

    ik_df = read_inverse_kinematics_internal('/Users/jimmybuffi/Desktop/RebootMotion/mujoco/demo_ik_rha.csv.gz')
    ik_df['right_elbow_var'] = 0  # set the target joint angle for the elbow varus valgus degrees of freedom
    ik_df['left_elbow_var'] = 0

    time_series = ik_df['time'].copy()
    dt = ik_df['time'].diff().median()

    model_xml_str = scale_human_xml(ik_df, 97.5)

    joint_names = get_model_info(model_xml_str, 'joint', return_names=True)

    model = mujoco.MjModel.from_xml_string(model_xml_str)
    model.opt.timestep = dt

    data = mujoco.MjData(model)

    ik_df = reorder_joint_angle_df_like_model(model, data, ik_df, joint_names)
    vel_df = ik_df.copy().apply(np.gradient, raw=True) / dt

    q_force_inverse = []

    for i, row_pos in ik_df.iterrows():

        row_vel = vel_df.iloc[i]

        data.qpos = row_pos.to_numpy()
        data.qvel = row_vel.to_numpy()

        mujoco.mj_step(model, data)
        mujoco.mj_inverse(model, data)

        q_force_inverse.append(data.qfrc_inverse.copy())

    force_df = pd.DataFrame(data=q_force_inverse, columns=joint_names)
    force_df['time'] = time_series

    plt.figure()

    plt.plot(force_df['x_translation'], label='x')
    plt.plot(force_df['y_translation'], label='y')
    plt.plot(force_df['z_translation'], label='z')

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
