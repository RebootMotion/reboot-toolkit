import boto3
import json
import mujoco
import numpy as np
import pandas as pd


def scale_human_xml(lambda_payload, function_name):

    lambda_client = boto3.session.Session().client("lambda")

    return lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=lambda_payload,
    )


def add_ik_joints(df, coord):
    if 'lEar_X' in df.columns:
        df[f'SKULL_{coord}'] = (df[f'lEar_{coord}'] + df[f'rEar_{coord}']) / 2.

    df[f'neck_{coord}'] = (df[f'LSJC_{coord}'] + df[f'RSJC_{coord}']) / 2.

    df[f'pelvis_{coord}'] = (df[f'LHJC_{coord}'] + df[f'RHJC_{coord}']) / 2.

    df[f'torso_{coord}'] = df[f'pelvis_{coord}']


def read_inverse_kinematics_internal(ik_file_path):
    df = pd.read_csv(ik_file_path, index_col=[0]).apply(pd.to_numeric, errors='ignore')

    df[['org_movement_id', 'movement_id']] = df[['org_movement_id', 'movement_id']].astype('string')

    for coord in ('X', 'Y', 'Z'):
        add_ik_joints(df, coord)

        df[f'{coord.lower()}_translation'] = df[f'pelvis_{coord}']

    return df


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


def main():
    function_name = 'reboot_toolkit_backend_dev'

    ik_df = read_inverse_kinematics_internal('/Users/jimmybuffi/Desktop/RebootMotion/mujoco/demo_ik_rha.csv.gz')

    xml_path = '/Users/jimmybuffi/Desktop/RebootMotion/mujoco/humanoid_dynamics.xml'

    xml_str = open(xml_path).read()

    args = {
        "xml_tree_str": xml_str,
        "bone_length_dict": get_bone_lengths(ik_df),
        "desired_mass": 97.5
    }

    payload = {"function_name": "scale_human_xml", "args": args}

    resp = scale_human_xml(json.dumps(payload), function_name=function_name)

    model_xml_str = json.loads(resp['Payload'].read().decode('utf-8'))
    print(model_xml_str)

    model = mujoco.MjModel.from_xml_string(model_xml_str)
    print(model)


if __name__ == '__main__':
    main()
