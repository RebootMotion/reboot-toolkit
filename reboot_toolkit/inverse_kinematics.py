import json
import logging
import os
import time
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

import boto3
import numpy as np
import pandas as pd
from botocore.errorfactory import ClientError

from . import utils as ut
from .datatypes import Functions, InvocationTypes


def get_log_level() -> Any:
    level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper())
    return level


logger = logging.getLogger(__name__)
logger.setLevel(get_log_level())


def read_trc(in_file_name: str) -> pd.DataFrame:
    trc_df = pd.read_csv(in_file_name, sep='\t', header=4, dtype=np.float32)

    n_lines = 4

    if in_file_name.startswith('s3://'):
        import s3fs
        FS = s3fs.S3FileSystem(anon=False)
        with FS.open(in_file_name, "r") as my_file:
            # data = my_file.readlines()
            data = [next(my_file) for _ in range(n_lines)]

    else:
        with open(in_file_name, "r") as my_file:
            data = [next(my_file) for _ in range(n_lines)]

    col_current = list(trc_df)
    col_prefixes = data[3].split('\t')
    col_prefixes = [col.rstrip() for col in col_prefixes if ((col != '') & (col != '\n'))]

    col_headers = {col_current[0]: col_prefixes[0], col_current[1]: 'time'}

    col_num = 2

    for col in col_prefixes[2:]:

        for coord in ['X', 'Y', 'Z']:
            col_headers[col_current[col_num]] = col + '_' + coord

            trc_df[col_current[col_num]] = trc_df[col_current[col_num]] / 1000

            col_num = col_num + 1

    trc_df = trc_df.rename(columns=col_headers).interpolate(method='linear')

    for coord in ['X', 'Y', 'Z']:
        trc_df[f'neck_{coord}'] = (trc_df[f'LSJC_{coord}'] + trc_df[f'RSJC_{coord}']) / 2.0

        trc_df[f'pelvis_{coord}'] = (trc_df[f'LHJC_{coord}'] + trc_df[f'RHJC_{coord}']) / 2.0

        trc_df[f'torso_{coord}'] = trc_df[f'pelvis_{coord}']

    return trc_df


def inverse_kinematics(
    session: boto3.Session,
    dom_hand: Optional[str],
    trc_df: pd.DataFrame,
    results_file_name: str,  # we assume movement ID is between the "_" characters
    movement_id: Optional[str],
    movement_type: str,
) -> str:
    print("Running inverse kinematics, this can take over 30s to run...")
    args = {
        "dom_hand": dom_hand,
        "trc_df": trc_df,
        "results_file_name": results_file_name,
        "movement_id": movement_id,
        "movement_type": movement_type,
    }
    payload = {"function_name": "inverse_kinematics", "args": args}
    payload = json.dumps(payload, default=ut.serialize)
    response = ut.invoke_lambda(
        session=session,
        lambda_function_name=Functions.INVERSE_KINEMATICS,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )

    payload = response["Payload"].read()
    if ut.lambda_has_error(response):
        print(f"Error in calculation")
        print(payload)
        return payload
        
    return pd.read_json(json.loads(payload))
