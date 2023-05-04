import json
from typing import Any, Dict, Generator, List, Optional
from uuid import uuid4

import boto3
import numpy as np
import pandas as pd
from botocore.errorfactory import ClientError

from . import utils as ut
from .datatypes import (Functions, Handedness, InvocationTypes, MocapType,
                        MovementType)


def recommendation(
    session: boto3.Session,
    metrics_df: pd.DataFrame,
    movement_type: MovementType,
    mocap_type: MocapType,
    dom_hand: Handedness,
) -> pd.DataFrame | dict:
    args = {
        "metrics_df": metrics_df,
        "movement_type": movement_type,
        "mocap_type": mocap_type,
        "dom_hand": dom_hand,
    }
    payload = {"function_name": "recommendation_values", "args": args}
    payload = json.dumps(payload, default=ut.serialize)
    response = ut.invoke_lambda(
        session=session,
        lambda_function_name=Functions.BACKEND_DEV,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )

    payload = response["Payload"].read()
    if ut.lambda_has_error(response):
        print(f"Error in calculation")
        print(payload)
        return payload
        
    return pd.read_json(json.loads(payload))
