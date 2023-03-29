from typing import Any, Dict, Generator, List
from uuid import UUID

import boto3
import numpy as np
import pandas as pd

from datatypes import Functions, InvocationTypes


def serialize(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_json(double_precision=5)
    elif isinstance(obj, UUID):
        return str(obj)
    else:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )


def invoke_lambda(
    session: boto3.session.Session,
    lambda_function_name: Functions,
    invocation_type: InvocationTypes,
    lambda_payload: str,
) -> Dict:
    lambda_client = session.client("lambda")

    lambda_response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType=invocation_type.value,
        Payload=lambda_payload,
    )

    return lambda_response


def invoke_lambda_local(
    lambda_function_name: Functions,
    invocation_type: InvocationTypes,
    lambda_payload: str,
) -> Dict:
    import requests

    url = "http://localhost:9000/2015-03-31/functions/function/invocations"
    res = requests.post(url, data=lambda_payload)

    return res.text
