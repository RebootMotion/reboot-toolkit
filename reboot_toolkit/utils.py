import logging
import os
from getpass import getpass
from typing import Any, Optional
from uuid import UUID

import boto3
import pandas as pd
from dotenv import dotenv_values, load_dotenv

from .datatypes import Functions, InvocationTypes

logger = logging.getLogger(__name__)


def setup_aws(
        org_id: Optional[str] = None, 
        aws_access_key_id: Optional[str] = None, 
        aws_secret_access_key: Optional[str] = None, 
        aws_default_region: Optional[str] = None
    ) -> boto3.Session:
<<<<<<< HEAD
    from dotenv import load_dotenv

    load_dotenv()

    if 'ORG_ID' not in os.environ:
        input_org_id = getpass(f'Input org_id here (or input empty string to use {org_id}):')

        if len(input_org_id.strip()) == 0:
            os.environ['ORG_ID'] = org_id

        else:
            os.environ['ORG_ID'] = input_org_id

    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id or getpass('Input AWS_ACCESS_KEY_ID here:')

    if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key or getpass('Input SECRET_ACCESS_KEY here:')

=======
    load_dotenv()

    if 'ORG_ID' not in os.environ:
        os.environ['ORG_ID'] = org_id or getpass('Enter reboot-motion org_id here:')
    if 'AWS_ACCESS_KEY_ID' not in os.environ:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id or getpass('Enter AWS_ACCESS_KEY_ID here:')
    if 'AWS_SECRET_ACCESS_KEY' not in os.environ:
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key or getpass('Enter SECRET_ACCESS_KEY here:')
>>>>>>> 0841190 (Improve dev enviornment)
    if 'AWS_DEFAULT_REGION' not in os.environ:
        os.environ['AWS_DEFAULT_REGION'] = aws_default_region or getpass('Input AWS_DEFAULT_REGION here:')

    boto3_session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
        region_name=os.environ['AWS_DEFAULT_REGION']
    )

    print('Current Boto3 Session:')
    print(boto3_session)
    return boto3_session


def serialize(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_json(double_precision=5)
    elif isinstance(obj, UUID):
        return str(obj)
    else:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )
    

def lambda_has_error(response: dict) -> bool:
    return 'FunctionError' in response


def invoke_lambda(
    session: boto3.Session,
    lambda_function_name: Functions,
    invocation_type: InvocationTypes,
    lambda_payload: str,
) -> dict:
    config = {**dotenv_values()}
    if "DEV" in config:
        return invoke_lambda_local(
            session=session,
            lambda_function_name=lambda_function_name,
            invocation_type=invocation_type,
            lambda_payload=lambda_payload
        )

    lambda_client = session.client("lambda")

    lambda_response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType=invocation_type.value,
        Payload=lambda_payload,
    )

    return lambda_response


def invoke_lambda_local(
    session: Any,
    lambda_function_name: Functions,
    invocation_type: InvocationTypes,
    lambda_payload: str,
) -> dict:
    from unittest.mock import Mock

    import requests

    url = "http://localhost:9000/2015-03-31/functions/function/invocations"
    res = requests.post(url, data=lambda_payload)

    text = res.text
    mock_res = {'Payload': Mock(read=Mock(return_value=text)), 'StatusCode': 200}

    if "errorMessage" in res.text:
        mock_res["FunctionError"] = "error!"


    return mock_res
