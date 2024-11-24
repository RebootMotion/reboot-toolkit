import json
import logging
import os

from getpass import getpass
from typing import Any, Union
from uuid import UUID

import boto3
import pandas as pd
from dotenv import dotenv_values, load_dotenv

from .datatypes import Functions, InvocationTypes
from mlb_statsapi import GameRequest, Game

logger = logging.getLogger(__name__)


def setup_aws(verbose: bool = True) -> boto3.Session:
    """
    Set up the necessary AWS credentials to connect to the AWS API
    :param verbose: whether to print status info
    :return: the AWS session
    """

    load_dotenv()

    if 'ORG_ID' not in os.environ:
        credentials_string = getpass('Input JSON string containing your credentials:')
        credentials = json.loads(credentials_string)

        try:
            os.environ['ORG_ID'] = credentials['org_id']
            os.environ['AWS_ACCESS_KEY_ID'] = credentials['aws_access_key_id']
            os.environ['AWS_SECRET_ACCESS_KEY'] = credentials['aws_secret_access_key']
            os.environ['AWS_SESSION_TOKEN'] = credentials['aws_session_token']
            os.environ['AWS_DEFAULT_REGION'] = credentials['aws_default_region']
        except KeyError as e:
            raise KeyError(f'Credentials string missing key: {e}')

    session_credentials = {
        "aws_access_key_id": os.environ['AWS_ACCESS_KEY_ID'],
        "aws_secret_access_key": os.environ['AWS_SECRET_ACCESS_KEY'],
        "region_name": os.environ['AWS_DEFAULT_REGION'],
    }

    if 'AWS_SESSION_TOKEN' in os.environ:
        session_credentials['aws_session_token'] = os.environ['AWS_SESSION_TOKEN']

    boto3_session = boto3.Session(**session_credentials)

    if verbose:
        print('Org ID:')
        print(os.environ['ORG_ID'])
        print()
        print('Current Boto3 Session:')
        print(boto3_session)

    return boto3_session


def decorate_primary_segment_df_with_stats_api(primary_segment_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decorate Reboot Motion play by play metrics with additional data from stats API
    """
    game_pks = primary_segment_data_df['session_num'].unique()
    all_game_metrics = None
    for game_pk in game_pks:
        print(f"Processing game: {game_pk}")
        game_request = GameRequest(game_pk=game_pk)
        game_data: Game = game_request.make_request()
        game_df = game_data.get_filtered_pitch_metrics_by_play_id_as_df(play_ids=primary_segment_data_df[primary_segment_data_df['session_num'] == game_pk]["org_movement_id"].tolist())
        if all_game_metrics is None:
            all_game_metrics = game_df
        else:
            all_game_metrics = pd.concat([all_game_metrics, game_df])

    if 'pitch_type' in primary_segment_data_df.columns:
        all_game_metrics.rename(columns={'pitch_type': 'pitch_type_stats_api'},inplace=True)    # If there are already pitch types defined in the primary DataFrame, renames the pitch types from the stats API

    return primary_segment_data_df.merge(all_game_metrics, how='left', left_on='org_movement_id', right_index=True)


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


def handle_lambda_invocation(session: boto3.Session, payload: dict, verbose: bool = True) -> Union[str, dict]:
    """
    Invoke a lambda function with the input payload.

    :param session: the boto3 session info to use
    :param payload: the lambda payload
    :param verbose: whether to print status
    :return: the serialized lambda response
    """

    payload = json.dumps(payload, default=serialize)

    if verbose:
        print('Sending to AWS...')
    response = invoke_lambda(
        session=session,
        lambda_function_name=Functions.BACKEND,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )

    if verbose:
        print('Reading Response...')
    payload = response["Payload"].read()

    if lambda_has_error(response):
        print(f"Error in calculation:")
        print(payload)
        return payload

    if verbose:
        print("Returning AWS Response...")
    return json.loads(payload)
