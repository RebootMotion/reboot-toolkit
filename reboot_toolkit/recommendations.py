from __future__ import annotations

import json

import boto3
import pandas as pd
import plotly
import plotly.graph_objects as go

from . import utils as ut
from .datatypes import (Functions, Handedness, InvocationTypes, MocapType,
                        MovementType)


def handle_recommendation_mocap_type(metrics_df: pd.DataFrame, mocap_type: MocapType) -> str | MocapType:
    """
    Handle a mocap type of None or raise an exception for an invalid mocap type.

    :param metrics_df: dataframe of Reboot Motion metrics
    :param mocap_type: the mocap type enum
    :return: a mocap type enum
    """
    if mocap_type is not None:
        pass

    elif mocap_type is None and len(metrics_df['mocap_type'].unique()) == 1:
        mocap_type = metrics_df['mocap_type'].iloc[0]

    else:
        raise KeyError(
            'input mocap_type not specified, and a unique mocap_type was not found in the file, please specify one'
        )

    return mocap_type


def recommendation(
    session: boto3.Session,
    metrics_df: pd.DataFrame,
    movement_type: MovementType,
    mocap_type: MocapType | None,
    dom_hand: Handedness,
) -> pd.DataFrame | dict:
    """
    This is a recommendation engine with the goal of giving coaches guidance on how to help their players improve.
    This recommendation engine is trained on player metrics as a predictor of fastball velocity for pitching and bat velocity for hitting. 
    Using a random forest like model for training, the recommendation engine then extracts the impact of individual metrics on the overall result.
    Given a player's metrics, the engine is then able to recommend aspects to focus on to improve the overall result.

    :param session: the input boto3 session
    :param metrics_df: processed metrics for a player
    :param movement_type: movement type for player
    :param mocap_type: motion capture type
    :param dom_hand: dominant hand for player
    :return: dataframe with recommendation information
    """

    mocap_type = handle_recommendation_mocap_type(metrics_df, mocap_type)

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
        lambda_function_name=Functions.BACKEND,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )

    payload = response["Payload"].read()
    if ut.lambda_has_error(response):
        print(f"Error in calculation")
        print(payload)
        return payload
        
    return pd.read_json(json.loads(payload))


def recommendation_violin_plot(
    session: boto3.Session,
    metrics_df: pd.DataFrame,
    movement_type: MovementType,
    mocap_type: MocapType | None,
    dom_hand: Handedness,
    num_features: int = 5
) -> go.Figure | dict | None:
    """
    Display the recommendation engine output using a violin plot. 

    :param session: the input boto3 session
    :param metrics_df: processed metrics for a player
    :param movement_type: movement type for player
    :param mocap_type: motion capture type
    :param dom_hand: dominant hand for player
    :param num_features: number of features to show on violin plot
    :return: plotly violin figure
    """
    if num_features > 8:
        print(f"Please reduce number of features to <= 8 for AWS size limitations")
        return None

    mocap_type = handle_recommendation_mocap_type(metrics_df, mocap_type)
    
    args = {
        "metrics_df": metrics_df,
        "movement_type": movement_type,
        "mocap_type": mocap_type,
        "dom_hand": dom_hand,
        "num_features": num_features
    }
    payload = {"function_name": "recommendation_violin_plot", "args": args}
    payload = json.dumps(payload, default=ut.serialize)
    response = ut.invoke_lambda(
        session=session,
        lambda_function_name=Functions.BACKEND,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )

    payload = response["Payload"].read()
    if ut.lambda_has_error(response):
        print(f"Error in calculation")
        print(f"If getting payload too large error, please try reducing num_features")
        print(payload)
        return payload
        
    return plotly.io.from_json(json.loads(payload))
