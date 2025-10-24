import json
import logging
import os
import time
from typing import Any, Dict, Generator, List
from uuid import uuid4

import boto3
import numpy as np
import pandas as pd
from botocore.errorfactory import ClientError

import utils as ut
from datatypes import Functions, InvocationTypes


def get_log_level() -> Any:
    level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper())
    return level


logger = logging.getLogger(__name__)
logger.setLevel(get_log_level())


BUCKET = "reboot-motion-toolbox-temp-files"


def file_exists(s3_client: Any, filename: str) -> bool:
    try:
        s3_client.head_object(Bucket=BUCKET, Key=filename)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # Object not available yet, 404 not found
            return False
        else:
            raise e
    return True


def gather_files(
    session: boto3.Session,
    filenames: List[str],
    poll_interval_s: int = 2,
    max_attempts: int = 120,
) -> None:
    s3_client = boto3.client("s3")
    missing_files = set(filenames)
    attempts_remaining = max_attempts
    while missing_files and attempts_remaining:
        logger.debug(f"{attempts_remaining} attempts remaining")

        found_files = set()
        for filename in filenames:
            if file_exists(s3_client, filename):
                found_files.add(filename)

        logger.debug(f"Found {found_files}")

        attempts_remaining -= 1
        missing_files -= found_files
        time.sleep(poll_interval_s)

    if missing_files:
        raise RuntimeError(f"Could not find files {missing_files}")
    logger.debug("Done gathering files")


def dispatch(session: boto3.Session) -> List[str]:
    filenames = []
    for i in range(3):
        filename = f"{uuid4()}.txt"
        args = {
            "filename": filename,
            "sleep_time": 10,
        }
        payload = {"function_name": "inverse_kinematics", "args": args}
        payload = json.dumps(payload, default=ut.serialize)
        response = ut.invoke_lambda(
            session=session,
            lambda_function_name=Functions.INVERSE_KINEMATICS,
            invocation_type=InvocationTypes.ASYNC,
            lambda_payload=payload,
        )

        assert (
            response["StatusCode"] == 202
        ), f"Returned with status code {response['StatusCode']}"
        logger.debug(f"Dispacted inverse kinematics thread {i} and filename {filename}")
        filenames.append(filename)
    return filenames


def inverse_kinematics(session: boto3.Session) -> str:
    # TODO partition problem
    filenames = dispatch(session)
    gather_files(session, filenames)
    # TODO Reassemble files
    # TODO Return
