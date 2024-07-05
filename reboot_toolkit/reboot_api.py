import json
import os

from gzip import decompress
from io import BytesIO, StringIO

import pandas as pd
import requests


def locals_to_input(local_vars: dict) -> dict:
    """Remove from dict where key is 'self' or value is None"""
    return {k: v for k, v in local_vars.items() if k != "self" and v is not None}


class RebootApi(object):
    def __init__(
        self,
        api_key: str | None = None,
        default_query_limit: int = 100,
        is_open: bool = False,
    ):
        """
        Initialize the reboot motion api with an api key and default headers.
        Must open the reboot api with RebootApi.open() to make requests, or use RebootApi as a context manager.

        :param api_key: the api key to use, will default to REBOOT_API_KEY environment variable if not set
        :param default_query_limit: the query limit to use as a default for all query string parameters
        :param is_open: whether to open the reboot api with RebootApi.open() upon first creation
        """
        self.base_url = "https://api.rebootmotion.com/"
        self.api_key = api_key or os.environ["REBOOT_API_KEY"]
        self.headers = {"x-api-key": self.api_key}
        self.default_query_limit = default_query_limit

        if is_open:
            self.open()

        else:
            self.requests_session = None

    def __enter__(self) -> "RebootApi":
        """Open the requests session when entering a context manager."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_traceback):
        """Close the requests session when exiting a context manager."""
        self.close()

    def open(self) -> None:
        """Open the requests session and set default values."""
        self.requests_session = requests.Session()
        self.requests_session.headers.update(self.headers)
        self.requests_session.params = {"limit": self.default_query_limit}

    def close(self) -> None:
        """Close the requests session."""
        self.requests_session.close()

    def _request(
        self,
        method: str,
        route: str,
        params: dict | list | None = None,
        data: dict | None = None,
        timeout: int | None = None,
        input_json: dict | list | None = None,
    ):
        """
        Send a request to the reboot motion api via an open requests session.
        Important note, 'params' are used for a 'get' request and 'input_json' is used for a 'post' request.

        :param method: post, get, etc.
        :param route: the reboot motion api route from the base url
        :param params: optional query parameters, used for get requests
        :param data: optional input data
        :param timeout: optional timeout for the request
        :param input_json: optional input json dict, used for post requests
        :return: a response json object from the reboot motion api
        """
        if self.requests_session is None:
            raise RuntimeError(
                "Must call RebootApi.open() before making requests, or use RebootApi as a context manager"
            )

        response = self.requests_session.request(
            method=method,
            url=f"{self.base_url}/{route}",
            params=params,
            data=data,
            timeout=timeout,
            json=input_json,
        )

        try:
            response.raise_for_status()

        except requests.RequestException:
            try:
                print(response.json())

            except json.decoder.JSONDecodeError:
                print(response)

            raise

        return response.json()

    def get_mocap_types(self, return_id_lookup: bool = True) -> list | dict:
        """
        Return a list of mocap types, or a lookup of mocap type to mocap type id if 'return_id_lookup' is True.
        See https://api.rebootmotion.com/docs for full documentation.

        :return: list of mocap types or dict of mocap types to mocap type ids
        """
        mocap_type_response = self._request(
            method="get",
            route="mocap_types",
        )

        if return_id_lookup:
            return {mocap["slug"]: mocap["id"] for mocap in mocap_type_response}

        return mocap_type_response

    def get_sessions(
        self,
        created_at: list[str] | None = None,
        created_since: str | None = None,
        updated_at: list[str] | None = None,
        updated_since: str | None = None,
        session_type_id: int | None = None,
        mocap_type_id: int | None = None,
        session_date: list[str] | None = None,
        session_date_in: list[str] | None = None,
        session_date_since: str = None,
        status: str = None,
        org_player_ids: list[str] | None = None,
        official_game_ids: list[str] | None = None,
        session_num: int | None = None,
        offset: int | None = None,
        limit: int | None = None,
    ) -> dict:
        """
        Get a list of sessions from query parameters.
        See https://api.rebootmotion.com/docs for full documentation.

        :return: a list of sessions with session metadata
        """
        return self._request(
            method="get",
            route="sessions",
            params=locals_to_input(locals()),
        )

    def post_data_export(
        self,
        session_id: str,
        movement_type_id: int,
        org_player_id: str,
        data_type: str,
        data_format: str = "parquet",
        aggregate: bool = False,
        return_column_info: bool = False,
        return_data: bool = True,
    ) -> dict | list | pd.DataFrame:
        """
        Create a data export request and optionally download the resulting data if 'return_data' is True.
        See https://api.rebootmotion.com/docs for full documentation.

        :return: either the request or response, or the dataframe of resulting data
        """
        local_vars = locals()
        return_data = local_vars.pop("return_data")

        response = self._request(
            method="post",
            route="data_export",
            input_json=locals_to_input(local_vars),
        )

        if not return_data:
            return response

        if data_format == "parquet":
            data_list = [
                pd.read_parquet(BytesIO(requests.get(download_url).content))
                for download_url in response["download_urls"]
            ]

        elif data_format == "csv":
            data_list = [
                pd.read_csv(
                    StringIO(
                        decompress(requests.get(download_url).content).decode("utf-8")
                    )
                )
                for download_url in response["download_urls"]
            ]

        else:
            raise ValueError("data_format must be parquet or csv")

        return pd.concat(data_list, ignore_index=True)
