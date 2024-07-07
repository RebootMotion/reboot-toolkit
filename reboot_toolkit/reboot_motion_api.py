from __future__ import annotations

import gzip
import json
import os

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import repeat

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.parquet as pq
import requests
import warnings


def locals_to_input(local_vars: dict) -> dict:
    """Remove from dict where key is 'self' or value is None"""
    return {k: v for k, v in local_vars.items() if k != "self" and v is not None}


def read_table_from_url(download_url: str, data_format: str) -> pa.Table:
    """Read a parquet or csv table with pyarrow from a url."""
    if data_format == "parquet" and ".parquet" not in download_url:
        raise ValueError("data_format is parquet, but .parquet not in download_url")

    elif data_format == "csv" and ".csv" not in download_url:
        raise ValueError("data_format is csv, but .csv not in download_url")

    downloaded_bytes = requests.get(download_url).content

    if any(suffix in download_url for suffix in (".parquet.gz", ".csv.gz")):
        downloaded_bytes = gzip.decompress(downloaded_bytes)

    if data_format == "parquet":
        pa_table = pq.read_table(BytesIO(downloaded_bytes))

    elif data_format == "csv":
        pa_table = csv.read_csv(BytesIO(downloaded_bytes))

    else:
        raise NotImplementedError("data_format {} is not supported".format(data_format))

    return pa_table


class RebootApi(object):
    def __init__(
        self,
        api_key: str | None = None,
        init_open: bool = False,
        default_query_limit: int = 100,
    ):
        """
        Initialize the reboot motion api with an api key and default headers.
        Must open the reboot api with RebootApi.open() to make requests, or use RebootApi as a context manager.

        :param api_key: the api key to use, will default to REBOOT_API_KEY environment variable if not set
        :param default_query_limit: the query limit to use as a default for all query string parameters
        :param init_open: whether to open the reboot api with RebootApi.open() upon first creation
        """
        self.base_url = "https://api.rebootmotion.com/"
        self.api_key = api_key or os.environ["REBOOT_API_KEY"]
        self.headers = {"x-api-key": self.api_key}
        self.default_query_limit = default_query_limit

        if init_open:
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

    def open(self) -> None:
        """Open the requests session and set default values."""
        self.requests_session = requests.Session()
        self.requests_session.headers.update(self.headers)
        self.requests_session.params = {"limit": self.default_query_limit}

    def close(self) -> None:
        """Close the requests session."""
        self.requests_session.close()
        self.requests_session = None

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
        return_data: bool = False,
        as_pyarrow: bool = False,
        use_threads: bool = False,
    ) -> dict | list | pa.Table:
        """
        Create a data export request and optionally download the resulting data if 'return_data' is True.

        'data_format' must be 'parquet' or 'csv'.

        Can return a pyarrow table if 'as_pyarrow' is True, which can be converted to a pandas or polars DataFrame.

        See https://api.rebootmotion.com/docs for full documentation.

        :return: either the reboot api response if return_data is False,
        or a list of date record dicts if 'return_data' is True and 'as_pyarrow' is False,
        or a pyarrow table if 'return_data' is True and 'as_pyarrow' is True.
        """
        accepted_data_formats = {"parquet", "csv"}

        if data_format not in accepted_data_formats:
            raise ValueError("data_format must be parquet or csv")

        if return_column_info and (return_data or as_pyarrow):
            raise NotImplementedError(
                "Cannot set return_column_info as True with either return_data or as_pyarrow as True"
            )

        if aggregate and use_threads:
            warnings.warn(
                "aggregate and use_threads are both True, "
                "but aggregate always returns just one download url so threading is not necessary"
            )

        if not return_data and as_pyarrow:
            warnings.warn(
                "as_pyarrow is True, but return_data is False, "
                "setting return_data to True so pyarrow data is returned"
            )
            return_data = True

        local_vars_for_api = {
            "session_id",
            "movement_type_id",
            "org_player_id",
            "data_type",
            "data_format",
            "aggregate",
            "return_column_info",
        }

        response = self._request(
            method="post",
            route="data_export",
            input_json=locals_to_input(
                {k: v for k, v in locals().items() if k in local_vars_for_api}
            ),
        )

        if not return_data:
            return response

        elif not isinstance(response, dict) or "download_urls" not in response:
            raise FileNotFoundError(
                "data export failed - expected download_urls not in response: {}".format(
                    response
                )
            )

        if use_threads:
            with ThreadPoolExecutor() as executor:
                pyarrow_tables = list(
                    executor.map(
                        read_table_from_url,
                        response["download_urls"],
                        repeat(data_format),
                    )
                )

        else:
            pyarrow_tables = [
                read_table_from_url(download_url, data_format)
                for download_url in response["download_urls"]
            ]

        pyarrow_table = pa.concat_tables(pyarrow_tables)

        if as_pyarrow:
            return pyarrow_table

        return pyarrow_table.to_pylist()
