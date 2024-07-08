from __future__ import annotations

import gzip
import json
import os

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from itertools import repeat

import pyarrow as pa
import requests
import warnings


DEFAULT_API_BASE = "https://api.rebootmotion.com/"


def create_lookup(instance_list: list[dict]) -> dict:
    """Return a lookup from instance slug to instance ID from a list of object instances."""
    return {
        instance_dict["slug"]: instance_dict["id"] for instance_dict in instance_list
    }


def locals_to_input(local_vars: dict) -> dict:
    """Remove from dict where key is 'self' or value is None."""
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
        pa_table = pa.parquet.read_table(BytesIO(downloaded_bytes))

    elif data_format == "csv":
        pa_table = pa.csv.read_csv(
            BytesIO(downloaded_bytes),
            read_options=pa.csv.ReadOptions(use_threads=True),
        )

    else:
        raise NotImplementedError("data_format {} is not supported".format(data_format))

    return pa_table


class _APIRequestor(object):
    def __init__(self, base_url: str, requests_session: requests.Session):
        self.base_url = base_url
        self.requests_session = requests_session


class RebootService(object):
    def __init__(self, requestor: _APIRequestor):
        self._base_url = requestor.base_url
        self._requests_session = requestor.requests_session

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
        response = self._requests_session.request(
            method=method,
            url=f"{self._base_url}/{route}",
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


class MocapTypesService(RebootService):
    def __init__(self, requestor: _APIRequestor):
        super().__init__(requestor)

    def list(self, return_id_lookup: bool = False) -> list | dict:
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
            return create_lookup(mocap_type_response)

        return mocap_type_response


class SessionsService(RebootService):
    def __init__(self, requestor: _APIRequestor):
        super().__init__(requestor)

    def list(
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


class DataExportsService(RebootService):
    def __init__(self, requestor: _APIRequestor):
        super().__init__(requestor)

    def create(
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

        Can return a pyarrow table if 'as_pyarrow' is True, which can easily be used with Pandas or Polars.

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

        elif not isinstance(response, dict) or not response.get("download_urls"):
            raise FileNotFoundError(
                "data export failed - expected download_urls not in response: {}".format(
                    response
                )
            )

        elif len(response["download_urls"]) == 1:
            pyarrow_table = read_table_from_url(
                response["download_urls"][0], data_format
            )

        else:
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


class RebootClient(object):
    def __init__(
        self,
        api_key: str | None = None,
        requests_session: requests.Session | None = None,
        default_query_limit: int = 100,
    ):
        """
        Initialize the reboot motion api with required headers, a requests Session, and default request parameters.

        :param api_key: the api key to use, will default to REBOOT_API_KEY environment variable if not set
        :param requests_session: the requests.Session() to use to make requests, or None to use a default Session
        :param default_query_limit: the query limit to use as a default for all query string parameters
        """
        if requests_session is None:
            requests_session = requests.Session()

        requests_session.headers.update(
            {"x-api-key": api_key or os.environ["REBOOT_API_KEY"]}
        )
        requests_session.params = {"limit": default_query_limit}

        self._requestor = _APIRequestor(DEFAULT_API_BASE, requests_session)

        self.mocap_types = MocapTypesService(self._requestor)
        self.sessions = SessionsService(self._requestor)
        self.data_exports = DataExportsService(self._requestor)


# def main():
#     from dotenv import load_dotenv
#     load_dotenv()
#
#     reboot_client = RebootClient(default_query_limit=150)
#
#     sessions = reboot_client.sessions.list()
#     print(len(sessions))
#
#
# if __name__ == "__main__":
#     main()
