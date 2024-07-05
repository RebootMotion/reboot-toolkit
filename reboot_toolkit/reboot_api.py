import json
import os

from gzip import decompress
from io import BytesIO, StringIO

import pandas as pd
import requests


def locals_to_input(local_vars: dict) -> dict:
    return {k: v for k, v in local_vars.items() if k != "self" and v is not None}


class RebootApi(object):
    def __init__(self, api_key: str | None = None, default_query_limit: int = 100):
        self.base_url = "https://api.rebootmotion.com/"
        self.api_key = api_key or os.environ["REBOOT_API_KEY"]
        self.headers = {"x-api-key": self.api_key}
        self.default_query_limit = default_query_limit

        self.requests_session = requests.Session()
        self.requests_session.headers.update(self.headers)
        self.requests_session.params = {"limit": default_query_limit}

    def _request(
        self,
        method: str,
        route: str,
        params: dict | list | None = None,
        data: dict | None = None,
        timeout: int | None = None,
        input_json: dict | list | None = None,
    ):
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

    def close(self) -> None:
        self.requests_session.close()

    def get_mocap_types(self, return_id_lookup: bool = True) -> dict:
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
    ):
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
