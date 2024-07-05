import json
import os
import requests

from .datatypes import PlayerMetadata


def get_request(
    request_uri: str, headers: dict, params: dict | None = None, timeout: int = 60
) -> dict | list:
    """
    Make a get request for an API.

    :param request_uri: the uri for the request
    :param headers: the headers for the request
    :param params: dict of params for the request
    :param timeout: seconds to wait for the request
    :return: json response dict
    """
    response = requests.get(
        request_uri,
        params=params if isinstance(params, dict) else {},
        headers=headers,
        timeout=timeout,
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


class RebootApi(object):
    def __init__(self, api_key: str | None = None, default_query_limit: int = 100):
        self.base_url = "https://api.rebootmotion.com/"
        self.api_key = api_key or os.environ["REBOOT_API_KEY"]
        self.headers = {"x-api-key": self.api_key}
        self.default_query_limit = default_query_limit

    def get_mocap_types(self) -> dict:
        return {
            mocap["slug"]: mocap["id"]
            for mocap in get_request(
                request_uri=f"{self.base_url}/mocap_types",
                headers=self.headers,
            )
        }

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
        if limit is None:
            limit = self.default_query_limit

        params = {k: v for k, v in locals().items() if k != "self" and v is not None}

        return get_request(
            request_uri=f"{self.base_url}/sessions",
            headers=self.headers,
            params=params,
        )
