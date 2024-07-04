import os
import requests


def get_request(
    request_uri: str, headers: dict, params: dict | None = None, timeout: int = 60
) -> dict:
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
        params=params if params else {},
        headers=headers,
        timeout=timeout,
    )

    response.raise_for_status()

    return response.json()


class RebootApi(object):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ["REBOOT_API_KEY"]

        self.headers = {"x-api-key": self.api_key}

        self.base_url = "https://api.rebootmotion.com/"

    def get_sessions(self) -> dict:
        return get_request(
            request_uri=f"{self.base_url}/sessions",
            headers=self.headers,
            params=None,
        )
