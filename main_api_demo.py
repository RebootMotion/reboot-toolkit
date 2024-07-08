import requests

from dotenv import load_dotenv

from reboot_toolkit.reboot_motion_api import RebootClient

load_dotenv()


def main():
    default_query_limit = 150

    with requests.Session() as requests_session:
        reboot_client = RebootClient(
            requests_session=requests_session, default_query_limit=default_query_limit
        )

        mocap_types = reboot_client.mocap_types.list()
        print(mocap_types)

        sessions = reboot_client.sessions.list()
        print(len(sessions))

        for session in sessions:
            print(session["id"])
            break


if __name__ == "__main__":
    main()
