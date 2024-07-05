import os

import reboot_toolkit as rtk

from reboot_toolkit import (
    FileType,
    Handedness,
    MocapType,
    MovementType,
    PlayerMetadata,
    RebootApi,
    S3Metadata,
)


def main():
    """Demo main script showcasing functionality"""

    verbose = False
    save_local = True
    default_query_limit = 10000

    year = int(os.environ["YEAR"])
    org_player_ids = [os.environ["ORG_PLAYER_ID"]]
    n_recent_games_to_load = 2

    mocap_types = [MocapType.HAWKEYE_HFR, MocapType.HAWKEYE]
    movement_type = MovementType.BASEBALL_HITTING
    handedness = Handedness.LEFT
    file_type = FileType.INVERSE_KINEMATICS

    rtk.setup_aws(verbose=verbose)

    reboot_api = RebootApi(default_query_limit=default_query_limit)

    s3_metadata = S3Metadata(
        org_id=os.environ["ORG_ID"],
        mocap_types=mocap_types,
        movement_type=movement_type,
        handedness=handedness,
        file_type=file_type,
    )

    primary_analysis_segment = PlayerMetadata(
        org_player_ids=org_player_ids,
        year=year,
        s3_metadata=s3_metadata,
    )

    print("Downloading summary dataframe...")
    s3_df = rtk.download_s3_summary_df(s3_metadata, verbose=verbose, save_local=save_local)

    print("Creating primary segment_df...")
    primary_segment_summary_df = rtk.filter_s3_summary_df(
        primary_analysis_segment, s3_df, verbose=verbose
    ).sort_values("session_date", ascending=True).iloc[-n_recent_games_to_load:]
    print(primary_segment_summary_df)

    print("Downloading sessions...")

    # session_dates = [
    #     primary_segment_summary_df["session_date"].iloc[0].strftime("%Y-%m-%d"),
    #     primary_segment_summary_df["session_date"].iloc[-1].strftime("%Y-%m-%d")
    # ]
    # print(session_dates)
    # resp = reboot_api.get_sessions(session_date=session_dates)

    org_player_ids = list(primary_segment_summary_df["org_player_id"].unique())
    resp = reboot_api.get_sessions(org_player_ids=org_player_ids)

    for session in resp:
        print(session)

    # s3_paths_games = primary_segment_summary_df["s3_path_delivery"].tolist()
    # print(s3_paths_games)

    # print("Downloading data for desired games...")
    # games_df = rtk.load_games_to_df_from_s3_paths(s3_paths_games, verbose=verbose)
    #
    # print("Done!")
    # print(games_df)


if __name__ == "__main__":
    main()
