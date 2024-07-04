import os

import reboot_toolkit as rtk

from reboot_toolkit import (
    FileType,
    Handedness,
    MocapType,
    MovementType,
    PlayerMetadata,
    S3Metadata,
)


def main():
    """Demo main script showcasing functionality"""

    verbose = False
    year = int(os.environ["YEAR"])
    org_player_ids = [os.environ["ORG_PLAYER_ID"]]
    n_recent_games_to_load = 2

    mocap_types = [rtk.MocapType.HAWKEYE_HFR, rtk.MocapType.HAWKEYE]
    movement_type = MovementType.BASEBALL_HITTING
    handedness = Handedness.LEFT
    file_type = FileType.INVERSE_KINEMATICS

    rtk.setup_aws(verbose=verbose)

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
    s3_df = rtk.download_s3_summary_df(s3_metadata, verbose=verbose)

    primary_segment_summary_df = rtk.filter_s3_summary_df(
        primary_analysis_segment, s3_df, verbose=verbose
    )

    s3_paths_games = primary_segment_summary_df["s3_path_delivery"].tolist()[
        -n_recent_games_to_load:
    ]

    print("Downloading data for desired games...")
    games_df = rtk.load_games_to_df_from_s3_paths(s3_paths_games, verbose=verbose)

    print("Done!")
    print(games_df)


if __name__ == "__main__":
    main()
