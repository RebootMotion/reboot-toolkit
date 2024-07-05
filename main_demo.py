import os

import pandas as pd

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

    year = int(os.environ["YEAR"])
    org_player_id = os.environ["ORG_PLAYER_ID"]
    n_recent_games_to_load = 2

    org_player_ids = [org_player_id]
    mocap_types = [MocapType.HAWKEYE_HFR, MocapType.HAWKEYE]
    movement_type_enum = MovementType.BASEBALL_HITTING
    handedness = Handedness.LEFT
    file_type = FileType.INVERSE_KINEMATICS

    rtk.setup_aws(verbose=verbose)

    s3_metadata = S3Metadata(
        org_id=os.environ["ORG_ID"],
        mocap_types=mocap_types,
        movement_type=movement_type_enum,
        handedness=handedness,
        file_type=file_type,
    )

    primary_analysis_segment = PlayerMetadata(
        org_player_ids=org_player_ids,
        year=year,
        s3_metadata=s3_metadata,
    )

    print("Downloading summary dataframe...")
    s3_df = rtk.download_s3_summary_df(
        s3_metadata, verbose=verbose, save_local=save_local
    )

    print("Creating primary segment_df...")
    primary_segment_summary_df = (
        rtk.filter_s3_summary_df(primary_analysis_segment, s3_df, verbose=verbose)
        .sort_values("session_date", ascending=True)
        .iloc[-n_recent_games_to_load:]
    )

    session_ids = list(primary_segment_summary_df["session_id"].unique())
    s3_paths_games = list(primary_segment_summary_df["s3_path_delivery"].unique())

    print("Downloading metadata...")
    with RebootApi() as reboot_api:
        metadata_df = rtk.export_data(
            reboot_api,
            org_player_id,
            movement_type_enum.value,
            "metadata",
            session_ids,
            verbose=verbose,
        )
    # metadata_df.to_parquet("metadata.parquet")
    # metadata_df = pd.read_parquet("metadata.parquet")
    print(metadata_df)

    print("Downloading data...")
    games_df = rtk.load_games_to_df_from_s3_paths(s3_paths_games, verbose=verbose)
    games_df.to_parquet("games.parquet")
    games_df = pd.read_parquet("games.parquet")
    print(games_df)

    print(games_df["RWJC_Z"])

    games_df = rtk.add_offsets_from_metadata(games_df, metadata_df)

    print(games_df["RWJC_Z"])


if __name__ == "__main__":
    main()
