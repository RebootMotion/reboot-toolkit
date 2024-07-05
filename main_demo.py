import os

import numpy as np
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
    MOVEMENT_TYPE_IDS,
)


def main():
    """Demo main script showcasing functionality"""

    verbose = False
    save_local = True
    default_query_limit = 10000

    year = int(os.environ["YEAR"])
    org_player_ids = [os.environ["ORG_PLAYER_ID"]]
    n_recent_games_to_load = 1

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
    s3_df = rtk.download_s3_summary_df(
        s3_metadata, verbose=verbose, save_local=save_local
    )

    print("Creating primary segment_df...")
    primary_segment_summary_df = (
        rtk.filter_s3_summary_df(primary_analysis_segment, s3_df, verbose=verbose)
        .sort_values("session_date", ascending=True)
        .iloc[-n_recent_games_to_load:]
    )

    print("Selecting most recent game...")
    row = primary_segment_summary_df.iloc[-1]
    s3_paths_games = [row["s3_path_delivery"]]

    print("Downloading metadata for most recent game...")
    metadata_df = reboot_api.post_data_export(
        row["session_id"],
        MOVEMENT_TYPE_IDS[row["movement_type"]],
        row["org_player_id"],
        data_type="metadata",
    )
    # metadata_df.to_parquet("metadata.parquet")
    # metadata_df = pd.read_parquet("metadata.parquet")

    print("Downloading data for desired games...")
    games_df = rtk.load_games_to_df_from_s3_paths(s3_paths_games, verbose=verbose)
    # games_df.to_parquet("games.parquet")
    # games_df = pd.read_parquet("games.parquet")

    print(games_df["RWJC_Z"])

    games_df = rtk.add_offsets_from_metadata(games_df, metadata_df)

    print(games_df["RWJC_Z"])


if __name__ == "__main__":
    main()
