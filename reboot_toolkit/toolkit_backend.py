from __future__ import annotations

import os

from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from itertools import repeat, chain
from typing import Optional, Union

import awswrangler as wr
import boto3
import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import pyarrow as pa

from rapidfuzz import fuzz
from tqdm import tqdm

from .datatypes import (
    PlayerMetadata,
    S3Metadata,
    MOVEMENT_TYPE_IDS_MAP,
    MovementType,
    DataType,
)
from .reboot_motion_api import RebootClient
from .utils import handle_lambda_invocation


def find_player_matches(
    s3_df: pd.DataFrame,
    name_to_match: str,
    match_threshold: float = 50.0,
    max_results: int = 5,
) -> pd.DataFrame:
    """
    Use the rapid fuzz library to find all players matching an input name above a threshold.

    :param s3_df: the s3 summary dataframe
    :param name_to_match: the name to use to find a player match
    :param match_threshold: the threshold above which to consider a successful match
    :param max_results: the max number of results to return
    :return: dataframe of matched results
    """

    name_to_match = name_to_match.lower()

    players_df = (
        s3_df[["first_name", "last_name", "org_player_id"]]
        .copy()
        .drop_duplicates(subset=["org_player_id"])
    )

    players_df["name"] = (
        players_df["first_name"].str.lower() + " " + players_df["last_name"].str.lower()
    )

    players_df["match"] = players_df["name"].transform(
        lambda x: fuzz.ratio(x, name_to_match)
    )

    matched_results_df = players_df.loc[
        players_df["match"] > match_threshold
    ].sort_values(by="match", ascending=False, ignore_index=True)

    if len(matched_results_df) > max_results:
        matched_results_df = matched_results_df.iloc[:max_results]

    return matched_results_df


def filter_df_on_custom_metadata(
    data_df: pd.DataFrame,
    custom_metadata_df: pd.DataFrame,
    play_id_col: str,
    metadata_col: str | None,
    vals_to_keep: list[Union[int, float, str]] | None,
    drop_extra_cols: bool = False,
):
    """
    Filter segment dataframe using a dataframe of custom metadata with a play ID column for merging.

    :param data_df: the segment dataframe
    :param custom_metadata_df: the custom dataframe with a play ID column
    :param play_id_col: the play ID column that can be merged with org_movement_id
    :param metadata_col: the metadata column to use for filtering
    :param vals_to_keep: list of values in the metadata column to keep after filtering
    :param drop_extra_cols: whether to drop all columns other than the column used for filtering
    :return: the filtered input dataframe
    """

    if metadata_col is not None and len(metadata_col) > 0 and drop_extra_cols:
        custom_metadata_df = custom_metadata_df[[play_id_col, metadata_col]].copy()

    data_df = data_df.merge(
        custom_metadata_df, left_on="org_movement_id", right_on=play_id_col, how="left"
    ).drop(
        columns=[
            col
            for col in custom_metadata_df.columns
            if col.lower().startswith("unnamed")
        ]
    )

    if vals_to_keep:
        return (
            data_df.loc[data_df[metadata_col].isin(vals_to_keep)]
            .copy()
            .reset_index(drop=True)
        )

    return data_df


def widget_interface(
    org_player_ids: tuple[str],
    session_nums: tuple[str],
    session_dates: tuple[str],
    session_date_start: date,
    session_date_end: date,
    year: int,
) -> dict:
    """Create a dict from kwargs input by an ipywidget."""
    return {
        "org_player_ids": list(org_player_ids) if org_player_ids else None,
        "session_nums": list(session_nums) if session_nums else None,
        "session_dates": list(session_dates) if session_dates else None,
        "session_date_start": session_date_start,
        "session_date_end": session_date_end,
        "year": None if year == 0 else year,
    }


def create_interactive_widget(s3_df: pd.DataFrame) -> widgets.VBox:
    """Create an interactive widget for selecting data from an S3 summary dataframe."""
    w1 = widgets.interactive(
        widget_interface,
        org_player_ids=widgets.SelectMultiple(
            options=sorted(list(s3_df["org_player_id"].unique())),
            description="Orgs Players",
            disabled=False,
        ),
        session_nums=widgets.SelectMultiple(
            options=sorted(
                list(s3_df["session_num"].unique())
                + list(s3_df["org_session_id"].dropna().unique())
            ),
            description="Session Nums",
            disabled=False,
        ),
        session_dates=widgets.SelectMultiple(
            options=sorted(list(s3_df["session_date"].astype(str).unique())),
            description="Dates",
            disabled=False,
        ),
        session_date_start=widgets.DatePicker(
            description="Start Range", disabled=False
        ),
        session_date_end=widgets.DatePicker(description="End Range", disabled=False),
        year=widgets.IntText(
            value=2024,
            min=2020,
            max=2025,
            step=1,
            description="Year (0 = All)",
            disabled=False,
        ),
    )
    w2 = widgets.Button(
        description="Submit",
        disabled=False,
        button_style="success",  # 'success', 'info', 'warning', 'danger' or ''
        tooltip="Submit",
        icon="check",  # (FontAwesome names without the `fa-` prefix)
    )
    w3 = widgets.Output(layout={"border": "1px solid black"})

    def f(_):
        with w3:
            w3.clear_output()
            print(w1.result)

    w2.on_click(f)
    return widgets.VBox(
        [
            widgets.Label(
                "Populate any subset of the filters. Rerun cell to clear selection"
            ),
            w1,
            w2,
            w3,
        ]
    )


def display_available_df_data(df: pd.DataFrame) -> None:
    """
    Display the data in each column of a pandas dataframe.

    :param df: the dataframe to be displayed
    """
    print("Available data...")
    print()

    for col in df.columns:
        print(col)
        print(df[col].unique())
        print()


def download_s3_summary_df(
    s3_metadata: S3Metadata, verbose: bool = True, save_local: bool = False
) -> pd.DataFrame:
    """
    Download the CSV that summarizes all the current data in S3 into a dataframe.

    :param s3_metadata: the S3Metadata object to use to download the S3 summary CSV
    :param verbose: whether to print available info from the dataframe
    :param save_local: whether to save the CSV as a local file
    :return: dataframe of all the data in S3
    """
    s3_summary_df = None
    status_local = "Not saving summary dataframe locally"

    if save_local:
        try:
            s3_summary_df = pd.read_csv("s3_summary.csv")
            status_local = "Loaded local summary dataframe"

        except FileNotFoundError:
            status_local = "Local summary dataframe not found"
            pass

    if s3_summary_df is None:
        s3_summary_df = wr.s3.read_csv(
            [f"s3://reboot-motion-{s3_metadata.org_id}/population/s3_summary.csv"],
            index_col=[0],
        )

    if (
        not s3_summary_df["s3_path_delivery"]
        .iloc[0]
        .endswith(s3_metadata.file_type.value)
    ):
        s3_summary_df["s3_path_delivery"] = (
            s3_summary_df["s3_path_delivery"] + s3_metadata.file_type.value
        )

    s3_summary_df["org_player_id"] = s3_summary_df["org_player_id"].astype("string")

    s3_summary_df["session_num"] = s3_summary_df["session_num"].astype("string")

    s3_summary_df["session_date"] = pd.to_datetime(s3_summary_df["session_date"])

    s3_summary_df["org_session_id"] = s3_summary_df["org_session_id"].astype("string")

    if verbose:
        if save_local:
            print(status_local)
        display_available_df_data(s3_summary_df)

    if save_local:
        s3_summary_df.to_csv("s3_summary.csv", index=False)

    return s3_summary_df


def add_to_mask(mask: pd.Series, df: pd.DataFrame, col: str, vals: list) -> pd.Series:
    """
    Add more criteria to the mask for a pandas dataframe based filtering where values are in a column.

    :param mask: the existing pandas mask to add to
    :param df: the pandas dataframe to mask
    :param col: the col to filter
    :param vals: the rows of the dataframe to return contain these values
    :return: a mask for a pandas dataframe
    """
    return mask & df[col].isin(vals)


def filter_s3_summary_df(
    player_metadata: PlayerMetadata, s3_df: pd.DataFrame, verbose: bool = True
) -> pd.DataFrame:
    """
    Filter the S3 summary dataframe for only rows that are associated with the input player metadata.

    :param player_metadata: the metadata to use to filter the dataframe
    :param s3_df: the s3 summary dataframe
    :param verbose: whether to display available info from the dataframe
    :return: the filtered dataframe that only includes rows related to the player metadata
    """

    mask = s3_df["mocap_type"].isin(player_metadata.s3_metadata.mocap_types) & (
        s3_df["movement_type"] == player_metadata.s3_metadata.movement_type
    )

    if player_metadata.org_player_ids:
        mask = add_to_mask(mask, s3_df, "org_player_id", player_metadata.org_player_ids)

    if player_metadata.session_dates:
        mask = add_to_mask(mask, s3_df, "session_date", player_metadata.session_dates)

    if player_metadata.session_nums:
        mask = mask & (
            s3_df["session_num"].isin(player_metadata.session_nums)
            | s3_df["org_session_id"].isin(player_metadata.session_nums)
        )

    if player_metadata.session_date_start is not None:
        mask = mask & (
            s3_df["session_date"] >= pd.Timestamp(player_metadata.session_date_start)
        )

    if player_metadata.session_date_end is not None:
        mask = mask & (
            s3_df["session_date"] <= pd.Timestamp(player_metadata.session_date_end)
        )

    if player_metadata.year is not None:
        mask = mask & (s3_df["year"] == player_metadata.year)

    return_df = s3_df.loc[mask].drop_duplicates(ignore_index=True)

    if verbose:
        display_available_df_data(return_df)

    return return_df


def list_available_s3_keys(org_id: str, df: pd.DataFrame) -> list[str]:
    """
    List all files associated with a dataframe that contains a column of 's3_path_delivery' values.

    :param org_id: the org ID to use in the delivery path
    :param df: the dataframe with a column of s3_path_delivery values
    :return: list of all available s3 file paths
    """
    s3_client = boto3.Session().client("s3")

    bucket = f"reboot-motion-{org_id}"

    all_files = []

    for s3_path_delivery in df["s3_path_delivery"]:
        print("s3 base path:", s3_path_delivery)

        key_prefix = "/".join(s3_path_delivery.split("s3://")[-1].split("/")[1:])
        objs = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)

        s3_files = sorted([obj["Key"] for obj in objs.get("Contents", [])])
        print("available s3 files:")
        print(s3_files)
        print()

        all_files.extend(s3_files)

    return all_files


def load_games_to_df_from_s3_paths(
    game_paths: list[str],
    add_ik_joints: bool = False,
    add_elbow_var_val: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    For a list of paths to the S3 folder of data for each game, load the data into a pandas dataframe.

    :param game_paths: list of paths to folders with data for a player
    :param add_ik_joints: whether to add joints necessary to run analyses dependent on IK
    :param add_elbow_var_val: whether to add elbow varus valgus columns set to 0 degrees
    :param verbose: whether to display status info as games are loading
    :return: dataframe of all data from all games
    """
    game_paths = sorted(list(set(game_paths)))

    all_games = []
    game_count = 0

    for game_path in game_paths:
        try:
            if "hitting-processed-series" in game_path:
                swing_filenames = wr.s3.list_objects(game_path)
                swing_dfs = []
                for swing_filename in swing_filenames:
                    swing_df = wr.s3.read_csv(
                        swing_filename, index_col=[0], use_threads=True
                    ).dropna(axis=1, how="all")

                    basepath = os.path.join(*swing_filename.split("/")[1:-2])
                    movement_id = os.path.basename(swing_filename)[:-8]
                    org_movement_id = movement_id.split("_")[-1]

                    if "time" not in swing_df.columns:
                        metrics_csv_filename = os.path.join(
                            "s3://",
                            basepath,
                            "hitting-processed-metrics",
                            f"{movement_id}_bhm.csv",
                        )
                        metrics_df = wr.s3.read_csv(metrics_csv_filename)

                        swing_df["time"] = (np.arange(swing_df.shape[0]) / 300).round(5)

                        if not np.isnan(metrics_df.loc[0, "impact_event"]):
                            end_index = int(metrics_df.loc[0, "impact_event"])
                        else:
                            end_index = int(metrics_df.loc[0, "peak_velocity_event"])

                        swing_df["time_from_swing_end"] = (
                            swing_df["time"].values - swing_df["time"].values[end_index]
                        ).round(5)
                        swing_df["rel_frame"] = (
                            np.arange(swing_df.shape[0]) - end_index
                        ).astype(int)

                        foot_down = int(metrics_df.loc[0, "foot_down_event"])
                        norm_time_step = 100 / (end_index - foot_down)

                        swing_df["norm_time"] = (
                            (np.arange(swing_df.shape[0]) - foot_down) * norm_time_step
                        ).round(5)

                        swing_df["org_movement_id"] = org_movement_id

                    swing_dfs.append(swing_df)

                current_game = pd.concat(swing_dfs)

            elif "hitting-processed-metrics" in game_path:
                swing_filenames = wr.s3.list_objects(game_path)
                current_game = wr.s3.read_csv(swing_filenames, use_threads=True).dropna(
                    axis=1, how="all"
                )

                if "org_movement_id" not in current_game.columns:
                    org_movement_ids = [
                        os.path.basename(swing_filename).split("_")[1]
                        for swing_filename in swing_filenames
                    ]
                    current_game["org_movement_id"] = org_movement_ids

            else:
                if "/metrics-" in game_path and not game_path[-1].isdigit():
                    try:
                        metrics_status = "Defaulting to v2 metrics"
                        current_game = wr.s3.read_csv(
                            f"{game_path}-v2-0-0", index_col=[0], use_threads=True
                        ).dropna(axis=1, how="all")

                    except wr.exceptions.NoFilesFound:
                        metrics_status = (
                            "No v2 metrics found, falling back to v1 metrics"
                        )
                        current_game = wr.s3.read_csv(
                            f"{game_path}-v1-0-0", index_col=[0], use_threads=True
                        ).dropna(axis=1, how="all")

                    if verbose:
                        print(metrics_status)

                else:
                    current_game = wr.s3.read_csv(
                        game_path, index_col=[0], use_threads=True
                    ).dropna(axis=1, how="all")

            supported_mocap_types = ["hawkeye", "hawkeyehfr"]

            session_date_idx_list = [
                i
                for i, s in enumerate(game_path.split("/"))
                if s.isnumeric() and (len(s) == 8)
            ]
            session_date_str = game_path.split("/")[session_date_idx_list[0]]
            current_game["session_date"] = pd.to_datetime(session_date_str)

            used_words = set(supported_mocap_types + [session_date_str])
            session_num_idx_list = [
                i
                for i, s in enumerate(game_path.split("/"))
                if s.isalnum()
                and s not in used_words
                and ((len(s) == 1) or (len(s) == 6) or (len(s) == 8))
            ]
            if 0 < len(session_num_idx_list) <= 2:
                current_game["session_num"] = game_path.split("/")[
                    session_num_idx_list[0]
                ]

            else:
                current_game["session_num"] = None

            mocap_type_list = [
                s
                for s in game_path.split("/")
                if any(mt in s for mt in supported_mocap_types)
            ]
            if mocap_type_list:
                current_game["mocap_type"] = mocap_type_list[0]

            else:
                current_game["mocap_type"] = None

            if add_ik_joints:
                if "time" in current_game.columns:
                    for coord in ("X", "Y", "Z"):
                        current_game[f"neck_{coord}"] = (
                            current_game[f"LSJC_{coord}"]
                            + current_game[f"RSJC_{coord}"]
                        ) / 2.0

                        current_game[f"pelvis_{coord}"] = (
                            current_game[f"LHJC_{coord}"]
                            + current_game[f"RHJC_{coord}"]
                        ) / 2.0

                        current_game[f"torso_{coord}"] = current_game[f"pelvis_{coord}"]

                        current_game[f"{coord.lower()}_translation"] = current_game[
                            f"pelvis_{coord}"
                        ]

                    if "Basketball_X" in current_game.columns:
                        current_game = current_game.rename(
                            columns={
                                "Basketball_X": "x_ball_translation",
                                "Basketball_Y": "y_ball_translation",
                                "Basketball_Z": "z_ball_translation",
                            }
                        )

                    # set the target joint angle for the elbow varus valgus degree of freedom
                    if add_elbow_var_val:
                        current_game["right_elbow_var"] = 0

                        current_game["left_elbow_var"] = 0

                    ik_status = "Added IK joints"

                else:
                    ik_status = "Attempted to add IK joints, but they cannot be added to dataframes without time"

                if verbose:
                    print(ik_status)

            all_games.append(current_game)
            game_count += 1

            if verbose:
                print("session_date:", current_game["session_date"].iloc[0])
                print("session_num:", current_game["session_num"].iloc[0])
                print(
                    "Loaded path:",
                    game_path,
                    "-",
                    game_count,
                    "out of",
                    len(game_paths),
                )
                print()

        except Exception as exc:
            print("Error reading path:", game_path)
            print("With exception:", exc)
            continue

    all_games_df = pd.concat(all_games).reset_index(drop=True)

    time_cols = ("time_from_max_hand", "time_from_max_height")

    if ("rel_frame" not in all_games_df.columns) and any(
        tc in all_games_df.columns for tc in time_cols
    ):
        time_col = next(tc for tc in time_cols if tc in all_games_df.columns)

        all_games_df["rel_frame"] = all_games_df[time_col].copy()
        all_games_df["rel_frame"] = all_games_df.groupby("org_movement_id")[
            "rel_frame"
        ].transform(get_relative_frame)

        if verbose:
            print("Created relative frame column")

    return all_games_df


def load_data_into_analysis_dict(
    player_metadata: PlayerMetadata,
    df: Optional[pd.DataFrame] = None,
    df_mean: Optional[pd.DataFrame] = None,
    df_std: Optional[pd.DataFrame] = None,
    segment_label: Optional[str] = None,
) -> dict:
    """
    Load data from multiple games in a summary dict for further analysis.

    :param player_metadata: the metadata used to create the analysis
    :param df: dataframe across multiple games, will be filtered to one game based on the play ID in the metadata
    :param df_mean: the desired mean dataframe, will be calculated if not provided
    :param df_std: the desired standard deviation dataframe, will be calculated if not provided
    :param segment_label: the desired label to use for the analysis segment
    :return: dict of data to be used for this segment of an analysis
    """
    print("Loading into dict player metadata:", player_metadata)

    analysis_dict = {
        "player_id": player_metadata.org_player_ids[0]
        if player_metadata.org_player_ids
        else None,
        "session_date": player_metadata.session_dates[0]
        if player_metadata.session_dates
        else None,
        "game_pk": player_metadata.session_nums[0]
        if player_metadata.session_nums
        else None,
        "play_guid": player_metadata.org_movement_id,
        "s3_prefix": player_metadata.s3_prefix,
        "eye_hand_multiplier": player_metadata.s3_metadata.handedness.eye_hand_multiplier,
        "segment_label": segment_label,
    }

    if df is None:
        print(
            "No df provided, downloading data using s3 prefix:",
            analysis_dict["s3_prefix"],
        )
        df = wr.s3.read_csv(analysis_dict["s3_prefix"], index_col=[0])

    if df_mean is None:
        analysis_dict["df_mean"] = (
            df.groupby("rel_frame").agg("mean", numeric_only=True).reset_index()
        )
        print("Aggregated mean from df")

    else:
        analysis_dict["df_mean"] = df_mean

    if df_std is None:
        analysis_dict["df_std"] = (
            df.groupby("rel_frame").agg("std", numeric_only=True).reset_index()
        )
        print("Aggregated std from df")

    else:
        analysis_dict["df_std"] = df_std

    if analysis_dict["play_guid"] is None:
        play_guid = list(df["org_movement_id"].unique())[0]
        analysis_dict["play_guid"] = play_guid

    else:
        play_guid = analysis_dict["play_guid"]

    analysis_dict["df"] = df.loc[df["org_movement_id"] == play_guid]

    return analysis_dict


def merge_data_df_with_s3_keys(
    data_df: pd.DataFrame, s3_keys: list[str]
) -> pd.DataFrame:
    """
    Add a column to a data df with the associated s3 key.

    :param data_df: the pandas dataframe with an org_movement_id column
    :param s3_keys: a list of s3 keys for files from the game
    :return: dataframe with an s3 key column added
    """

    id_path_df = pd.DataFrame(
        {
            "movement_num": key.split("/")[-1].split("_")[0],
            "org_movement_id": key.split("/")[-1].split("_")[1],
            "s3_key": key,
        }
        for key in s3_keys
    )

    id_path_df["movement_num"] = id_path_df["movement_num"].astype(int)

    data_df = data_df.merge(
        id_path_df, left_on="org_movement_id", right_on="org_movement_id", how="left"
    ).sort_values(by=["session_date", "session_num", "movement_num"], ignore_index=True)

    data_df["count"] = data_df.groupby(["session_date", "session_num"]).cumcount() + 1

    return data_df


def list_chunks(lst: list, n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_relative_frame(time_series: pd.Series) -> pd.Series:
    """Input a pandas series of floats, and return ranks with 0 as rank = 0 and negative numbers as negative ranks."""
    all_ranks = (
        time_series.where(time_series >= 0).rank(method="first") - 1
    )  # start at 0 instead of 1
    negative_ranks = time_series.where(time_series < 0).rank(
        method="first", ascending=False
    )
    all_ranks.update(-negative_ranks)
    return all_ranks


def get_available_joint_angles(analysis_dicts: list[dict]) -> list[str]:
    """Get available joint angle columns from a list of analysis dicts."""
    non_angle_col_prefixes = (
        "optim_",
        "time",
        "norm_time",
        "movement_id",
        "org_movement_id",
        "event",
        "rel_frame",
        "game_pk",
        "session_num",
        "session_date",
    )

    df_columns = list(analysis_dicts[0]["df"])

    ind_end = df_columns.index("time")

    return [
        jnt
        for jnt in df_columns[:ind_end]
        if not jnt.startswith(non_angle_col_prefixes)
        and not jnt.endswith(("_X", "_Y", "_Z", "_vel", "_acc"))
    ]


def filter_pop_df(
    mean_df: pd.DataFrame,
    std_df: pd.DataFrame,
    time_col: str,
    angle_cols: list[str],
    downsample: int = 1,
) -> pd.DataFrame:
    """
    Filter a population dataframe and combine the mean and std dataframes.

    :param mean_df: the population mean dataframe
    :param std_df: the population standard deviation dataframe
    :param time_col: the column to use for time
    :param angle_cols: the joint angle columns to use
    :param downsample: how many frames to skip when down sampling the dataframe
    :return: the filtered and down-sampled population dataframe
    """

    if len(mean_df) != len(std_df):
        raise IndexError("Mismatched frames between mean and std population data")

    df = mean_df[[time_col, *angle_cols]].copy()

    for angle_col in angle_cols:
        df[f"{angle_col}_std"] = std_df[angle_col].copy()

    df = df.iloc[::downsample, :]

    return df


def filter_analysis_dicts(
    analysis_dicts: list[dict],
    time_col: str,
    angle_cols: list[str],
    keep_df: bool = True,
    keep_df_mean: bool = True,
    downsample: int = 1,
) -> list[dict]:
    """
    Filter an analysis dict so that it is a smaller size for sending to AWS for analysis.

    :param analysis_dicts: list of segment dicts to analyze
    :param time_col: column to use for time in the analysis
    :param angle_cols: list of joint angle columns to analyze
    :param keep_df: whether to keep the 'df' key in the analysis dict
    :param keep_df_mean: whether to keep the 'df_mean' in the analysis dict
    :param downsample: how may frames to skip when down-sampling
    :return: list of filtered analysis dicts
    """
    joints = [
        "NECK",
        "LSJC",
        "RSJC",
        "LEJC",
        "REJC",
        "LWJC",
        "RWJC",
        "LHJC",
        "RHJC",
        "LKJC",
        "RKJC",
        "LAJC",
        "RAJC",
    ]

    keys_to_keep = {"play_guid", "player_id"}

    if keep_df:
        keys_to_keep.add("df")

    if keep_df_mean:
        keys_to_keep.add("df_mean")

    df_cols_to_keep = {
        time_col,
        "time_from_max_hand",
        "org_movement_id",
        "norm_time",
        "torso_rot",
        *angle_cols,
    }

    df_cols_to_keep.update(
        [f"{joint}_{dim}" for joint in joints for dim in ["X", "Y", "Z"]]
    )

    res = []

    for analysis_dict in analysis_dicts:
        filt_analysis_dict = {
            k: v for k, v in analysis_dict.items() if k in keys_to_keep
        }

        if keep_df:
            filt_analysis_dict["df"] = (
                filt_analysis_dict["df"]
                .filter(df_cols_to_keep, axis=1)
                .astype(np.float16, errors="ignore")
                .iloc[::downsample, :]
            )

        if keep_df_mean:
            filt_analysis_dict["df_mean"] = (
                filt_analysis_dict["df_mean"]
                .filter(df_cols_to_keep, axis=1)
                .astype(np.float16)
                .iloc[::downsample, :]
            )

        res.append(filt_analysis_dict)

    return res


def get_animation(
    session: boto3.Session,
    analysis_dicts: list[dict],
    pop_mean_df: pd.DataFrame,
    pop_std_df: pd.DataFrame,
    time_column: str,
    joint_angle: str,
    plot_joint_angle_mean: bool,
    frame_step: int = 25,
    downsample_data: int = 2,
) -> go.Figure:
    """
    Use the input data to retrieve the synchronized skeleton and joint angle animation from AWS.

    :param session: the input boto3 session
    :param analysis_dicts: list of analysis dicts to animate
    :param pop_mean_df: the population mean dataframe
    :param pop_std_df: the population standard deviation dataframe
    :param time_column: the column to use as the time
    :param joint_angle: the joint angle to plot
    :param plot_joint_angle_mean: whether to plot the joint angle mean or the joint angle for the specific pitch
    :param frame_step: the number of frames between animation time points
    :param downsample_data: by how much to downsample the data
    :return: the plotly animation figure
    """
    times = analysis_dicts[0]["df"][time_column].tolist()[:: int(frame_step)]

    pop_df = filter_pop_df(
        pop_mean_df,
        pop_std_df,
        time_column,
        [joint_angle],
        downsample=downsample_data,
    )

    filtered_analysis_dicts = filter_analysis_dicts(
        analysis_dicts,
        time_column,
        [joint_angle],
        keep_df=True,
        keep_df_mean=plot_joint_angle_mean,
        downsample=downsample_data,
    )

    args = {
        "analysis_dicts": filtered_analysis_dicts,
        "pop_df": pop_df,
        "times": times,
        "time_column": time_column,
        "joint_angle": joint_angle,
        "eye_hand_multiplier": analysis_dicts[0]["eye_hand_multiplier"],
        "plot_joint_angle_mean": plot_joint_angle_mean,
    }

    payload = {"function_name": "get_joint_angle_animation", "args": args}

    return plotly.io.from_json(handle_lambda_invocation(session, payload))


def get_joint_plot(
    session: boto3.Session,
    analysis_dicts: list[dict],
    pop_mean_df: pd.DataFrame,
    pop_std_df: pd.DataFrame,
    time_column: str,
    joint_angles: list[str],
    plot_colors: list[str] = (
        "rgb(31, 119, 180)",
        "rgb(255, 127, 14)",
        "rgb(44, 160, 44)",
        "rgb(214, 39, 40)",
        "rgb(148, 103, 189)",
    ),
    downsample_data: int = 2,
) -> go.Figure:
    """
    Plot a selection of joint angles on a plotly 2d scatter plot.

    :param session: the input boto3 session
    :param analysis_dicts: list of analysis dicts to animate
    :param pop_mean_df: the population mean dataframe
    :param pop_std_df: the population standard deviation dataframe
    :param time_column: the column to use as the time
    :param joint_angles: the list of joint angles to plot
    :param plot_colors: the colors to use in the plot
    :param downsample_data: by how much to downsample the input data
    :return: the 2d plotly joint plot
    """
    pop_df = filter_pop_df(
        pop_mean_df,
        pop_std_df,
        time_column,
        joint_angles,
        downsample=downsample_data,
    )

    filtered_analysis_dicts = filter_analysis_dicts(
        analysis_dicts,
        time_column,
        joint_angles,
        keep_df=False,
        keep_df_mean=True,
        downsample=downsample_data,
    )

    args = {
        "analysis_dicts": filtered_analysis_dicts,
        "pop_df": pop_df,
        "time_column": time_column,
        "joint_angles": joint_angles,
        "plot_colors": plot_colors,
    }

    payload = {"function_name": "get_joint_angle_plots", "args": args}

    return plotly.io.from_json(handle_lambda_invocation(session, payload))


def save_figs_to_html(
    figs: list[plotly.graph_objects.Figure],
    output_report_name: str = "report.html",
) -> None:
    """
    Save a list of plotly figures to an html report.

    :param figs: the list of plotly figures to put into the report.
    :param output_report_name: the output name of the report.
    """
    if os.path.exists(output_report_name):
        os.remove(output_report_name)
        print("Removed old report")

    html_header = """
        <html><head>
        <title>Reboot Motion, Inc.</title>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head><body>
    """

    html_footer = (
        "<div width='100%' align='left' style=\"padding:20px 0 20px 0; font-size: 12px; font-family: Verdana, sans-serif;\"> "
        "DISCLAIMER<br><br>This report is for general informational purposes only, it does not constitute the practice of medicine, "
        "nursing, or other professional health care services, including the giving of medical advice. "
        "The use of this information is at the risk of the user. The content is not intended to be a substitute for professional medical "
        "advice, diagnosis, or treatment. Users should not disregard or delay in obtaining medical advice for any medical condition they have "
        "and should seek the assistance of their health care professionals for any such conditions. "
        "Users should seek the advice of a medical professional before beginning any exercise or training regimen, "
        "including following any advice or tips depicted or implied by this report.<br><br>We cannot guarantee any outcomes "
        "from using our products, and your results may vary. The use of our information, products and services should be "
        "based on your own due diligence and you agree that our company is not liable for any injury, success, or failure of your performance "
        "that is directly or indirectly related to the purchase and/or use of our information, products and services."
        "<br><br>Copyright Reboot Motion, Inc 2023</div></body></html>"
    )

    with open(output_report_name, "a+") as f:
        f.write(html_header)

        for fig in figs:
            if fig is None:
                continue

            f.write(
                fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    auto_play=False,
                    default_height="95%",
                )
            )
        f.write(html_footer.encode("utf-8").decode("utf-8"))

    print(
        f"Now you can download the new {output_report_name} "
        f"file from the files tab in the left side bar (refresh the list and click the three dots)"
    )


def export_data(
    reboot_client: RebootClient,
    org_player_id: str,
    movement_type_enum: MovementType,
    data_type_enum: DataType,
    session_ids: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Export data using the Reboot API.

    :param reboot_client: RebootClient object
    :param org_player_id: the org player ID to download data for
    :param movement_type_enum: the movement type enum to download data for
    :param data_type_enum: the data type enum to download
    :param session_ids: list of session IDs to download
    :param verbose: whether to print progress
    :return: dataframe of returned data
    """
    data_format = "parquet"
    aggregate = True
    return_column_info = False
    return_data = True
    as_pyarrow = True
    use_threads = False

    with ThreadPoolExecutor() as executor:
        pyarrow_tables = list(
            tqdm(
                executor.map(
                    reboot_client.data_exports.create,
                    session_ids,
                    repeat(MOVEMENT_TYPE_IDS_MAP[movement_type_enum.value]),
                    repeat(org_player_id),
                    repeat(data_type_enum.value),
                    repeat(data_format),
                    repeat(aggregate),
                    repeat(return_column_info),
                    repeat(return_data),
                    repeat(as_pyarrow),
                    repeat(use_threads),
                ),
                total=len(session_ids),
                disable=not verbose,
            )
        )

    return pa.concat_tables(pyarrow_tables).to_pandas()


def add_offsets_from_metadata(
    data_df: pd.DataFrame, metadata_df: pd.DataFrame, movement_type_enum: MovementType
) -> pd.DataFrame:
    """
    Add the X, Y, and Z offsets from the metadata to the position columns of an existing dataframe.

    :param data_df: the data for adding the offsets
    :param metadata_df: the dataframe containing the offsets
    :param movement_type_enum: the movement type enum
    :return: the dataframe with the added offsets
    """
    accepted_movement_types = {"baseball-hitting"}

    if movement_type_enum.value not in accepted_movement_types:
        raise NotImplementedError(
            "Adding offsets only implemented for movement types: {}, others coming soon!".format(
                accepted_movement_types
            )
        )

    if not set(data_df["org_movement_id"]).issubset(
        set(metadata_df["org_movement_id"])
    ):
        raise IndexError(
            "All org movement ids in data_df must be found in metadata_df, "
            "but some were not found: "
            f"{list(set(data_df['org_movement_id']).difference(set(metadata_df['org_movement_id'])))}"
        )

    base_cols = list(data_df)

    offset_cols = [
        "X_offset",
        "Y_offset",
        "Z_offset",
    ]

    data_df = data_df.merge(
        metadata_df[["org_movement_id"] + offset_cols], on="org_movement_id", how="left"
    )

    for coord in ("X", "Y", "Z"):
        coord_cols = [c for c in base_cols if c.endswith(f"_{coord}")]

        data_df[coord_cols] = data_df[coord_cols].to_numpy() + np.tile(
            np.expand_dims(data_df[f"{coord}_offset"].to_numpy(), 1),
            (1, len(coord_cols)),
        )

    data_df.drop(columns=offset_cols, inplace=True)

    return data_df
