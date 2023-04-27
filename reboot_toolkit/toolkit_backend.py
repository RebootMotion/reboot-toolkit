import awswrangler as wr
import boto3
import ipywidgets as widgets
import json
import numpy as np
import os
import pandas as pd
import plotly

from datetime import date
from typing import Optional, Union
from collections.abc import Generator
from . import utils as ut
from .datatypes import Functions, InvocationTypes, S3Metadata, PlayerMetadata


def filter_df_on_custom_metadata(
        data_df: pd.DataFrame,
        custom_metadata_df: pd.DataFrame,
        play_id_col: str,
        metadata_col: str,
        vals_to_keep: list[Union[int, float, str]]
):
    """
    Filter segment dataframe using a dataframe of custom metadata with a play ID column for merging.

    :param data_df: the segment dataframe
    :param custom_metadata_df: the custom dataframe with a play ID column
    :param play_id_col: the play ID column that can be merged with org_movement_id
    :param metadata_col: the metadata column to use for filtering
    :param vals_to_keep: list of values in the metadata column to keep after filtering
    :return: the filtered input dataframe
    """

    data_df = data_df.merge(
        custom_metadata_df[[play_id_col, metadata_col]], left_on='org_movement_id', right_on=play_id_col, how='left'
    )

    return data_df.loc[data_df[metadata_col].isin(vals_to_keep)].reset_index(drop=True)


def widget_interface(
        org_player_ids: tuple[str],
        session_nums: tuple[str],
        session_dates: tuple[str],
        session_date_start: date,
        session_date_end: date,
        year: int
) -> dict:
    """Create a dict from kwargs input by an ipywidget."""
    return {
        "org_player_ids": list(org_player_ids) if org_player_ids else None,
        "session_nums": list(session_nums) if session_nums else None,
        "session_dates": list(session_dates) if session_dates else None,
        "session_date_start": session_date_start,
        "session_date_end": session_date_end,
        "year": None if year == 0 else year
    }


def create_interactive_widget(s3_df: pd.DataFrame) -> widgets.widgets.interaction.interactive:
    """Create an interactive widget for selecting data from an S3 summary dataframe."""
    return widgets.interactive(
        widget_interface,
        org_player_ids=widgets.SelectMultiple(
            options=sorted(list(s3_df['org_player_id'].unique())), description='Orgs Players', disabled=False
        ),
        session_nums=widgets.SelectMultiple(
            options=sorted(list(s3_df['session_num'].unique())), description='Session Nums', disabled=False
        ),
        session_dates=widgets.SelectMultiple(
            options=sorted(list(s3_df['session_date'].astype(str).unique())), description='Dates', disabled=False
        ),
        session_date_start=widgets.DatePicker(
            description='Start Range', disabled=False
        ),
        session_date_end=widgets.DatePicker(
            description='End Range', disabled=False
        ),
        year=widgets.IntText(
            value=2023, min=2020, max=2024, step=1, description='Year (0 = All)', disabled=False
        )
    )


def display_available_df_data(df: pd.DataFrame) -> None:
    """
    Display the data in each column of a pandas dataframe.

    :param df: the dataframe to be displayed
    """
    print('Available data...')
    print()

    for col in df.columns:
        print(col)
        print(df[col].unique())
        print()


def download_s3_summary_df(s3_metadata: S3Metadata) -> pd.DataFrame:
    """
    Download the CSV that summarizes all the current data in S3 into a dataframe.

    :param s3_metadata: the S3Metadata object to use to download the S3 summary CSV
    :return: dataframe of all the data in S3
    """

    s3_summary_df = wr.s3.read_csv(
        [f"s3://reboot-motion-{s3_metadata.org_id}/population/s3_summary.csv"], index_col=[0]
    )

    s3_summary_df['s3_path_delivery'] = s3_summary_df['s3_path_delivery'] + s3_metadata.file_type.value + '/'

    s3_summary_df['org_player_id'] = s3_summary_df['org_player_id'].astype('string')

    s3_summary_df['session_num'] = s3_summary_df['session_num'].astype('string')

    s3_summary_df['session_date'] = pd.to_datetime(s3_summary_df['session_date'])

    display_available_df_data(s3_summary_df)

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


def filter_s3_summary_df(player_metadata: PlayerMetadata, s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the S3 summary dataframe for only rows that are associated with the input player metadata.

    :param player_metadata: the metadata to use to filter the dataframe
    :param s3_df: the s3 summary dataframe
    :return: the filtered dataframe that only includes rows related to the player metadata
    """

    mask = s3_df['mocap_type'].isin(player_metadata.s3_metadata.mocap_types) \
        & (s3_df['movement_type'] == player_metadata.s3_metadata.movement_type)

    if player_metadata.org_player_ids:
        mask = add_to_mask(mask, s3_df, 'org_player_id', player_metadata.org_player_ids)

    if player_metadata.session_dates:
        mask = add_to_mask(mask, s3_df, 'session_date', player_metadata.session_dates)

    if player_metadata.session_nums:
        mask = add_to_mask(mask, s3_df, 'session_num', player_metadata.session_nums)

    if player_metadata.session_date_start is not None:
        mask = mask & (s3_df['session_date'] >= pd.Timestamp(player_metadata.session_date_start))

    if player_metadata.session_date_end is not None:
        mask = mask & (s3_df['session_date'] <= pd.Timestamp(player_metadata.session_date_end))

    if player_metadata.year is not None:
        mask = mask & (s3_df['year'] == player_metadata.year)

    return_df = s3_df.loc[mask]

    display_available_df_data(return_df)

    return return_df


def list_available_s3_keys(org_id: str, df: pd.DataFrame) -> list[str]:
    """
    List all files associated with a dataframe that contains a column of 's3_path_delivery' values.

    :param org_id: the org ID to use in the delivery path
    :param df: the dataframe with a column of s3_path_delivery values
    :return: list of all available s3 file paths
    """
    s3_client = boto3.session.Session().client("s3")

    bucket = f"reboot-motion-{org_id}"

    all_files = []

    for s3_path_delivery in df['s3_path_delivery']:
        print('s3 base path:', s3_path_delivery)

        key_prefix = '/'.join(s3_path_delivery.split('s3://')[-1].split('/')[1:])
        objs = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)

        s3_files = sorted([obj['Key'] for obj in objs.get('Contents', [])])
        print('available s3 files:')
        print(s3_files)
        print()

        all_files.extend(s3_files)

    return all_files


def load_games_to_df_from_s3_paths(game_paths: list[str]) -> pd.DataFrame:
    """
    For a list of paths to the S3 folder of data for each game, load the data into a pandas dataframe.

    :param game_paths: list of paths to folders with data for a player
    :return: dataframe of all data from all games
    """
    game_paths = sorted(list(set(game_paths)))

    all_games = []

    for i, game_path in enumerate(game_paths):

        try:
            current_game = wr.s3.read_csv(game_path, index_col=[0], use_threads=True).dropna(axis=1, how='all')

            current_game['session_date'] = pd.to_datetime(game_path.split('/')[-6])
            print(current_game['session_date'].iloc[0])

            current_game['session_num'] = game_path.split('/')[-5]
            print(current_game['session_num'].iloc[0])

            all_games.append(current_game)

            print('Loaded path:', game_path, '-', i + 1, 'out of', len(game_paths))

        except Exception as exc:
            print('Error reading path', game_path, exc)
            continue

    all_games_df = pd.concat(all_games).reset_index(drop=True)

    if ('rel_frame' not in all_games_df.columns) and ('time_from_max_hand' in all_games_df.columns):
        print('Creating relative frame column...')
        all_games_df['rel_frame'] = all_games_df['time_from_max_hand'].copy()
        all_games_df['rel_frame'] = all_games_df.groupby('org_movement_id')['rel_frame'].transform(get_relative_frame)
        print('Done!')

    return all_games_df


def load_data_into_analysis_dict(
        player_metadata: PlayerMetadata,
        df: Optional[pd.DataFrame] = None,
        df_mean: Optional[pd.DataFrame] = None,
        df_std: Optional[pd.DataFrame] = None,
        segment_label: Optional[str] = None
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
    print('Loading into dict player metadata:', player_metadata)

    analysis_dict = {
        'player_id': player_metadata.org_player_ids[0] if player_metadata.org_player_ids else None,
        'session_date': player_metadata.session_dates[0] if player_metadata.session_dates else None,
        'game_pk': player_metadata.session_nums[0] if player_metadata.session_nums else None,
        'play_guid': player_metadata.org_movement_id,
        's3_prefix': player_metadata.s3_prefix,
        'eye_hand_multiplier': player_metadata.s3_metadata.handedness.eye_hand_multiplier,
        'segment_label': segment_label
    }

    if df is None:
        print('No df provided, downloading data using s3 prefix:', analysis_dict['s3_prefix'])
        df = wr.s3.read_csv(analysis_dict['s3_prefix'], index_col=[0])

    if df_mean is None:
        analysis_dict['df_mean'] = df.groupby('rel_frame').agg('mean', numeric_only=True).reset_index()
        print('Aggregated mean from df')

    else:
        analysis_dict['df_mean'] = df_mean

    if df_std is None:
        analysis_dict['df_std'] = df.groupby('rel_frame').agg('std', numeric_only=True).reset_index()
        print('Aggregated std from df')

    else:
        analysis_dict['df_std'] = df_std

    if analysis_dict['play_guid'] is None:
        play_guid = list(df['org_movement_id'].unique())[0]
        analysis_dict['play_guid'] = play_guid

    else:
        play_guid = analysis_dict['play_guid']

    analysis_dict['df'] = df.loc[df['org_movement_id'] == play_guid]

    return analysis_dict


def merge_data_df_with_s3_keys(data_df: pd.DataFrame, s3_keys: list[str]) -> pd.DataFrame:
    """
    Add a column to a data df with the associated s3 key.

    :param data_df: the pandas dataframe with an org_movement_id column
    :param s3_keys: a list of s3 keys for files from the game
    :return: dataframe with an s3 key column added
    """

    id_path_df = pd.DataFrame(
        {
            'movement_num': key.split('/')[-1].split('_')[0],
            'org_movement_id': key.split('/')[-1].split('_')[1],
            's3_key': key,
        } for key in s3_keys
    )

    id_path_df['movement_num'] = id_path_df['movement_num'].astype(int)

    return data_df.merge(id_path_df, left_on='org_movement_id', right_on='org_movement_id', how='left')


def list_chunks(lst: list, n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


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
        "session_date"
    )
    joint_angle_names = [
        jnt
        for jnt in analysis_dicts[0]["df"].columns
        if not jnt.startswith(non_angle_col_prefixes)
        and not jnt.endswith(("_X", "_Y", "_Z", "_vel", "_acc"))
    ]
    return joint_angle_names


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

    assert len(mean_df) == len(std_df), "Mismatched frames between mean and std population data"

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


def handle_lambda_invocation(session: boto3.session.Session, payload: dict) -> str:
    """
    Invoke a lambda function with the input payload.

    :param session: the boto3 session info to use
    :param payload: the lambda payload
    :return: the serialized lambda response
    """
    payload = json.dumps(payload, default=ut.serialize)

    print('Sending to AWS...')
    response = ut.invoke_lambda(
        session=session,
        lambda_function_name=Functions.VISUALIZATIONS,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )

    print('Reading Response...')
    payload = response["Payload"].read()

    if ut.lambda_has_error(response):
        print(f"Error in calculation")
        print(payload)
        return payload

    print("Returning Plotly figure...")
    return plotly.io.from_json(json.loads(payload))


def get_animation(
        session: boto3.session.Session,
        analysis_dicts: list[dict],
        pop_mean_df: pd.DataFrame,
        pop_std_df: pd.DataFrame,
        time_column: str,
        joint_angle: str,
        plot_joint_angle_mean: bool,
        frame_step: int = 25,
        downsample_data: int = 2,
) -> str:
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
    :return: the serialized plotly animation figure
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
        "eye_hand_multiplier": analysis_dicts[0]['eye_hand_multiplier'],
        "plot_joint_angle_mean": plot_joint_angle_mean,
    }

    payload = {"function_name": "get_joint_angle_animation", "args": args}

    return handle_lambda_invocation(session, payload)


def get_joint_plot(
        session: boto3.session.Session,
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
) -> str:
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
    :return: the serialized 2d plotly joint plot
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

    return handle_lambda_invocation(session, payload)


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
