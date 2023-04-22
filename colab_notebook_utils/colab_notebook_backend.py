import awswrangler as wr
import boto3
import json
import numpy as np
import os
import pandas as pd
import plotly

from typing import Any, Dict, Generator, List, Optional
from . import utils as ut
from .datatypes import Functions, InvocationTypes, S3Metadata, PlayerMetadata


def download_s3_summary_df(s3_metadata: S3Metadata) -> pd.DataFrame:

    s3_summary_df = wr.s3.read_csv(
        [f"s3://reboot-motion-{s3_metadata.org_id}/population/s3_summary.csv"], index_col=[0]
    )

    s3_summary_df['s3_path_delivery'] = s3_summary_df['s3_path_delivery'] + s3_metadata.file_type + '/'

    s3_summary_df['org_player_id'] = s3_summary_df['org_player_id'].astype('string')

    s3_summary_df['session_num'] = s3_summary_df['session_num'].astype('string')

    s3_summary_df['session_date'] = pd.to_datetime(s3_summary_df['session_date'])

    return s3_summary_df


def add_to_mask(mask, df, col, vals):
    return mask & df[col].isin(vals)


def filter_s3_summary_df(player_metadata: PlayerMetadata, s3_df: pd.DataFrame) -> pd.DataFrame:
    mask = (s3_df['mocap_type'] == player_metadata.s3_metadata.mocap_type) \
           & (s3_df['movement_type'] == player_metadata.s3_metadata.movement_type)

    if player_metadata.mlbam_player_ids:
        mask = add_to_mask(mask, s3_df, 'org_player_id', player_metadata.mlbam_player_ids)

    if player_metadata.session_dates:
        mask = add_to_mask(mask, s3_df, 'session_date', player_metadata.session_dates)

    if player_metadata.game_pks:
        mask = add_to_mask(mask, s3_df, 'session_num', player_metadata.game_pks)

    if player_metadata.session_date_start is not None:
        mask = mask & (s3_df['session_date'] >= pd.Timestamp(player_metadata.session_date_start))

    if player_metadata.session_date_end is not None:
        mask = mask & (s3_df['session_date'] <= pd.Timestamp(player_metadata.session_date_end))

    if player_metadata.year is not None:
        mask = mask & (s3_df['year'] == player_metadata.year)

    return s3_df.loc[mask]


def list_available_s3_keys(org_id, primary_df):
    s3_client = boto3.session.Session().client("s3")

    bucket = f"reboot-motion-{org_id}"

    all_files = []

    for s3_path_delivery in primary_df['s3_path_delivery']:
        print('s3 base path', s3_path_delivery)

        key_prefix = '/'.join(s3_path_delivery.split('s3://')[-1].split('/')[1:])
        objs = s3_client.list_objects_v2(Bucket=bucket, Prefix=key_prefix)

        s3_files = sorted([obj['Key'] for obj in objs.get('Contents', [])])
        print(s3_files)
        print()

        all_files.extend(s3_files)

    return all_files


def load_games_from_df_with_s3_paths(df: pd.DataFrame) -> pd.DataFrame:
    all_games = []

    for i, game_path in enumerate(df['s3_path_delivery'].unique()):

        try:
            current_game = wr.s3.read_csv(game_path, index_col=[0], use_threads=True).dropna(axis=1, how='all')

            current_game['game_pk'] = game_path.split('/')[-5]

            all_games.append(current_game)

            print('read path:', game_path, ';', i + 1, 'out of', len(df))

        except Exception as exc:
            print('error reading path', game_path, exc)
            continue

    all_games_df = pd.concat(all_games).reset_index(drop=True)

    if 'rel_frame' not in all_games_df.columns:
        all_games_df['rel_frame'] = all_games_df['time_from_max_hand'].copy()
        all_games_df['rel_frame'] = all_games_df.groupby('org_movement_id')['rel_frame'].transform(get_relative_frame)

    return all_games_df


def load_data_into_analysis_dict(
        player_metadata: PlayerMetadata,
        df: Optional[pd.DataFrame] = None,
        df_mean: Optional[pd.DataFrame] = None,
        df_std: Optional[pd.DataFrame] = None
) -> dict:
    print('Loading player data into analysis dict', player_metadata)

    analysis_dict = {
        'mlbam_player_id': player_metadata.mlbam_player_ids[0] if player_metadata.mlbam_player_ids else None,
        'session_date': player_metadata.session_dates[0] if player_metadata.session_dates else None,
        'game_pk': player_metadata.game_pks[0] if player_metadata.game_pks else None,
        'mlb_play_guid': player_metadata.mlb_play_guid,
        's3_prefix': player_metadata.s3_prefix
    }

    if df is None:
        print('Downloading data...')
        df = wr.s3.read_csv(analysis_dict['s3_prefix'], index_col=[0])

    if df_mean is None:
        print('aggregating mean')
        analysis_dict['df_mean'] = df.groupby('rel_frame').agg('mean', numeric_only=True).reset_index()

    else:
        analysis_dict['df_mean'] = df_mean

    if df_std is None:
        print('aggregating std')
        analysis_dict['df_std'] = df.groupby('rel_frame').agg('std', numeric_only=True).reset_index()

    else:
        analysis_dict['df_std'] = df_std

    if analysis_dict['mlb_play_guid'] is None:
        mlb_play_guid = list(df['org_movement_id'].unique())[0]
        analysis_dict['mlb_play_guid'] = mlb_play_guid

    else:
        mlb_play_guid = analysis_dict['mlb_play_guid']

    analysis_dict['df'] = df.loc[df['org_movement_id'] == mlb_play_guid]

    return analysis_dict


def list_chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
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


def get_available_joint_angles(analysis_dicts: List[Dict]) -> List[str]:
    non_angle_col_prefixes = (
        "optim_",
        "time",
        "norm_time",
        "movement_id",
        "org_movement_id",
        "event",
        "rel_frame",
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
    angle_cols: List[str],
    downsample: int = 1,
) -> pd.DataFrame:
    assert len(mean_df) == len(
        std_df
    ), "Mismatched frames between mean and std population data"
    df = mean_df[[time_col, *angle_cols]].copy()
    for angle_col in angle_cols:
        df[f"{angle_col}_std"] = std_df[angle_col].copy()
    df = df.iloc[::downsample, :]
    return df


def filter_analysis_dicts(
    analysis_dicts: List[Dict],
    time_col: str,
    angle_cols: List[str],
    keep_df: bool = True,
    keep_df_mean: bool = True,
    downsample: int = 1,
) -> List[Dict]:
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
    keys_to_keep = {"mlb_play_guid", "mlbam_player_id"}
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
    session: boto3.session.Session,
    analysis_dicts: List[Dict[Any, Any]],
    pop_mean_df: pd.DataFrame,
    pop_std_df: pd.DataFrame,
    time_column: str,
    joint_angle: str,
    eye_hand_multiplier: int,
    plot_joint_angle_mean: bool,
    frame_step: int = 25,
    downsample_data: int = 2,
) -> str:
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
        "eye_hand_multiplier": eye_hand_multiplier,
        "plot_joint_angle_mean": plot_joint_angle_mean,
    }
    payload = {"function_name": "get_joint_angle_animation", "args": args}
    payload = json.dumps(payload, default=ut.serialize)
    response = ut.invoke_lambda(
        session=session,
        lambda_function_name=Functions.VISUALIZATIONS,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )
    payload = response["Payload"].read()
    if ut.lambda_has_error(response):
        print(f"Error in calculation")
        print(payload)
        return payload
    return plotly.io.from_json(json.loads(payload))


def get_joint_plot(
    session: boto3.session.Session,
    analysis_dicts: List[Dict[Any, Any]],
    pop_mean_df: pd.DataFrame,
    pop_std_df: pd.DataFrame,
    time_column: str,
    joint_angles: List[str],
    plot_colors: List[str] = [
        "rgb(31, 119, 180)",
        "rgb(255, 127, 14)",
        "rgb(44, 160, 44)",
        "rgb(214, 39, 40)",
        "rgb(148, 103, 189)",
    ],
    downsample_data: int = 2,
) -> str:
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
    payload = json.dumps(payload, default=ut.serialize)
    response = ut.invoke_lambda(
        session=session,
        lambda_function_name=Functions.VISUALIZATIONS,
        invocation_type=InvocationTypes.SYNC,
        lambda_payload=payload,
    )
    payload = response["Payload"].read()
    if ut.lambda_has_error(response):
        print(f"Error in calculation")
        print(payload)
        return payload
    return plotly.io.from_json(json.loads(payload))


def save_figs_to_html(
    figs: List[plotly.graph_objects.Figure],
    output_report_name: str = "report.html",
) -> None:
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
        f"Now you can download the new {output_report_name} file from the files tab in the left side bar (refresh the list and click the three dots)"
    )
