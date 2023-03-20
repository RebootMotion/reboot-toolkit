import json
import os
from typing import Any, Dict, Generator, List

import numpy as np
import pandas as pd
import plotly
import boto3


def serialize(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_json(double_precision=5)
    else:
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )


def invoke_lambda(lambda_function_name: str, lambda_payload: str) -> Dict:
    lambda_client = boto3.session.Session().client('lambda')

    lambda_response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=lambda_payload
    )

    return lambda_response



def list_chunks(lst: List[Any], n: int) -> Generator[List[Any], None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_relative_frame(time_series: pd.Series) -> pd.Series:
    """Input a pandas series of floats, and return ranks with 0 as rank = 0 and negative numbers as negative ranks."""
    all_ranks = time_series.where(time_series >= 0).rank(method='first') - 1  # start at 0 instead of 1
    negative_ranks = time_series.where(time_series < 0).rank(method='first', ascending=False)
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
    payload = json.dumps(payload, default=serialize)
    response = invoke_lambda(lambda_function_name="colab_notebook_backend", lambda_payload=payload)
    return plotly.io.from_json(json.loads(response["Payload"].read()))


def get_joint_plot(
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
    payload = json.dumps(payload, default=serialize)
    response = invoke_lambda(lambda_function_name="colab_notebook_backend", lambda_payload=payload)
    return plotly.io.from_json(json.loads(response["Payload"].read()))


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
