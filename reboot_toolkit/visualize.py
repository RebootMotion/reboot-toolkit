from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.signal import savgol_filter

COLORS_FOR_PLOTS = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
    "rgb(148, 103, 189)",
]


def get_vert_line(
    df: pd.DataFrame,
    x_col: str,
    x_value: float,
    y_cols: str,
    line_name: str = None,
    dash: str = "dash",
    color: str = "black",
) -> go.Scatter:
    """
    Create a plotly trace for a vertical line from a dataframe at a given x value for a list of y values.
    The line will go from the min of the y cols to the max of the y cols.

    :param df: the dataframe with data for creating the vertical line.
    :param x_col: the x_col for creating the x location
    :param x_value: the value of the x location
    :param y_cols: the y_cols across which we want to create a vertical line
    :param line_name: the name to assign the line
    :param dash: the dash style to use
    :param color: the color to use
    :return: the plotly trace for a vertical line
    """

    if line_name is None:
        show_legend = False
    else:
        show_legend = True

    x_loc = df[x_col].loc[(df[x_col] - float(x_value)).abs().idxmin()]

    return go.Scatter(
        x=[x_loc, x_loc],
        y=[df[y_cols].min().min(), df[y_cols].max().max()],
        showlegend=show_legend,
        fill=None,
        name=line_name,
        mode="lines",
        line=dict(color=color, width=1, dash=dash),
    )


def get_population_traces(
    pop_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    pop_color: str,
    prefix: str,
    opacity: float = 0.15,
    dash: str = "dash",
    visible=True,
) -> tuple[go.Scatter, go.Scatter, go.Scatter]:
    """
    Get upper and lower scatter plots, filled in between, showing the population range of a joint angle.

    :param pop_df: dataframe with population data
    :param x_col: the x column to use for the traces
    :param y_col: the y column to use for the traces
    :param pop_color: the color to use for the population
    :param prefix: the prefix to use for the legend label
    :param opacity: the opacity for the traces
    :param dash: the dash style for the traces
    :param visible: whether the trace is default visible
    :return: the lower, upper, and actual traces
    """

    pop_color_low_opac = f"rgba{pop_color[3:-1]}, 0.15)"

    x_population = pop_df[x_col]

    if f"{y_col}_25" in pop_df.columns:
        y_pop_lower = pop_df[f"{y_col}_25"]
        y_pop_upper = pop_df[f"{y_col}_75"]

    else:
        y_pop_lower = pop_df[y_col] - pop_df[f"{y_col}_std"]
        y_pop_upper = pop_df[y_col] + pop_df[f"{y_col}_std"]

    y_lower_trace = go.Scatter(
        x=x_population,
        y=y_pop_lower,
        mode="lines",
        line=dict(color=pop_color_low_opac, width=0.01),
        name=f"{prefix} {y_col}",
        visible=visible,
        showlegend=False,
        legendgroup=f"{prefix} {y_col}",
    )

    y_upper_trace = go.Scatter(
        x=x_population,
        y=y_pop_upper,
        mode="lines",
        line=dict(color=pop_color_low_opac, width=0.01),
        fill="tonexty",
        fillcolor=pop_color_low_opac,
        name=f"{prefix} {y_col}",
        visible=visible,
        showlegend=False,
        legendgroup=f"{prefix} {y_col}",
    )

    y_trace = go.Scatter(
        x=x_population,
        y=pop_df[y_col],
        mode="lines",
        line=dict(
            color=pop_color.replace(f", {round(opacity, 2)}", ""), width=1, dash=dash
        ),
        name=f"{prefix} {y_col}",
        visible=visible,
        showlegend=True,
        legendgroup=f"{prefix} {y_col}",
    )

    return y_lower_trace, y_upper_trace, y_trace


def get_joint_angle_plots(
    joint_angles,
    rep_df: pd.DataFrame | None,
    player_df: pd.DataFrame | None,
    pop_df: pd.DataFrame,
    time_column: str,
    time_value: float = 0,
    vert_line_name: str = None,
    filter_primary_trace: bool = True,
):
    """Get all the listed joint angle plots overlaid, including population data."""
    line_width = 2

    if rep_df is None:
        df_to_plot_label = "Player"
        df_to_plot = player_df

    else:
        df_to_plot_label = "Current"
        df_to_plot = rep_df

    trace_data = [
        get_vert_line(
            player_df if player_df is not None else rep_df,
            time_column,
            df_to_plot.loc[df_to_plot["dom_shoulder_rot"].idxmin()]["norm_time"],
            joint_angles,
            line_name="MER",
            dash="solid",
            color="gray",
        )
    ]
    for ai, angle in enumerate(joint_angles):
        if pop_df is not None:
            y_low, y_up, y = get_population_traces(
                pop_df, time_column, angle, COLORS_FOR_PLOTS[ai], "All", opacity=0.05
            )
            trace_data.extend([y_low, y_up, y])

        if (rep_df is not None and player_df is not None) or (pop_df is None):
            y_low, y_up, y = get_population_traces(
                player_df,
                time_column,
                angle,
                COLORS_FOR_PLOTS[ai],
                "Player",
                opacity=0.1,
                dash="dot",
            )
            trace_data.extend([y_low, y_up, y])

        trace_data.append(
            go.Scatter(
                x=df_to_plot[time_column],
                y=(
                    savgol_filter(df_to_plot[angle], window_length=21, polyorder=7)
                    if filter_primary_trace
                    else df_to_plot[angle]
                ),
                name=f"{df_to_plot_label} {angle}",
                legendgroup=angle,
                mode="lines",
                line=dict(
                    color=COLORS_FOR_PLOTS[ai],
                    width=line_width,
                    dash="solid",
                ),
            )
        )

    trace_data.append(
        get_vert_line(
            player_df if player_df is not None else rep_df,
            time_column,
            time_value,
            joint_angles,
            line_name=vert_line_name,
        )
    )

    return trace_data


def _get_skeleton_3d(
    skel_df: pd.DataFrame,
    time_col: str,
    time: float,
    legend_label: str,
    line_color: str = "black",
) -> go.Scatter3d:
    """Input a pandas dataframe with key points, the column to use as the time, the time, a legend, label, and line color, and get a 3D skeleton scatterplot."""

    idx = abs(skel_df[time_col] - float(time)).idxmin()

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

    bones = [
        (joints[1], joints[3]),
        (joints[3], joints[5]),
        (joints[2], joints[4]),
        (joints[4], joints[6]),
        (joints[7], joints[9]),
        (joints[9], joints[11]),
        (joints[8], joints[10]),
        (joints[10], joints[12]),
        (joints[7], joints[8]),
        (joints[1], joints[2]),
        (joints[1], joints[7]),
        (joints[2], joints[8]),
        (joints[5], joints[5]),
        (joints[6], joints[6]),
    ]

    x_bones = []
    y_bones = []
    z_bones = []

    for bone in bones:
        x_bones.extend(
            [
                skel_df[bone[0] + "_X"].loc[idx],
                skel_df[bone[1] + "_X"].loc[idx],
                None,
            ]
        )
        y_bones.extend(
            [
                skel_df[bone[0] + "_Y"].loc[idx],
                skel_df[bone[1] + "_Y"].loc[idx],
                None,
            ]
        )
        z_bones.extend(
            [
                skel_df[bone[0] + "_Z"].loc[idx],
                skel_df[bone[1] + "_Z"].loc[idx],
                None,
            ]
        )

    return go.Scatter3d(
        x=x_bones,
        y=y_bones,
        z=z_bones,
        name=legend_label,
        mode="lines",
        line=dict(color=line_color, width=8, dash="solid"),
    )


def generic_get_bounds(df: pd.DataFrame) -> dict:
    """
    derived from get_bounds in main_vertical_jumping

    Get min and max bounds for specific columns from a dataframe.

    :param df: the dataframe from which to get the bounds
    :return: dict of bounds
    """

    bounds = {}

    for coord in ("X", "Y", "Z"):
        coord_cols = [
            c for c in list(df) if c.endswith(f"_{coord}") and "Basketball" not in c
        ]
        bounds[f"{coord}min"] = df[coord_cols].min().min()
        bounds[f"{coord}max"] = df[coord_cols].max().max()

    return bounds


def get_joint_angle_animation(
    joint_angles,
    rep_df: pd.DataFrame,
    player_df: pd.DataFrame | None,
    pop_df: pd.DataFrame | None,
    time_column: str,
    y_axis_label: str,
    y_range: list | None = None,
    animation_title: str = "",
    frame_step: int = 2,
    plot_rep_time_series: bool = True,
) -> go.Figure:
    """Input a list of analysis dicts and a list of time points at which to analyze them, and get a paired skeleton animation and joint angle plot."""

    bounds = generic_get_bounds(rep_df)

    fig = make_subplots(
        rows=1,
        cols=2,
        # subplot_titles=[animation_title, joint_angle_title],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
    )

    steps = []
    frames = []

    times = (
        player_df[time_column].tolist()
        if pop_df is None
        else pop_df[time_column].tolist()
    )[::frame_step]

    for i, t in enumerate(times):
        step = {
            "args": [
                [i],
                {
                    "frame": {
                        "duration": 0.0,
                        "easing": "linear",
                        "redraw": True,
                    },
                    "transition": {"duration": 0, "easing": "linear"},
                },
            ],
            "label": round(t, 4),
            "method": "animate",
        }

        steps.append(step)

        frame_data = get_joint_angle_plots(
            joint_angles,
            rep_df if plot_rep_time_series else None,
            player_df,
            pop_df,
            time_column,
            t,
        )

        if i == 0:
            for trace in frame_data:
                fig.add_trace(trace, row=1, col=2)

        skel_3d_trace = _get_skeleton_3d(
            rep_df,
            time_column,
            t,
            "Skeleton",
            line_color="black",
        )
        if i == 0:
            fig.add_trace(skel_3d_trace, row=1, col=1)

        frame_data.append(skel_3d_trace)

        frame = dict(name=i, data=frame_data, traces=list(range(len(frame_data))))

        frames.append(frame)

    fig.update(frames=frames)

    updatemenus = [
        dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        [f"{i}" for i in range(len(times))],
                        dict(
                            frame=dict(duration=200, redraw=True),
                            transition=dict(duration=0),
                            easing="linear",
                            fromcurrent=True,
                        ),
                    ],
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        dict(
                            frame={"duration": 0, "redraw": False},
                            transition=dict(duration=0),
                            mode="immediate",
                        ),
                    ],
                ),
            ],
            direction="left",
            xanchor="left",
            showactive=True,
            x=0.2,
            y=-0.025,
        )
    ]

    sliders = [
        {
            "currentvalue": {
                "font": {"size": 12},
                "prefix": f"{time_column}: ",
                "visible": True,
            },
            "transition": {"duration": 100.0, "easing": "linear"},
            "pad": {"t": 25, "b": 0},
            "steps": steps,
        }
    ]

    fig.update_layout(
        title=animation_title,
        updatemenus=updatemenus,
        sliders=sliders,
        scene1=go.layout.Scene(
            camera=dict(
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2.5, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            domain=dict(x=[0, 0.45], y=[0, 1.0]),
            # aspectmode="data",
            # xaxis=dict(showticklabels=False, title_text=""),
            # yaxis=dict(showticklabels=False, title_text=""),
            # zaxis=dict(showticklabels=False, title_text=""),
            aspectmode="manual",
            aspectratio=dict(
                x=1,
                y=(bounds["Ymax"] - bounds["Ymin"]) / (bounds["Xmax"] - bounds["Xmin"]),
                z=(bounds["Zmax"] - bounds["Zmin"]) / (bounds["Xmax"] - bounds["Xmin"]),
            ),
            xaxis=dict(
                showticklabels=False,
                title_text="",
                range=[bounds["Xmin"], bounds["Xmax"]],
            ),
            yaxis=dict(
                showticklabels=False,
                title_text="",
                range=[bounds["Ymin"], bounds["Ymax"]],
            ),
            zaxis=dict(
                showticklabels=False,
                title_text="",
                range=[bounds["Zmin"], bounds["Zmax"]],
            ),
        ),
    )

    if not y_range:
        if plot_rep_time_series:
            y_range = [
                rep_df[joint_angles].min().min(),
                rep_df[joint_angles].max().max(),
            ]

        else:
            y_range = [
                player_df[joint_angles].min().min(),
                player_df[joint_angles].max().max(),
            ]

    fig.update_yaxes(
        patch={"title_text": y_axis_label, "range": y_range, "showticklabels": True},
        row=1,
        col=2,
    )

    fig.update_xaxes(
        patch={
            "title": time_column,
            "type": "linear",
            "showticklabels": True,
        },
        row=1,
        col=2,
    )

    return fig
