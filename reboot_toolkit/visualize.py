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
    y_cols: list[str],
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


def get_player_population_traces(
    cols_to_plot: list[str],
    rep_df: pd.DataFrame | None,
    player_df: pd.DataFrame | None,
    pop_df: pd.DataFrame,
    x_column: str,
    x_value: float,
    vert_line_name: str = None,
    filter_primary_trace: bool = True,
    line_width: int = 2,
) -> list:
    """
    Get a list of plotly traces that depict the player and population plots for a list of columns.

    :param cols_to_plot: the columns to plot
    :param rep_df: Optional representative dataframe to plot a single movement
    :param player_df: Optional player dataframe to plot the traces for aggregated movements for a player
    :param pop_df: Population dataframe to plot the traces for aggregated movements for a player
    :param x_column: the name of the x column to use for the traces
    :param x_value: the x value to use for a vertical line for the traces
    :param vert_line_name: optional name for the vertical line
    :param filter_primary_trace: whether to apply filtering to the primary trace
    :param line_width: the width for the trace lines
    :return: list of plotly traces
    """
    if rep_df is None:
        df_to_plot_label = "Player"
        df_to_plot = player_df

    else:
        df_to_plot_label = "Current"
        df_to_plot = rep_df

    trace_data = [
        get_vert_line(
            player_df if player_df is not None else rep_df,
            x_column,
            df_to_plot.loc[df_to_plot["dom_shoulder_rot"].idxmin()]["norm_time"],
            cols_to_plot,
            line_name="MER",
            dash="solid",
            color="gray",
        )
    ]
    for ai, angle in enumerate(cols_to_plot):
        if pop_df is not None:
            y_low, y_up, y = get_population_traces(
                pop_df, x_column, angle, COLORS_FOR_PLOTS[ai], "All", opacity=0.05
            )
            trace_data.extend([y_low, y_up, y])

        if (rep_df is not None and player_df is not None) or (pop_df is None):
            y_low, y_up, y = get_population_traces(
                player_df,
                x_column,
                angle,
                COLORS_FOR_PLOTS[ai],
                "Player",
                opacity=0.1,
                dash="dot",
            )
            trace_data.extend([y_low, y_up, y])

        trace_data.append(
            go.Scatter(
                x=df_to_plot[x_column],
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
            x_column,
            x_value,
            cols_to_plot,
            line_name=vert_line_name,
        )
    )

    return trace_data


def get_skeleton_3d(
    skel_df: pd.DataFrame,
    x_col: str,
    x_value: float,
    legend_label: str,
    line_color: str = "black",
) -> go.Scatter3d:
    """
    Input a dataframe with key points, the x column to use, the x value at which to plot,
    a legend, label, and line color, and get a 3D skeleton scatterplot.

    :param skel_df: the dataframe with key points
    :param x_col: the x column to use
    :param x_value: the x value to use for a vertical line for the traces
    :param legend_label: the legend label for identifying this trace
    :param line_color: the line color for the trace
    :return: the 3d scatter plot for the skeleton
    """

    idx = (skel_df[x_col] - float(x_value)).abs().idxmin()

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


def generic_get_bounds(df: pd.DataFrame, coords: tuple[str] = ("X", "Y", "Z")) -> dict:
    """
    Get min and max bounds for specific columns from a dataframe.

    :param df: the dataframe from which to get the bounds
    :param coords: the coords for which to calculate the bounds
    :return: dict of bounds for each of the input coords
    """

    bounds = {}

    for coord in coords:
        coord_cols = [
            c for c in list(df) if c.endswith(f"_{coord}") and "ball" not in c.lower()
        ]
        bounds[f"{coord}min"] = df[coord_cols].min().min()
        bounds[f"{coord}max"] = df[coord_cols].max().max()

    return bounds


def get_player_animation(
    cols_to_plot: list[str],
    rep_df: pd.DataFrame,
    player_df: pd.DataFrame | None,
    pop_df: pd.DataFrame | None,
    x_column: str,
    y_axis_label: str,
    y_range: list | None = None,
    animation_title: str = "",
    frame_step: int = 2,
    plot_rep_time_series: bool = True,
    subplot_titles: list[str] | None = None,
) -> go.Figure:
    """
    Create a plotly figure that shows a player animation paired with time series scatter traces.

    :param cols_to_plot: the cols to plot for the scatter traces
    :param rep_df: Optional dataframe for the representative movement for the scatter traces
    :param player_df: optional dataframe of all player movements for the scatter traces
    :param pop_df: dataframe to use as the population for all movements
    :param x_column: the x column to use for the scatter traces
    :param y_axis_label: the label for the y axis
    :param y_range: Optional specified range for the y axis
    :param animation_title: the title for the animation
    :param frame_step: how many frames to traverse in one step for the animation
    :param plot_rep_time_series: whether to plot the representative time series or not
    :param subplot_titles: optional list of subplot titles
    :return: the plotly figure with the skeleton animation paired with scatter traces
    """

    bounds = generic_get_bounds(rep_df)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=subplot_titles,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
    )

    steps = []
    frames = []

    times = (
        player_df[x_column].tolist() if pop_df is None else pop_df[x_column].tolist()
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

        frame_data = get_player_population_traces(
            cols_to_plot=cols_to_plot,
            rep_df=rep_df if plot_rep_time_series else None,
            player_df=player_df,
            pop_df=pop_df,
            x_column=x_column,
            x_value=t,
        )

        if i == 0:
            for trace in frame_data:
                fig.add_trace(trace, row=1, col=2)

        skel_3d_trace = get_skeleton_3d(
            rep_df, x_column, t, "Skeleton", line_color="black"
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
                "prefix": f"{x_column}: ",
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
                rep_df[cols_to_plot].min().min(),
                rep_df[cols_to_plot].max().max(),
            ]

        else:
            y_range = [
                player_df[cols_to_plot].min().min(),
                player_df[cols_to_plot].max().max(),
            ]

    fig.update_yaxes(
        patch={"title_text": y_axis_label, "range": y_range, "showticklabels": True},
        row=1,
        col=2,
    )

    fig.update_xaxes(
        patch={
            "title": x_column,
            "type": "linear",
            "showticklabels": True,
        },
        row=1,
        col=2,
    )

    return fig
