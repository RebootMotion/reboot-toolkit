import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots

plot_colors = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
    "rgb(148, 103, 189)",
]


def _get_vert_line(
    df: pd.DataFrame,
    time_col: str,
    time: float,
    angle_col: str,
    line_name: str = None,
) -> go.Scatter:
    """Input a pandas dataframe of joint angles, the column to use as time, the time, the angle column, and get a vertical dashed line at that time."""

    if line_name is None:
        show_legend = False
    else:
        show_legend = True

    idx = abs(df[time_col] - float(time)).idxmin()

    return go.Scatter(
        x=[df[time_col].loc[idx], df[time_col].loc[idx]],
        y=[df[angle_col].max(), df[angle_col].min()],
        showlegend=show_legend,
        fill=None,
        name=line_name,
        mode="lines",
        line=dict(color="black", width=1, dash="dash"),
    )


def get_population_joint_angles(
    pop_df: pd.DataFrame,
    time_col: str,
    angle_col: str,
    pop_color: str,
    prefix: str,
    visible=True,
):
    """Get upper and lower scatter plots, filled in between, showing the population range of a joint angle."""

    x_population = pop_df[time_col]

    y_pop_lower = pop_df[angle_col] - pop_df[f"{angle_col}_std"]
    y_pop_upper = pop_df[angle_col] + pop_df[f"{angle_col}_std"]

    y_lower_trace = go.Scatter(
        x=x_population,
        y=y_pop_lower,
        showlegend=False,
        fill=None,
        mode="lines",
        line=dict(color=pop_color, width=0.01),
        visible=visible,
    )

    y_upper_trace = go.Scatter(
        x=x_population,
        y=y_pop_upper,
        name=f"{prefix} {angle_col}",
        fill="tonexty",
        fillcolor=pop_color,
        mode="lines",
        line=dict(color=pop_color, width=0.01),
        visible=visible,
    )

    return y_lower_trace, y_upper_trace


def get_joint_angle_plots(
    joint_angles,
    rep_df: pd.DataFrame,
    player_df: pd.DataFrame,
    pop_df: pd.DataFrame,
    time_column: str,
    time_value: float = 0,
) -> go.Figure:
    """Get all the listed joint angle plots overlaid, including population data."""
    plot_colors_low_opac = [f"rgba{c[3:-1]}, 0.2)" for c in plot_colors]
    line_styles = ["dashdot", "dot", "dash"]
    line_width = 2

    fig_data = []
    for ai, angle in enumerate(joint_angles):
        if pop_df is not None:
            y_low_pop, y_up_pop = get_population_joint_angles(
                pop_df, time_column, angle, plot_colors_low_opac[ai], "pop"
            )
            fig_data.extend([y_low_pop, y_up_pop])

        if player_df is not None:
            y_low_ply, y_up_ply = get_population_joint_angles(
                player_df, time_column, angle, plot_colors_low_opac[ai], "player"
            )
            fig_data.extend([y_low_ply, y_up_ply])

        fig_data.append(
            go.Scatter(
                x=rep_df[time_column],
                y=rep_df[angle],
                name=angle,
                legendgroup=angle,
                mode="lines",
                line=dict(
                    color=plot_colors[ai],
                    width=line_width,
                    dash="solid",
                ),
            )
        )

    fig_data.append(
        _get_vert_line(
            rep_df,
            time_column,
            time_value,
            "torso_rot",
            line_name="foot contact",
        )
    )

    fig = go.Figure(data=fig_data)

    fig.update_layout(title="   |   ".join(joint_angles))

    fig.update_yaxes(patch={"title_text": "Joint Angle (deg)", "showticklabels": True})

    fig.update_xaxes(
        patch={
            "title": f"{time_column}",
            "type": "linear",
            "showticklabels": True,
        },
    )

    return fig


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


def get_joint_angle_animation(
    analysis_dicts,
    pop_df: pd.DataFrame,
    times,
    time_column: str,
    joint_angle: str,
    eye_hand_multiplier: int,
    plot_joint_angle_mean: bool = True,
) -> go.Figure:
    """Input a list of analysis dicts and a list of time points at which to analyze them, and get a paired skeleton animation and joint angle plot."""

    if len(analysis_dicts) > 1:
        colors = ["darkgray", "black"]

    else:
        colors = ["black"]

    animation_title = "<br> vs ".join(ad["play_guid"] for ad in analysis_dicts)

    if plot_joint_angle_mean:
        joint_angle_title = f"{joint_angle} mean"
    else:
        joint_angle_title = f"{joint_angle} for play"

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[animation_title, joint_angle_title],
        specs=[[{"type": "scene"}, {"type": "xy"}]],
    )
    y_low, y_up = get_population_joint_angles(pop_df, time_column, joint_angle)

    steps = []
    frames = []

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
        frame_data = []

        if i == 0:
            fig.add_trace(y_low, row=1, col=2)
            fig.add_trace(y_up, row=1, col=2)
        frame_data.extend([y_low, y_up])

        for j, analysis_dict in enumerate(reversed(analysis_dicts)):
            df_mean = (
                analysis_dict["df_mean"]
                if plot_joint_angle_mean
                else analysis_dict["df"]
            )

            y_o_i = go.Scatter(
                x=df_mean[time_column],
                y=df_mean[joint_angle],
                name=joint_angle,
                fill=None,
                mode="lines",
                line=dict(color=colors[j], width=3),
            )

            skel_3d_trace = _get_skeleton_3d(
                analysis_dict["df"],
                time_column,
                t,
                analysis_dict["player_id"],
                line_color=colors[j],
            )

            if i == 0:
                df_0_mean = df_mean
                fig.add_trace(y_o_i, row=1, col=2)
                fig.add_trace(skel_3d_trace, row=1, col=1)

            frame_data.extend([y_o_i, skel_3d_trace])

        vert_line = _get_vert_line(df_0_mean, time_column, t, joint_angle)

        if i == 0:
            fig.add_trace(vert_line, row=1, col=2)
        frame_data.append(vert_line)

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
            x=0.05,
            y=0.04,
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
            "pad": {"t": 30, "b": 0},
            "steps": steps,
        }
    ]

    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
    )

    fig.update_yaxes(
        patch={"title_text": "Joint Angle (deg)", "showticklabels": True},
        row=1,
        col=2,
    )

    fig.update_xaxes(
        patch={
            "title": time_column,
            "type": "linear",
            "showticklabels": True,
        },
        range=[-1.75, 0.75],
        row=1,
        col=2,
    )

    fig.update_layout(
        scene1=go.layout.Scene(
            camera=dict(
                center=dict(x=0, y=0, z=0),
                eye=dict(x=3 * eye_hand_multiplier, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            domain=dict(x=[0, 0.5], y=[0, 1.0]),
            aspectmode="data",
            xaxis=dict(showticklabels=False, title_text=""),
            yaxis=dict(showticklabels=False, title_text=""),
            zaxis=dict(showticklabels=False, title_text=""),
        )
    )

    return fig
