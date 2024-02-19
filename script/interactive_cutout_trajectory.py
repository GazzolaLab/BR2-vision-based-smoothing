import os, sys

import re
import numpy as np
import pandas as pd

import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback, State

import br2_vision
from br2_vision.utility.logging import config_logging, get_script_logger
from br2_vision.cv2_custom.extract_info import get_video_frame_count
from br2_vision.data import MarkerPositions, TrackingData, FlowQueue

import click

from br2_vision.interactive.app import app  # , cache, long_callback_manager
from br2_vision.interactive.sidebar import get_sidebar, CONTENT_STYLE, BANNER_STYLE

# ---------------------------- Config ----------------------------

config = br2_vision.load_config()
config_logging(False)
logger = get_script_logger(os.path.basename(__file__))
ctx = {}

# ---------------------------- App ----------------------------

sidebar = get_sidebar()

tabs = []
tabs_name_camera = lambda i: f"tab-{i+1}-camera-data"
tabs_name_regex = r"tab-(\d+)-camera-data"
get_cid = lambda tab: int(re.match(tabs_name_regex, tab).group(1))
last_clicked_data = None

contents = html.Div(
    [
        html.H5("Optical Flow Interactive Editor", style={"text-align": "center"}),
        html.Hr(),
        dcc.Tabs(id="tabs-camera-data", value="tab-1-camera-data", children=tabs),
        html.Div(
            id="tabs-content-camera-data",
            children=[
                dcc.Graph(
                    id=f"graph-x",
                ),
                dcc.Graph(
                    id=f"graph-y",
                ),
                # trim-button
                html.Button("Trim", id="trim-button", n_clicks=0),
                # selection textbox
                dcc.Textarea(
                    id=f"textarea-x",
                    value="Selected data",
                    style={"width": "100%", "height": 150},
                ),
                dcc.Textarea(
                    id=f"textarea-y",
                    value="Selected data",
                    style={"width": "100%", "height": 150},
                ),
            ],
        ),
        # dash_table.DataTable(impedances.to_dict('records'), id='selected-impedances-tbl'),
    ],
    style=CONTENT_STYLE,
)

banner = html.Div(
    id="banner",
    className="banner",
    children=[
        html.Img(
            src="https://github.com/skim0119/BR2-simulator/blob/main/docs/_static/assets/logo_v1.png",
            style=BANNER_STYLE,
        ),
    ],
    style={"background-color": "#6262FF"},
)

app.layout = html.Div(
    [
        banner,
        sidebar,
        contents,
        dcc.Interval(
            id="interval-component", interval=5 * 1000, n_intervals=0  # in milliseconds
        ),
    ],
    id="app-container",
)

# ---------------------------- Callback ----------------------------


@callback(
    [Output("graph-x", "figure"), Output("graph-y", "figure")],
    [Input("tabs-camera-data", "value"), Input("trim-button", "n_clicks")],
)
def render_content(tab, click):
    global last_clicked_data
    cid = get_cid(tab)

    if click is not None and last_clicked_data is not None:
        _cid, click_data = last_clicked_data
        trim_data(click_data, _cid)

    # Reset clicked data
    last_clicked_data = None

    fig_x, fig_y = render_camera_data(cid)
    return fig_x, fig_y


def render_camera_data(cid):
    df = get_camera_data(cid)  # shape: (num_queues, num_frames, 2)

    # plot using px
    fig_x = px.line(
        data_frame=df,
        x="frame",
        y="x",
        color="label",
        title=f"{cid} x position",
        labels={"frame": "Frame", "x": "X location (pixel)"},
    )
    fig_x.update_traces(line=dict(width=1))
    fig_x.update_xaxes(minor=dict(ticks="inside", showgrid=True))
    fig_x.update_yaxes(minor=dict(ticks="inside", showgrid=True))

    fig_y = px.line(
        data_frame=df,
        x="frame",
        y="y",
        color="label",
        title=f"{cid} y position",
        labels={"frame": "Frame", "y": "Y location (pixel)"},
    )
    fig_y.update_traces(line=dict(width=1))
    fig_y.update_xaxes(minor=dict(ticks="inside", showgrid=True))
    fig_y.update_yaxes(minor=dict(ticks="inside", showgrid=True))

    return fig_x, fig_y


# Callback: click plot, display label, frame, x, and y.
# TODO: See if it can be combined
@callback(
    Output("textarea-x", "value"),
    Input("graph-x", "clickData"),
    State("tabs-camera-data", "value"),
)
def cb_trim_data_x(click_data, tab):
    global last_clicked_data
    cid = get_cid(tab)
    last_clicked_data = (cid, click_data)
    return f"Selected data - x (camera {cid}): \n {click_data=}"


@callback(
    Output("textarea-y", "value"),
    Input("graph-y", "clickData"),
    State("tabs-camera-data", "value"),
)
def cb_trim_data_y(click_data, tab):
    global last_clicked_data
    cid = get_cid(tab)
    last_clicked_data = (cid, click_data)
    return f"Selected data - y (camera {cid}): \n {click_data=}"


def trim_data(click_data, cid):
    global ctx
    tag = ctx["tag"]
    run_id = ctx["run_id"]

    if click_data is None:
        return

    qid = click_data["points"][0]["curveNumber"]
    frame = click_data["points"][0]["x"]
    y = click_data["points"][0]["y"]

    datapath = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    with TrackingData.load(path=datapath) as dataset:
        queue = dataset.get_flow_queues(camera=cid, force_run_all=True)[qid]
        dataset.trim_trajectory(queue.get_tag(), frame)


# ---------------------------- Methods ----------------------------


def get_camera_data(cid) -> pd.DataFrame:
    global ctx
    tag = ctx["tag"]
    run_id = ctx["run_id"]

    headers = ["label", "x", "y", "frame"]
    df = pd.DataFrame(columns=headers)

    # TODO: maybe move this to a function in TrackingData
    datapath = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    with TrackingData.load(path=datapath) as dataset:
        queues = dataset.get_flow_queues(camera=cid, force_run_all=True)
        num_queues = len(queues)
        num_frames = get_video_frame_count(
            config["PATHS"]["footage_video_path"].format(tag, cid, run_id)
        )

        for qid, q in enumerate(queues):
            data = dataset.load_pixel_flow_trajectory(q, full_trajectory=True)
            _df = pd.DataFrame(data, columns=["x", "y"])
            _df["label"] = q.get_tag()
            _df["frame"] = np.arange(num_frames)
            df = pd.concat([df, _df], ignore_index=True)

    # set negative x and y to be nan
    df.loc[df["x"] < 0, "x"] = np.nan
    df.loc[df["y"] < 0, "y"] = np.nan

    # Other operation
    # df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2)

    return df


def set_tabs(n_cameras):
    global tabs

    # remove all items in tabs
    tabs.clear()

    # add three tabs
    for i in range(n_cameras):
        tab = dcc.Tab(label=f"Camera {i+1}", value=f"tab-{i+1}-camera-data")
        tabs.append(tab)


@click.command()
@click.option(
    "-t",
    "--tag",
    type=str,
    required=True,
    help="Experiment tag.",
)
@click.option(
    "-r",
    "--run-id",
    type=int,
    required=True,
    help="Specify run index..",
)
@click.option("-v", "--verbose", is_flag=True, help="Verbose mode.")
@click.option("-d", "--dry", is_flag=True, help="Dry run.")
def launch(tag, run_id, verbose, dry):
    global ctx
    ctx.update(locals())
    # Run optical flow for each run-id
    datapath = config["PATHS"]["tracing_data_path"].format(tag, run_id)
    with TrackingData.load(path=datapath) as dataset:
        num_cameras = len(dataset.iter_cameras())
        num_cameras = 3
        set_tabs(num_cameras)

    if not dry:
        app.run_server("0.0.0.0", 8000, debug=True)


if __name__ == "__main__":
    app.run_server("0.0.0.0", 8000, debug=True)
