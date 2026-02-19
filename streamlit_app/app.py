"""Streamlit interactive demo for the APF Navigation Simulator.

Launch with:
    streamlit run streamlit_app/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

from src.config import Config
from src.environment import Environment
from src.evaluation import Evaluator
from streamlit_app.canvas_editor import (
    get_custom_obstacles,
    get_custom_start,
    get_custom_waypoints,
    render_custom_canvas,
    render_preset_canvas,
)
from streamlit_app.config_builder import (
    SLIDER_DEFS,
    build_config,
    get_scenario_names,
    load_default_config_dict,
)
from streamlit_app.simulation_runner import run_scenario_headless

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="APF Navigation Simulator",
    page_icon="\U0001f916",  # noqa: RUF001
    layout="wide",
)

# ---------------------------------------------------------------------------
# Button contrast CSS for dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Secondary / default buttons — light-blue border matching theme */
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="baseButton-secondary"],
    .stButton > button {
        border: 1px solid #00c8ff !important;
        color: #00c8ff !important;
        background-color: rgba(0, 200, 255, 0.08) !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    .stButton > button:hover {
        border: 1px solid #00c8ff !important;
        color: #00c8ff !important;
        background-color: rgba(0, 200, 255, 0.18) !important;
    }
    /* Keep primary button distinct */
    button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        border: 1px solid #00c8ff !important;
        background-color: #00c8ff !important;
        color: #000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("APF Navigation Simulator")
st.caption("Interactive demo \u2014 Artificial Potential Fields with vortex circulation, "
           "directional attenuation, and PID heading control.")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "mode": "Preset",
        "scenario_name": "corridor",
        "obstacles": [],
        "waypoints": [],
        "start_pos": None,
        "placement_mode": "Obstacles",
        "obstacle_radius": 25,
        "video_bytes": None,
        "run_data": None,
        "metrics": None,
        "canvas_obj_count": 0,
        "canvas_reset_counter": 0,
        "view_mode": "editor",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _on_mode_change() -> None:
    """Reset results state when switching between Preset and Custom modes."""
    st.session_state.view_mode = "editor"
    st.session_state.video_bytes = None
    st.session_state.run_data = None
    st.session_state.metrics = None


_init_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    # Mode selection
    mode = st.radio("Mode", ["Preset", "Custom"], horizontal=True,
                    key="mode", on_change=_on_mode_change)

    if mode == "Preset":
        scenario_names = get_scenario_names()
        scenario_name = st.selectbox(
            "Scenario", scenario_names,
            index=scenario_names.index(st.session_state.scenario_name)
            if st.session_state.scenario_name in scenario_names else 0,
            key="scenario_name",
        )

    if mode == "Custom":
        st.divider()
        st.subheader("Placement")
        st.slider("Obstacle radius", min_value=10, max_value=80,
                   value=25, step=5, key="obstacle_radius")

    st.divider()
    st.subheader("APF Parameters")

    # Collect slider overrides
    slider_overrides: dict = {}

    for label, path, default, lo, hi, step in SLIDER_DEFS:
        val = st.slider(
            label, min_value=float(lo), max_value=float(hi),
            value=float(default), step=float(step),
        )
        slider_overrides[path] = val

    # Advanced settings
    with st.expander("Advanced"):
        duration_override = st.slider(
            "Max time (s)", min_value=15, max_value=60,
            value=60, step=5,
        )

    st.divider()
    if st.button("Reset to Defaults"):
        for key in ["obstacles", "waypoints", "start_pos", "video_bytes",
                     "run_data", "metrics", "canvas_obj_count", "view_mode"]:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.canvas_reset_counter = st.session_state.get(
            "canvas_reset_counter", 0) + 1
        _init_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
in_results = (st.session_state.view_mode == "results"
              and st.session_state.video_bytes is not None)

col_main, col_right = st.columns([3, 1])

with col_main:
    if in_results:
        # --- Results view (replaces canvas) ---
        st.subheader("Simulation Playback")
        st.video(st.session_state.video_bytes, format="video/mp4")
        if st.button("Back to Editor"):
            # Full reset for a clean canvas
            for key in ["obstacles", "waypoints", "start_pos", "video_bytes",
                         "run_data", "metrics", "canvas_obj_count"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.canvas_reset_counter = st.session_state.get(
                "canvas_reset_counter", 0) + 1
            st.session_state.view_mode = "editor"
            _init_state()
            st.rerun()
    else:
        # --- Editor view (canvas) ---
        if mode == "Preset":
            config_dict = load_default_config_dict()
            config = Config(config_dict)
            scenario_name = st.session_state.scenario_name

            env = Environment.from_scenario(config, scenario_name)
            scenarios = config.environment.scenarios  # type: ignore
            scenario_data = getattr(scenarios, scenario_name)
            preset_waypoints = [
                (float(wp[0]), float(wp[1]))
                for wp in scenario_data.waypoints  # type: ignore
            ]
            preset_start = (
                float(scenario_data.start[0]),  # type: ignore
                float(scenario_data.start[1]),  # type: ignore
            )

            st.subheader(f"Scenario: {scenario_name}")
            render_preset_canvas(
                env.get_obstacles(), preset_waypoints, preset_start,
            )

        else:
            st.subheader("Custom Scenario")

            # Placement mode selector
            placement_mode = st.radio(
                "Placement mode",
                ["Obstacles", "Waypoints", "Start"],
                horizontal=True,
                key="placement_mode",
            )

            render_custom_canvas(
                placement_mode,
                obstacle_radius=st.session_state.get("obstacle_radius", 25),
            )

            # Info bar
            n_obs = len(st.session_state.get("obstacles", []))
            n_wp = len(st.session_state.get("waypoints", []))
            has_start = st.session_state.get("start_pos") is not None
            st.caption(
                f"Obstacles: {n_obs} | Waypoints: {n_wp} | "
                f"Start: {'set' if has_start else 'not set'}"
            )

            col_clear, col_undo, _ = st.columns([1, 1, 2])
            with col_clear:
                if st.button("Clear All"):
                    st.session_state.obstacles = []
                    st.session_state.waypoints = []
                    st.session_state.start_pos = None
                    st.session_state.canvas_obj_count = 0
                    st.session_state.canvas_reset_counter = st.session_state.get(
                        "canvas_reset_counter", 0) + 1
                    st.session_state.video_bytes = None
                    st.session_state.run_data = None
                    st.session_state.metrics = None
                    st.session_state.view_mode = "editor"
                    st.rerun()
            with col_undo:
                if st.button("Undo Last"):
                    if placement_mode == "Obstacles" and st.session_state.obstacles:
                        st.session_state.obstacles.pop()
                    elif placement_mode == "Waypoints" and st.session_state.waypoints:
                        st.session_state.waypoints.pop()
                    elif placement_mode == "Start":
                        st.session_state.start_pos = None
                    st.session_state.canvas_obj_count = 0
                    st.session_state.canvas_reset_counter = st.session_state.get(
                        "canvas_reset_counter", 0) + 1
                    st.rerun()

with col_right:
    if in_results:
        # --- Metrics panel (results mode) ---
        st.subheader("Metrics")
        m = st.session_state.metrics
        if m is not None:
            success_icon = "pass" if m.success else "fail"
            st.metric("Result", success_icon)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Collisions", m.collisions)
                st.metric("Path Efficiency", f"{m.path_efficiency:.1%}")
                st.metric("Avg Speed", f"{m.avg_speed:.1f} u/s")
            with col_m2:
                st.metric("Waypoints", f"{m.waypoints_reached}/{m.total_waypoints}")
                st.metric("Min Safety", f"{m.min_safety_margin:.1f}")
                st.metric("Time", f"{m.completion_time:.1f}s")

            with st.expander("Full metrics"):
                st.json({
                    "scenario": m.scenario,
                    "success": m.success,
                    "collisions": m.collisions,
                    "waypoints_reached": m.waypoints_reached,
                    "total_waypoints": m.total_waypoints,
                    "path_length": round(m.path_length, 1),
                    "optimal_length": round(m.optimal_length, 1),
                    "path_efficiency": round(m.path_efficiency, 4),
                    "smoothness": round(m.smoothness, 4),
                    "min_safety_margin": round(m.min_safety_margin, 1),
                    "avg_safety_margin": round(m.avg_safety_margin, 1),
                    "avg_speed": round(m.avg_speed, 1),
                    "completion_time": round(m.completion_time, 2),
                    "time_in_avoid": round(m.time_in_avoid, 4),
                    "time_in_navigate": round(m.time_in_navigate, 4),
                })
    else:
        # --- Controls panel (editor mode) ---
        st.subheader("Controls")

        speed_multiplier = st.select_slider(
            "Playback speed",
            options=[1.0, 1.5, 2.0, 3.0],
            value=1.0,
        )

        # Run button
        can_run = True
        if mode == "Custom":
            waypoints = get_custom_waypoints()
            if not waypoints:
                st.warning("Place at least one waypoint.")
                can_run = False

        if st.button("Run Simulation", type="primary", disabled=not can_run):
            config = build_config(slider_overrides)

            progress_bar = st.progress(0, text="Simulating...")

            def on_progress(frac: float) -> None:
                progress_bar.progress(frac, text=f"Simulating... {frac * 100:.0f}%")

            try:
                if mode == "Preset":
                    video_bytes, run_data = run_scenario_headless(
                        config,
                        scenario_name=st.session_state.scenario_name,
                        duration=duration_override,
                        frame_skip=2,
                        speed_multiplier=speed_multiplier,
                        progress_callback=on_progress,
                    )
                else:
                    custom_obstacles = get_custom_obstacles()
                    custom_waypoints = get_custom_waypoints()
                    custom_start = get_custom_start()

                    video_bytes, run_data = run_scenario_headless(
                        config,
                        obstacles=custom_obstacles,
                        waypoints_xy=custom_waypoints,
                        start_pos=custom_start,
                        duration=duration_override,
                        frame_skip=2,
                        speed_multiplier=speed_multiplier,
                        progress_callback=on_progress,
                    )

                st.session_state.video_bytes = video_bytes
                st.session_state.run_data = run_data
                st.session_state.metrics = Evaluator.evaluate(run_data)
                st.session_state.view_mode = "results"
                progress_bar.empty()
                st.rerun()

            except Exception as e:
                progress_bar.empty()
                st.error(f"Simulation failed: {e}")
