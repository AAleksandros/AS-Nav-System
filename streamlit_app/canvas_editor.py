"""Canvas editor for interactive obstacle/waypoint placement.

Uses streamlit-drawable-canvas for drawing circles (obstacles) and
points (waypoints/start), with coordinate conversion between canvas
(y-down) and world (y-up) systems.

State management:
- A solid background_color is used for the canvas (no background_image),
  avoiding Streamlit media-file-eviction errors entirely.
- Session state is the source of truth for placed objects.
- Object detection uses count-based comparison: since the canvas key is
  stable and the background is constant, the canvas preserves its internal
  drawn-objects across reruns, making count tracking reliable.
"""

from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import streamlit as st
from PIL import Image  # type: ignore
from streamlit_drawable_canvas import st_canvas  # type: ignore

from src.models import Obstacle

# Canvas dimensions — scaled down from world size (800x600) to fit
# Streamlit Cloud column layouts without clipping.
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480

# World dimensions (used for coordinate conversion)
WORLD_WIDTH = 800
WORLD_HEIGHT = 600


def canvas_to_world(cx: float, cy: float) -> Tuple[float, float]:
    """Convert canvas (y-down, scaled) to world (y-up, 800x600) coordinates."""
    wx = cx * WORLD_WIDTH / CANVAS_WIDTH
    wy = WORLD_HEIGHT - (cy * WORLD_HEIGHT / CANVAS_HEIGHT)
    return (wx, wy)


def world_to_canvas(wx: float, wy: float) -> Tuple[float, float]:
    """Convert world (y-up, 800x600) to canvas (y-down, scaled) coordinates."""
    cx = wx * CANVAS_WIDTH / WORLD_WIDTH
    cy = (WORLD_HEIGHT - wy) * CANVAS_HEIGHT / WORLD_HEIGHT
    return (cx, cy)



def _draw_scenario_preview(
    obstacles: List[Dict],
    waypoints: List[Tuple[float, float]],
    start_pos: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Draw a complete scenario preview showing all placed objects.

    Returns BGR image (for st.image via cv2 → RGB conversion).
    """
    bg = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), [40, 40, 40], dtype=np.uint8)

    # Grid
    for gx in range(0, CANVAS_WIDTH, 50):
        cv2.line(bg, (gx, 0), (gx, CANVAS_HEIGHT), (60, 60, 60), 1)
    for gy in range(0, CANVAS_HEIGHT, 50):
        cv2.line(bg, (0, gy), (CANVAS_WIDTH, gy), (60, 60, 60), 1)

    # Boundary
    cv2.rectangle(bg, (0, 0), (CANVAS_WIDTH - 1, CANVAS_HEIGHT - 1),
                  (200, 200, 200), 2)

    # Obstacles
    for obs in obstacles:
        cx, cy = world_to_canvas(obs["x"], obs["y"])
        r = int(obs["radius"] * CANVAS_WIDTH / WORLD_WIDTH)
        cv2.circle(bg, (int(cx), int(cy)), r, (80, 80, 80), -1)
        cv2.circle(bg, (int(cx), int(cy)), r, (200, 200, 200), 2)

    # Waypoint path lines
    if start_pos and waypoints:
        sp = world_to_canvas(*start_pos)
        wp0 = world_to_canvas(*waypoints[0])
        cv2.line(bg, (int(sp[0]), int(sp[1])), (int(wp0[0]), int(wp0[1])),
                 (180, 180, 180), 2)
    if len(waypoints) >= 2:
        for i in range(len(waypoints) - 1):
            p1 = world_to_canvas(*waypoints[i])
            p2 = world_to_canvas(*waypoints[i + 1])
            cv2.line(bg, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                     (180, 180, 180), 2)

    # Waypoint markers
    for i, wp in enumerate(waypoints):
        cx, cy = world_to_canvas(*wp)
        cv2.circle(bg, (int(cx), int(cy)), 8, (0, 255, 255), -1)
        cv2.circle(bg, (int(cx), int(cy)), 10, (255, 255, 255), 2)
        cv2.putText(bg, str(i + 1), (int(cx) - 4, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Start position
    if start_pos:
        cx, cy = world_to_canvas(*start_pos)
        cv2.circle(bg, (int(cx), int(cy)), 10, (0, 200, 255), -1)
        cv2.circle(bg, (int(cx), int(cy)), 12, (255, 255, 255), 2)
        cv2.putText(bg, "S", (int(cx) - 5, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return bg


@st.cache_data
def _render_preset_image(
    obs_tuples: Tuple[Tuple[float, float, float], ...],
    waypoints: Tuple[Tuple[float, float], ...],
    start_pos: Tuple[float, float],
) -> Image.Image:
    """Render and cache a preset scenario preview as a PIL Image.

    Using hashable tuple args so st.cache_data works. Caching prevents
    Streamlit's media file storage from evicting the image between reruns.
    """
    obs_dicts = [{"x": o[0], "y": o[1], "radius": o[2]} for o in obs_tuples]
    preview = _draw_scenario_preview(obs_dicts, list(waypoints), start_pos)
    return Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))


def render_preset_canvas(
    obstacles: List[Obstacle],
    waypoints: List[Tuple[float, float]],
    start_pos: Tuple[float, float],
) -> None:
    """Display a read-only preview of a preset scenario using st.image."""
    obs_tuples = tuple((o.x, o.y, o.radius) for o in obstacles)
    wp_tuples = tuple(waypoints)
    img = _render_preset_image(obs_tuples, wp_tuples, start_pos)
    st.image(img)


def render_custom_canvas(placement_mode: str, obstacle_radius: int = 25) -> None:
    """Display an interactive canvas and process new drawings.

    Uses a static cached background so the canvas never resets its
    internal object state. Object detection is count-based.

    Parameters
    ----------
    placement_mode : str
        One of "Obstacles", "Waypoints", "Start".
    obstacle_radius : int
        Default radius for obstacles when drawn as points.
    """
    # Ensure session state
    if "obstacles" not in st.session_state:
        st.session_state.obstacles = []
    if "waypoints" not in st.session_state:
        st.session_state.waypoints = []
    if "canvas_obj_count" not in st.session_state:
        st.session_state.canvas_obj_count = 0
    if "canvas_reset_counter" not in st.session_state:
        st.session_state.canvas_reset_counter = 0

    # Canvas for drawing new objects (static background)
    st.caption("Click/draw below to place objects:")

    if placement_mode == "Obstacles":
        drawing_mode = "circle"
        stroke_color = "rgba(200, 200, 200, 1.0)"
        fill_color = "rgba(80, 80, 80, 0.8)"
    else:
        drawing_mode = "point"
        stroke_color = "rgba(0, 255, 255, 1.0)"
        fill_color = "rgba(0, 255, 255, 0.8)"

    canvas_result = st_canvas(
        background_color="#282828",
        drawing_mode=drawing_mode,
        stroke_width=2,
        stroke_color=stroke_color,
        fill_color=fill_color,
        width=CANVAS_WIDTH,
        height=CANVAS_HEIGHT,
        point_display_radius=int(obstacle_radius * CANVAS_WIDTH / WORLD_WIDTH) if placement_mode == "Obstacles" else 8,
        display_toolbar=False,
        key=f"custom_canvas_{st.session_state.canvas_reset_counter}",
    )

    # --- Detect and process new objects ---
    if canvas_result is None or canvas_result.json_data is None:
        return

    objects = canvas_result.json_data.get("objects", [])
    current_count = len(objects)
    prev_count = st.session_state.canvas_obj_count

    if current_count <= prev_count:
        return

    # Process only new objects
    new_objects = objects[prev_count:]
    st.session_state.canvas_obj_count = current_count

    for obj in new_objects:
        obj_type = obj.get("type", "")

        if placement_mode == "Obstacles":
            scale = WORLD_WIDTH / CANVAS_WIDTH
            if obj_type == "circle":
                # Drag-drawn circle: center at (left + radius, top + radius)
                cx = obj.get("left", 0) + obj.get("radius", obstacle_radius)
                cy = obj.get("top", 0) + obj.get("radius", obstacle_radius)
                radius_canvas = max(obj.get("radius", obstacle_radius), 10)
            else:
                # Fallback: treat as point click, use default radius
                cx = obj.get("left", 0)
                cy = obj.get("top", 0)
                radius_canvas = int(obstacle_radius * CANVAS_WIDTH / WORLD_WIDTH)
            wx, wy = canvas_to_world(cx, cy)
            world_radius = radius_canvas * scale
            st.session_state.obstacles.append({"x": wx, "y": wy, "radius": world_radius})

        elif placement_mode == "Waypoints":
            cx = obj.get("left", 0)
            cy = obj.get("top", 0)
            wx, wy = canvas_to_world(cx, cy)
            st.session_state.waypoints.append((wx, wy))

        elif placement_mode == "Start":
            cx = obj.get("left", 0)
            cy = obj.get("top", 0)
            wx, wy = canvas_to_world(cx, cy)
            st.session_state.start_pos = (wx, wy)

    # No st.rerun() needed — session state is already updated and the
    # info bar (rendered after this function returns) will reflect the change.


def get_custom_obstacles() -> List[Obstacle]:
    """Convert session state obstacles to Obstacle dataclass instances."""
    return [
        Obstacle(x=o["x"], y=o["y"], radius=o["radius"])
        for o in st.session_state.get("obstacles", [])
    ]


def get_custom_waypoints() -> List[Tuple[float, float]]:
    """Get waypoints from session state."""
    return list(st.session_state.get("waypoints", []))


def get_custom_start() -> Optional[Tuple[float, float]]:
    """Get start position from session state."""
    return st.session_state.get("start_pos", None)
