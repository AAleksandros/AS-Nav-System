"""Canvas editor for interactive obstacle/waypoint placement.

Uses streamlit-drawable-canvas for drawing circles (obstacles) and
points (waypoints/start), with coordinate conversion between canvas
(y-down) and world (y-up) systems.

The canvas dimensions match the world dimensions (800x600) exactly,
so the only conversion needed is a y-axis flip — no scaling.

Fabric.js object model (both circle and point drawing modes):
    originX = "left"  → ``left`` is the LEFT edge of the bounding box
    originY = "center" → ``top`` IS the vertical center
    Circle centers are therefore at:
        cx = left + radius * cos(angle)
        cy = top  + radius * sin(angle)
    Point tool pre-adjusts ``left`` so the visual dot lands on the click,
    but the center formula above still applies.

State management:
- A solid background_color is used for the canvas (no background_image),
  avoiding Streamlit media-file-eviction errors entirely.
- Session state is the source of truth for placed objects.
- Object detection uses count-based comparison: since the canvas key is
  stable and the background is constant, the canvas preserves its internal
  drawn-objects across reruns, making count tracking reliable.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import streamlit as st
from PIL import Image  # type: ignore
from streamlit_drawable_canvas import st_canvas  # type: ignore

from src.models import Obstacle

# Canvas dimensions match the world exactly — no scaling needed.
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600


def canvas_to_world(cx: float, cy: float) -> Tuple[float, float]:
    """Convert canvas (y-down) to world (y-up) coordinates."""
    return (cx, CANVAS_HEIGHT - cy)


def world_to_canvas(wx: float, wy: float) -> Tuple[float, float]:
    """Convert world (y-up) to canvas (y-down) coordinates."""
    return (wx, CANVAS_HEIGHT - wy)


def _fabric_circle_center(obj: Dict) -> Tuple[float, float, float]:
    """Extract the true center and radius from a fabric.js circle object.

    The streamlit-drawable-canvas circle tool creates circles with
    ``originX="left", originY="center"`` and a rotation ``angle``.
    The center in canvas pixel space is:
        cx = left + radius * cos(angle)
        cy = top  + radius * sin(angle)

    Returns (center_x, center_y, radius) in canvas coordinates.
    """
    left = obj.get("left", 0)
    top = obj.get("top", 0)
    radius = obj.get("radius", 0)
    angle_deg = obj.get("angle", 0)
    scale_x = obj.get("scaleX", 1.0)
    scale_y = obj.get("scaleY", 1.0)
    angle_rad = math.radians(angle_deg)
    cx = left + radius * scale_x * math.cos(angle_rad)
    cy = top + radius * scale_y * math.sin(angle_rad)
    effective_radius = radius * max(scale_x, scale_y)
    return cx, cy, effective_radius


def _fabric_point_center(obj: Dict) -> Tuple[float, float]:
    """Extract the true center from a fabric.js point object.

    The point tool also uses ``originX="left", originY="center"`` with
    ``angle=0``, so the center is simply ``(left + radius, top)``.

    Returns (center_x, center_y) in canvas coordinates.
    """
    left = obj.get("left", 0)
    top = obj.get("top", 0)
    radius = obj.get("radius", 0)
    return (left + radius, top)


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
        r = int(obs["radius"])
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
    st.image(img, use_container_width=True)


def render_custom_canvas(placement_mode: str, obstacle_radius: int = 25) -> None:
    """Display an interactive canvas and process new drawings.

    Uses a static cached background so the canvas never resets its
    internal object state. Object detection is count-based.

    Parameters
    ----------
    placement_mode : str
        One of "Obstacles", "Waypoints", "Start".
    obstacle_radius : int
        Radius for obstacles in world units (= canvas pixels).
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
        point_display_radius=obstacle_radius if placement_mode == "Obstacles" else 8,
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
        if placement_mode == "Obstacles":
            cx, cy, radius = _fabric_circle_center(obj)
            radius = max(radius, 10)
            # Canvas pixels = world units (no scaling)
            wx, wy = canvas_to_world(cx, cy)
            st.session_state.obstacles.append(
                {"x": wx, "y": wy, "radius": radius}
            )

        elif placement_mode == "Waypoints":
            cx, cy = _fabric_point_center(obj)
            wx, wy = canvas_to_world(cx, cy)
            st.session_state.waypoints.append((wx, wy))

        elif placement_mode == "Start":
            cx, cy = _fabric_point_center(obj)
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
