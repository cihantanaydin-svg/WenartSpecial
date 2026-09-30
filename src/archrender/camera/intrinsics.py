"""Camera matrices matching Blender's conventions exactly (horizontal sensor fit).

Blender cameras look down local −Z with +Y up. ``build.py`` sets the Euler XYZ rotation to
``(90° + pitch, 0, yaw − 90°)``. We export OpenCV-convention intrinsics/extrinsics so QA can reproject
known 3D points (tested against Blender's own projection).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.scene import CameraSpec

Mat = NDArray[np.float64]

_BL_TO_CV = np.diag([1.0, -1.0, -1.0])


def _rx(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rz(a: float) -> Mat:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def blender_rotation(cam: CameraSpec) -> Mat:
    """World-from-camera rotation (Blender camera axes), Euler XYZ = Rz · Ry · Rx."""
    return _rz(math.radians(cam.yaw_deg - 90.0)) @ _rx(math.radians(90.0 + cam.pitch_deg))


def intrinsics(cam: CameraSpec, width: int, height: int) -> Mat:
    fx = cam.focal_mm / cam.sensor_width_mm * width
    cx = width / 2.0 - cam.shift_x * width
    cy = height / 2.0 + cam.shift_y * width
    return np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])


def extrinsics_cv(cam: CameraSpec) -> tuple[Mat, Mat]:
    """OpenCV ``R, t`` with ``x_cam = R · X_world + t`` (x right, y down, z forward)."""
    r_wc = blender_rotation(cam)
    c = np.array(cam.position.as_tuple())
    r = _BL_TO_CV @ r_wc.T
    return r, -r @ c


def project(cam: CameraSpec, width: int, height: int, points: Mat) -> tuple[Mat, Mat]:
    """Project world points (N,3) → pixel coords (N,2) and camera depth z (N,)."""
    k = intrinsics(cam, width, height)
    r, t = extrinsics_cv(cam)
    xc = points @ r.T + t
    z = xc[:, 2]
    uv = (xc[:, :2] / z[:, None]) * np.array([k[0, 0], k[1, 1]]) + np.array([k[0, 2], k[1, 2]])
    return uv, z


def camera_json(cam: CameraSpec, width: int, height: int) -> dict[str, object]:
    k = intrinsics(cam, width, height)
    r, t = extrinsics_cv(cam)
    return {
        "camera_id": cam.id,
        "width": width,
        "height": height,
        "K": k.tolist(),
        "R_cv": r.tolist(),
        "t_cv": t.tolist(),
        "blender": {
            "location": list(cam.position.as_tuple()),
            "rotation_euler_xyz_deg": [90.0 + cam.pitch_deg, 0.0, cam.yaw_deg - 90.0],
            "focal_mm": cam.focal_mm,
            "sensor_width_mm": cam.sensor_width_mm,
            "sensor_fit": "HORIZONTAL",
            "shift_x": cam.shift_x,
            "shift_y": cam.shift_y,
        },
        "clip": [cam.clip_start, cam.clip_end],
    }


def horizontal_fov_deg(cam: CameraSpec) -> float:
    return math.degrees(2.0 * math.atan(cam.sensor_width_mm / (2.0 * cam.focal_mm)))
