"""Hamilton quaternions, scalar-first [w, x, y, z]; active body->world rotation."""
from __future__ import annotations
import numpy as np


def identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b (broadcasts over leading axes)."""
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def exp_rotvec(rv: np.ndarray) -> np.ndarray:
    """Quaternion for rotation vector rv (angle*axis), broadcasting over leading axes."""
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * th
    # sin(th/2)/th with a series for tiny angles
    s = np.where(th > 1e-8, np.sin(half) / np.where(th > 0, th, 1.0), 0.5 - th**2 / 48.0)
    return np.concatenate([np.cos(half), s * rv], axis=-1)


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R(q) (..., 3, 3) such that v_world = R v_body."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return to_matrix(q) @ v
