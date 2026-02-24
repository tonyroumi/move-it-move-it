
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Quaternion helpers  (scalar-first: [w, x, y, z])
# ---------------------------------------------------------------------------

def normalize_angle(x):
    return np.arctan2(np.sin(x), np.cos(x))

def normalize_quat(x, eps: float = 1e-9):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.clip(norms, a_min=eps, a_max=None)
    return x / norms

def normalize_exp_map(exp_map):
    angle = np.linalg.norm(exp_map, axis=-1)
    angle = np.maximum(angle, 1e-9)
    norm_angle = normalize_angle(angle)
    scale = norm_angle / angle
    norm_exp_map = exp_map * scale[..., np.newaxis]
    return norm_exp_map

def quat_pos(x):
    q = x.copy()
    z = (q[..., 3:] < 0).astype(q.dtype)
    q = (1 - 2 * z) * q
    return q

def quat_mul(q1, q2):
    """Hamilton product of two quaternions (scalar-first)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)

def quat_unit(a):
    return normalize_quat(a)

def quat_normalize(q):
    q = quat_unit(quat_pos(q))
    return q

def quat_conjugate(q):
    """Conjugate (inverse for unit quaternions), scalar-first."""
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_rotate(q, v):
    """Rotate vector *v* by quaternion *q*.  Both broadcastable."""
    q_v = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return quat_mul(quat_mul(q, q_v), quat_conjugate(q))[..., 1:]


def quat_to_matrix(q):
    """Convert scalar-first quaternion to 3x3 rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - z*w)
    R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] = 2*(x*y + z*w)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] = 2*(x*z - y*w)
    R[..., 2, 1] = 2*(y*z + x*w)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R

def quat_diff(q0, q1):
    dq = quat_mul(q1, quat_conjugate(q0))
    return dq

def quat_to_axis_angle(q):
    eps = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    q = quat_pos(q)
    length = np.linalg.norm(q[..., qx:qw], axis=-1)

    angle = 2.0 * np.arctan2(length, q[..., qw])
    axis = q[..., qx:qw] / length[..., np.newaxis]

    default_axis = np.zeros_like(axis)
    default_axis[..., -1] = 1
    mask = length > eps

    angle = np.where(mask, angle, np.zeros_like(angle))
    mask_expand = mask[..., np.newaxis]
    axis = np.where(mask_expand, axis, default_axis)

    return axis, angle

def quat_to_exp_map(q):
    axis, angle = quat_to_axis_angle(q)
    exp_map = axis_angle_to_exp_map(axis, angle)
    return exp_map

def exp_map_to_axis_angle(exp_map):
    min_theta = 1e-5

    angle = np.linalg.norm(exp_map, axis=-1)
    angle_exp = angle[..., np.newaxis]
    axis = exp_map / angle_exp
    angle = normalize_angle(angle)

    default_axis = np.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = np.abs(angle) > min_theta
    angle = np.where(mask, angle, np.zeros_like(angle))
    mask_expand = mask[..., np.newaxis]
    axis = np.where(mask_expand, axis, default_axis)

    return axis, angle

def axis_angle_to_exp_map(axis, angle):
    angle_expand = angle[..., np.newaxis]
    exp_map = angle_expand * axis
    return exp_map

def axis_angle_to_quat(axis, angle):
    theta = (angle / 2)[..., np.newaxis]
    xyz = normalize_quat(axis) * np.sin(theta)
    w = np.cos(theta)
    return quat_unit(np.concatenate([xyz, w], axis=-1))

def calc_heading(q):
    ref_dir = np.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading

def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = np.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = axis_angle_to_quat(axis, -heading)
    return heading_q

def quat_to_tan_norm(q):
    ref_tan = np.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = np.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = np.concatenate([tan, norm], axis=len(tan.shape) - 1)
    return norm_tan

def heading_quat_from_root(root_quat):
    """
    Extract the yaw-only (heading) quaternion from a full root orientation.

    Projects the root's local x-axis onto the ground plane to determine
    the heading angle, then returns a quaternion that represents *only*
    the rotation about the world z-axis by that angle.

    Parameters
    ----------
    root_quat : ndarray, shape (..., 4)
        Root orientation as a scalar-first quaternion.

    Returns
    -------
    heading_q : ndarray, shape (..., 4)
        Yaw-only quaternion (rotation about z).
    """
    forward_local = np.zeros(root_quat.shape[:-1] + (3,), dtype=root_quat.dtype)
    forward_local[..., 0] = 1.0
    forward_world = quat_rotate(root_quat, forward_local)

    heading_angle = np.arctan2(forward_world[..., 1], forward_world[..., 0])

    half = heading_angle / 2.0
    heading_q = np.zeros_like(root_quat)
    heading_q[..., 0] = np.cos(half)
    heading_q[..., 3] = np.sin(half)
    return heading_q


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def compute_root_relative_frame(root_pos, root_quat):
    """
    Compute the transform that places the origin at the root and aligns
    the x-axis with the root's facing (heading) direction.

    Parameters
    ----------
    root_pos : (..., 3)
    root_quat : (..., 4)  scalar-first

    Returns
    -------
    inv_heading_q : (..., 4)
    origin        : (..., 3)
    """
    heading_q = heading_quat_from_root(root_quat)
    inv_heading_q = quat_conjugate(heading_q)

    origin = root_pos.copy()
    origin[..., 2] = 0.0

    return inv_heading_q, origin


def transform_positions_to_root_frame(positions, root_pos, root_quat):
    """
    Transform world-frame positions into the root-relative heading frame.

    Parameters
    ----------
    positions : (..., N, 3) or (..., 3)
    root_pos  : (..., 3)
    root_quat : (..., 4)

    Returns
    -------
    local_positions : same shape as *positions*
    """
    inv_heading_q, origin = compute_root_relative_frame(root_pos, root_quat)

    shifted = positions - np.expand_dims(origin, -2) if positions.ndim > root_pos.ndim else positions - origin

    if shifted.ndim > inv_heading_q.ndim:
        q = np.broadcast_to(np.expand_dims(inv_heading_q, -2), shifted.shape[:-1] + (4,))
    else:
        q = inv_heading_q

    return quat_rotate(q, shifted)


def transform_velocities_to_root_frame(velocities, root_quat):
    """
    Rotate world-frame velocities into the root-relative heading frame.
    (Velocities are direction-only, no translation offset.)
    """
    inv_heading_q, _ = compute_root_relative_frame(
        np.zeros_like(root_quat[..., :3]), root_quat
    )
    if velocities.ndim > inv_heading_q.ndim:
        q = np.broadcast_to(np.expand_dims(inv_heading_q, -2), velocities.shape[:-1] + (4,))
    else:
        q = inv_heading_q
    return quat_rotate(q, velocities)

def transform_rotations_to_root_frame(quats, root_quat):
    """
    Express body orientations relative to the root heading frame.

    Parameters
    ----------
    quats     : (..., N, 4) or (..., 4)  body orientations in world frame
    root_quat : (..., 4)

    Returns
    -------
    local_quats : same shape as *quats*
    """
    inv_heading_q, _ = compute_root_relative_frame(
        np.zeros_like(root_quat[..., :3]), root_quat
    )
    if quats.ndim > inv_heading_q.ndim:
        q = np.broadcast_to(np.expand_dims(inv_heading_q, -2), quats.shape[:-1] + (4,))
    else:
        q = inv_heading_q
    return quat_mul(q, quats)
