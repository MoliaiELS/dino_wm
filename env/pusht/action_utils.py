"""Shared PushT action conventions.

Actions passed to ``PushTEnv.step`` are command-space values.  Relative
actions are bounded displacements which are multiplied by ``action_scale``
to obtain a target displacement in simulator pixels.  Absolute actions are
scaled target positions.
"""

import numpy as np


DEFAULT_ACTION_SCALE = 100.0
DEFAULT_RELATIVE_ACTION_LIMIT = 1.0
DEFAULT_WORKSPACE_SIZE = 512.0


def get_action_bounds(
    relative=True,
    action_scale=DEFAULT_ACTION_SCALE,
    workspace_size=DEFAULT_WORKSPACE_SIZE,
    relative_action_limit=DEFAULT_RELATIVE_ACTION_LIMIT,
):
    """Return lower and upper command-space bounds for a two-dimensional action."""
    action_scale = float(action_scale)
    if action_scale <= 0:
        raise ValueError(f"action_scale must be positive, got {action_scale}")

    if relative:
        limit = float(relative_action_limit)
        if limit <= 0:
            raise ValueError(
                f"relative_action_limit must be positive, got {limit}"
            )
        low = np.full(2, -limit, dtype=np.float64)
        high = np.full(2, limit, dtype=np.float64)
    else:
        high_value = float(workspace_size) / action_scale
        low = np.zeros(2, dtype=np.float64)
        high = np.full(2, high_value, dtype=np.float64)
    return low, high


def clip_action(action, low, high):
    """Validate and clip a single PushT command-space action."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (2,):
        raise ValueError(f"PushT action must have shape (2,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("PushT action must contain only finite values")
    return np.clip(action, np.asarray(low), np.asarray(high))
