"""A deterministic geometry-based oracle for the PushT pilot domain."""

from dataclasses import dataclass

import numpy as np
import shapely.geometry as sg
from shapely.ops import unary_union

from env.pusht.action_utils import clip_action


@dataclass(frozen=True)
class OracleConfig:
    """Controller settings expressed in simulator pixels and command units."""

    staging_margin: float = 4.0
    staging_tolerance: float = 7.0
    max_staging_command: float = 0.85
    max_push_command: float = 0.55
    min_push_command: float = 0.06
    push_gain: float = 0.75
    ray_length: float = 180.0


def _iter_coordinates(geometry):
    if geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for child in geometry.geoms:
            yield from _iter_coordinates(child)
        return
    if hasattr(geometry, "coords"):
        for coordinate in geometry.coords:
            yield np.asarray(coordinate[:2], dtype=np.float64)


class GeometricPushTOracle:
    """Restage behind the object, then push through its center toward the goal.

    The Phase 1 pilot deliberately samples goal-aligned object orientations.
    This controller is therefore an oracle for the controlled translation
    domain, not a claim of solving arbitrary planar manipulation.
    """

    def __init__(self, config=None):
        self.config = OracleConfig() if config is None else config

    @staticmethod
    def _unit(vector):
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return np.zeros(2, dtype=np.float64), norm
        return np.asarray(vector, dtype=np.float64) / norm, norm

    @staticmethod
    def _block_geometry(env):
        polygons = []
        for shape in env.block.shapes:
            vertices = [env.block.local_to_world(v) for v in shape.get_vertices()]
            polygons.append(sg.Polygon(vertices))
        return unary_union(polygons)

    def _support_distance(self, env, direction):
        center = np.asarray(tuple(env.block.position), dtype=np.float64)
        endpoint = center + direction * self.config.ray_length
        ray = sg.LineString([center, endpoint])
        intersection = self._block_geometry(env).boundary.intersection(ray)
        projections = []
        for coordinate in _iter_coordinates(intersection):
            projection = float(np.dot(coordinate - center, direction))
            if projection > 1e-6:
                projections.append(projection)
        if projections:
            return min(projections)

        # A conservative fallback for unusual shapes whose body origin lies
        # outside the polygon union.
        return 70.0

    def staging_point(self, env, push_direction):
        behind = -np.asarray(push_direction, dtype=np.float64)
        support_distance = self._support_distance(env, behind)
        agent_radius = max(
            float(getattr(shape, "radius", 0.0)) for shape in env.agent.shapes
        )
        offset = support_distance + agent_radius + self.config.staging_margin
        return np.asarray(tuple(env.block.position)) + behind * offset

    def act(self, env):
        """Return one bounded relative command using current simulator state."""
        if env.check_success():
            return np.zeros(2, dtype=np.float64)

        block_position = np.asarray(tuple(env.block.position), dtype=np.float64)
        goal_position = np.asarray(env.goal_pose[:2], dtype=np.float64)
        push_direction, goal_distance = self._unit(goal_position - block_position)
        if goal_distance < 1e-9:
            return np.zeros(2, dtype=np.float64)

        stage = self.staging_point(env, push_direction)
        agent_position = np.asarray(tuple(env.agent.position), dtype=np.float64)
        stage_direction, stage_distance = self._unit(stage - agent_position)

        if stage_distance > self.config.staging_tolerance:
            magnitude = min(
                self.config.max_staging_command,
                stage_distance / env.action_scale,
            )
            command = stage_direction * magnitude
        else:
            magnitude = np.clip(
                self.config.push_gain * goal_distance / env.action_scale,
                self.config.min_push_command,
                self.config.max_push_command,
            )
            command = push_direction * magnitude

        return clip_action(command, env.action_space.low, env.action_space.high)
