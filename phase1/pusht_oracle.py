"""A deterministic geometry-based oracle for the PushT pilot domain."""

import math
from dataclasses import dataclass

import numpy as np

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
    # The controlled pilot moves opposite the T stem direction. The stem's
    # rear support is 120 px from the body origin in the current environment;
    # this is also the fallback if a shape query cannot find the surface.
    rear_support_distance: float = 120.0
    support_ray_length: float = 180.0
    support_ray_resolution: float = 1.0
    clearance_margin: float = 15.0
    orbit_angle_tolerance: float = 0.12
    max_orbit_angle: float = 0.20


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

    def _support_distance(self, env, direction):
        """Find the first exit from the union of block shapes along a ray.

        Pymunk's signed point query avoids the topology instability observed
        when unioning the T's two touching polygons with Shapely.
        """
        center = np.asarray(tuple(env.block.position), dtype=np.float64)

        def inside(distance):
            point = tuple(center + direction * distance)
            return any(shape.point_query(point).distance <= 1e-7 for shape in env.block.shapes)

        previous = 0.0
        seen_inside = inside(0.0)
        distances = np.arange(
            self.config.support_ray_resolution,
            self.config.support_ray_length + self.config.support_ray_resolution,
            self.config.support_ray_resolution,
        )
        for distance in distances:
            is_inside = inside(float(distance))
            if is_inside:
                seen_inside = True
                previous = float(distance)
                continue
            if seen_inside:
                lower, upper = previous, float(distance)
                for _ in range(12):
                    midpoint = (lower + upper) / 2
                    if inside(midpoint):
                        lower = midpoint
                    else:
                        upper = midpoint
                return lower
        return self.config.rear_support_distance

    def staging_point(self, env, push_direction):
        behind = -np.asarray(push_direction, dtype=np.float64)
        support_distance = self._support_distance(env, behind)
        agent_radius = max(
            float(getattr(shape, "radius", 0.0)) for shape in env.agent.shapes
        )
        offset = (
            support_distance
            + agent_radius
            + self.config.staging_margin
        )
        return np.asarray(tuple(env.block.position)) + behind * offset

    @staticmethod
    def _wrapped_angle(angle):
        return (float(angle) + np.pi) % (2 * np.pi) - np.pi

    def _safe_staging_target(self, env, stage):
        """Route around the object instead of crossing it during restaging."""
        block_position = np.asarray(tuple(env.block.position), dtype=np.float64)
        agent_position = np.asarray(tuple(env.agent.position), dtype=np.float64)
        desired_relative = stage - block_position
        current_relative = agent_position - block_position
        current_direction, current_radius = self._unit(current_relative)
        if current_radius < 1e-9:
            current_direction = np.asarray([0.0, 1.0])

        desired_angle = math.atan2(desired_relative[1], desired_relative[0])
        current_angle = math.atan2(current_direction[1], current_direction[0])
        angle_error = self._wrapped_angle(desired_angle - current_angle)
        agent_radius = max(
            float(getattr(shape, "radius", 0.0)) for shape in env.agent.shapes
        )
        clearance_radius = (
            self.config.rear_support_distance
            + agent_radius
            + self.config.staging_margin
            + self.config.clearance_margin
        )

        if abs(angle_error) <= self.config.orbit_angle_tolerance:
            return stage
        if current_radius < clearance_radius - self.config.staging_tolerance:
            return block_position + current_direction * clearance_radius

        next_angle = current_angle + np.clip(
            angle_error,
            -self.config.max_orbit_angle,
            self.config.max_orbit_angle,
        )
        return block_position + clearance_radius * np.asarray(
            [np.cos(next_angle), np.sin(next_angle)]
        )

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
            navigation_target = self._safe_staging_target(env, stage)
            stage_direction, stage_distance = self._unit(
                navigation_target - agent_position
            )
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
