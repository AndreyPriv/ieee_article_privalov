# -*- coding: utf-8 -*-
"""Short-horizon predictive safety filter for highway-env discrete meta-actions.

The ego vehicle is rolled forward with the same low-level lane and speed
controllers used by the simulator. Surrounding vehicles are not rolled with
their IDM/MOBIL drivers. They are predicted either at constant velocity or,
for vehicles ahead, with a prescribed braking deceleration.
"""

from __future__ import annotations

import numpy as np

ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
PREFERENCE = (4, 1, 0, 2, 3)  # SLOWER, IDLE, LANE_LEFT, LANE_RIGHT, FASTER


def _snapshot(vehicle) -> dict:
    data = {
        "position": np.array(vehicle.position, dtype=np.float64),
        "heading": float(vehicle.heading),
        "speed": float(vehicle.speed),
        "action": {
            "steering": float(vehicle.action["steering"]),
            "acceleration": float(vehicle.action["acceleration"]),
        },
        "crashed": bool(vehicle.crashed),
        "hit": bool(getattr(vehicle, "hit", False)),
        "impact": None if vehicle.impact is None else np.array(vehicle.impact, dtype=np.float64),
        "lane_index": vehicle.lane_index,
    }
    if hasattr(vehicle, "target_lane_index"):
        data["target_lane_index"] = vehicle.target_lane_index
        data["target_speed"] = float(vehicle.target_speed)
    if hasattr(vehicle, "speed_index"):
        data["speed_index"] = int(vehicle.speed_index)
    return data


def _restore(vehicle, data: dict) -> None:
    vehicle.position = np.array(data["position"], dtype=np.float64)
    vehicle.heading = data["heading"]
    vehicle.speed = data["speed"]
    vehicle.action = {
        "steering": data["action"]["steering"],
        "acceleration": data["action"]["acceleration"],
    }
    vehicle.crashed = data["crashed"]
    vehicle.hit = data["hit"]
    vehicle.impact = None if data["impact"] is None else np.array(data["impact"], dtype=np.float64)
    vehicle.lane_index = data["lane_index"]
    if vehicle.road is not None and vehicle.lane_index is not None:
        vehicle.lane = vehicle.road.network.get_lane(vehicle.lane_index)
    if "target_lane_index" in data:
        vehicle.target_lane_index = data["target_lane_index"]
        vehicle.target_speed = data["target_speed"]
    if "speed_index" in data:
        vehicle.speed_index = data["speed_index"]


def _polygon(vehicle, margin: float) -> np.ndarray:
    length = vehicle.LENGTH + margin
    width = vehicle.WIDTH + margin
    points = np.array(
        [
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
            [length / 2, -width / 2],
        ]
    ).T
    cosine, sine = np.cos(vehicle.heading), np.sin(vehicle.heading)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    points = (rotation @ points).T + vehicle.position
    return np.vstack([points, points[0:1]])


def _hits(ego, other, margin: float, dt: float) -> bool:
    from highway_env import utils

    if np.linalg.norm(other.position - ego.position) > 25.0:
        return False
    intersecting, _will, _transition = utils.are_polygons_intersecting(
        _polygon(ego, margin),
        other.polygon(),
        ego.velocity * dt,
        other.velocity * dt,
    )
    return bool(intersecting)


class PredictiveSafetyFilter:
    def __init__(
        self,
        horizon: float = 3.0,
        margin: float = 1.0,
        predictor: str = "brake",
        leader_brake: float = 3.0,
        lateral_window: float = 6.0,
        dt: float = 0.2,
        neighbor_count: int = 4,
        neighbor_range: float = 120.0,
    ) -> None:
        if predictor not in {"cv", "brake"}:
            raise ValueError(predictor)
        self.horizon = float(horizon)
        self.margin = float(margin)
        self.predictor = predictor
        self.leader_brake = float(leader_brake)
        self.lateral_window = float(lateral_window)
        self.dt = float(dt)
        self.neighbor_count = int(neighbor_count)
        self.neighbor_range = float(neighbor_range)

    def bind(self, env):
        self._env = env.unwrapped

    def _neighbors(self, ego):
        road = self._env.road
        found = road.close_vehicles_to(
            ego,
            self.neighbor_range,
            count=self.neighbor_count,
            see_behind=True,
            sort=True,
        )
        return list(found)

    def _rollout(self, action_id: int) -> tuple[bool, bool, float]:
        env = self._env
        ego = env.vehicle
        others = self._neighbors(ego)
        tracked = [ego, *others]
        snaps = [_snapshot(vehicle) for vehicle in tracked]
        label = ACTIONS[int(action_id)]
        steps = max(1, int(round(self.horizon / self.dt)))
        hit = False
        offroad = False
        min_distance = np.inf
        try:
            for index in range(steps):
                if index == 0:
                    ego.act(label)
                else:
                    ego.act(None)
                for other in others:
                    ahead = float(ego.front_distance_to(other)) > 0.0
                    lateral = abs(float(other.position[1] - ego.position[1]))
                    braking = self.predictor == "brake" and ahead and lateral <= self.lateral_window
                    other.action = {
                        "steering": 0.0,
                        "acceleration": -self.leader_brake if braking else 0.0,
                    }
                ego.step(self.dt)
                for other in others:
                    other.step(self.dt)
                if not bool(ego.on_road):
                    offroad = True
                for other in others:
                    distance = float(np.linalg.norm(other.position - ego.position))
                    min_distance = min(min_distance, distance)
                    if _hits(ego, other, self.margin, self.dt):
                        hit = True
                if hit or offroad:
                    break
        finally:
            for vehicle, snap in zip(tracked, snaps):
                _restore(vehicle, snap)
        return hit, offroad, float(min_distance)

    def assess(self) -> dict[int, tuple[bool, bool, float]]:
        available = set(self._env.action_type.get_available_actions())
        report = {}
        for action_id in ACTIONS:
            if action_id not in available:
                report[action_id] = (True, True, -1.0)
                continue
            hit, offroad, distance = self._rollout(action_id)
            report[action_id] = (hit, offroad, distance)
        return report

    def mask_and_choice(self, proposed: int) -> tuple[np.ndarray, int, dict]:
        report = self.assess()
        safe = [
            action_id
            for action_id, (hit, offroad, _distance) in report.items()
            if not hit and not offroad
        ]
        if not safe:
            ranked = sorted(
                ACTIONS,
                key=lambda action_id: (
                    report[action_id][0] or report[action_id][1],
                    -report[action_id][2],
                    PREFERENCE.index(action_id),
                ),
            )
            safe = [ranked[0]]
        mask = np.zeros(len(ACTIONS), dtype=bool)
        mask[safe] = True
        proposed = int(proposed)
        available = set(self._env.action_type.get_available_actions())
        if proposed in safe:
            executed = proposed
        else:
            executed = sorted(safe, key=lambda action_id: (PREFERENCE.index(action_id), -report[action_id][2]))[0]
        info = {
            "proposed_action": proposed,
            "executed_action": int(executed),
            "intervened": int(executed) != proposed,
            "safety_intervention": proposed in available and int(executed) != proposed,
            "availability_correction": proposed not in available,
            "n_safe_actions": int(len([a for a in safe if not report[a][0] and not report[a][1]])),
            "proposed_predicted_hit": bool(report[proposed][0] or report[proposed][1]),
        }
        return mask, int(executed), info
