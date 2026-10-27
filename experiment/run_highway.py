# -*- coding: utf-8 -*-
"""Train and evaluate PPO highway driving with a predictive safety filter.

Run from the IEEE Access directory:
    python experiment/run_highway.py --mode pilot
    python experiment/run_highway.py --mode full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401  registers highway-v0
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from safety_filter import ACTIONS, PredictiveSafetyFilter

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4, 5]


def base_config(vehicles_count: int = 20, collision_reward: float = 0.0) -> dict:
    return {
        "lanes_count": 4,
        "vehicles_count": int(vehicles_count),
        "duration": 40,
        "simulation_frequency": 5,
        "policy_frequency": 1,
        "vehicles_density": 1.5,
        "ego_spacing": 1.5,
        "offroad_terminal": True,
        "normalize_reward": False,
        "collision_reward": float(collision_reward),
        "right_lane_reward": 0.0,
        "high_speed_reward": 1.0,
        "lane_change_reward": 0.0,
        "reward_speed_range": [20, 30],
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 5,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
            "normalize": True,
            "see_behind": True,
        },
        "action": {"type": "DiscreteMetaAction"},
    }


def make_env(config: dict, seed: int, filter_name: str | None = None, rank: int = 0):
    def _init():
        env = gym.make("highway-v0")
        env.unwrapped.configure(config)
        env = EgoCollisionEnv(env)
        if filter_name is not None:
            env = FilterEnv(env, **FILTERS[filter_name])
        env.reset(seed=seed + rank)
        return env

    return _init


def _ego_only_collisions(env) -> None:
    raw = env.unwrapped
    ego = raw.vehicle
    for vehicle in raw.road.vehicles:
        if vehicle is not ego:
            vehicle.check_collisions = False


class EgoCollisionEnv(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        _ego_only_collisions(self)
        return obs, info


class FilterEnv(gym.Wrapper):
    """Applies the filter as an action mask. Sampling stays inside the mask."""

    def __init__(self, env, **filter_kwargs):
        super().__init__(env)
        self.filter = PredictiveSafetyFilter(**filter_kwargs)
        self.filter.bind(self.env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.filter.bind(self.env)
        _ego_only_collisions(self.env)
        return obs, info

    def action_masks(self) -> np.ndarray:
        mask, _executed, _info = self.filter.mask_and_choice(1)
        return mask

    def step(self, action):
        _mask, executed, shield_info = self.filter.mask_and_choice(int(action))
        obs, reward, terminated, truncated, info = self.env.step(executed)
        info = dict(info)
        info.update(shield_info)
        return obs, reward, terminated, truncated, info


def ppo_kwargs(seed: int) -> dict:
    return dict(
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=seed,
        verbose=0,
        device="cpu",
    )


def train_one(name: str, seed: int, config: dict, filter_name: str | None, timesteps: int, n_envs: int) -> Path:
    path = MODELS / f"{name}_seed{seed}.zip"
    log_path = RESULTS / f"train_{name}_seed{seed}.json"
    history = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    done = int(history[-1]["timesteps"]) if history else 0
    if done >= timesteps and path.exists():
        print(f"skip train {name} seed {seed} at {done}", flush=True)
        return path
    torch.manual_seed(seed)
    np.random.seed(seed)
    vec = DummyVecEnv([make_env(config, seed, filter_name, rank) for rank in range(n_envs)])
    algo = MaskablePPO if filter_name else PPO
    if path.exists() and done > 0:
        model = algo.load(path, env=vec, device="cpu")
        print(f"resume {name} seed {seed} from {done}", flush=True)
    else:
        model = algo("MlpPolicy", vec, **ppo_kwargs(seed))
        done = 0
        history = []
    chunk = 10_000
    while done < timesteps:
        step = min(chunk, timesteps - done)
        model.learn(total_timesteps=step, reset_num_timesteps=(done == 0), progress_bar=False)
        done += step
        metrics = evaluate_model(model, config, seed=10_000 + seed, episodes=8, filter_name=filter_name, masked=bool(filter_name))
        metrics["timesteps"] = done
        history.append(metrics)
        print(
            f"train {name} seed {seed} steps {done} "
            f"crash {metrics['collision_rate']:.2f} speed {metrics['mean_speed']:.2f}",
            flush=True,
        )
        model.save(path)
        log_path.write_text(json.dumps(history), encoding="utf-8")
    model.save(path)
    log_path.write_text(json.dumps(history), encoding="utf-8")
    vec.close()
    return path


def _heuristic_action(env) -> int:
    raw = env.unwrapped
    ego = raw.vehicle
    road = raw.road
    available = set(raw.action_type.get_available_actions())

    def gap(lane_index):
        front, _rear = road.neighbour_vehicles(ego, lane_index)
        if front is None:
            return 1.0e3
        return float(ego.lane_distance_to(front))

    current = ego.lane_index
    front_gap = gap(current)
    desired = 8.0 + 1.2 * float(ego.speed)
    choice = 1
    if front_gap < desired:
        side_gaps = []
        for lane_index in road.network.side_lanes(current):
            side_gaps.append((gap(lane_index), lane_index[2]))
        if side_gaps:
            best_gap, best_id = max(side_gaps, key=lambda item: item[0])
            if best_gap > front_gap + 8.0:
                choice = 0 if best_id < current[2] else 2
            elif 4 in available:
                choice = 4
        elif 4 in available:
            choice = 4
    elif front_gap > desired + 12.0 and 3 in available and ego.speed < 29.0:
        choice = 3
    if choice not in available:
        choice = 1 if 1 in available else next(iter(available))
    return int(choice)


def evaluate_model(model, config, seed: int, episodes: int, filter_name: str | None, masked: bool) -> dict:
    env = gym.make("highway-v0")
    env.unwrapped.configure(config)
    env = EgoCollisionEnv(env)
    shield = None
    if filter_name is not None:
        env = FilterEnv(env, **FILTERS[filter_name])
        shield = env.filter
    rows = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        crashed = False
        offroad = False
        speeds = []
        interventions = 0
        safety_interventions = 0
        decisions = 0
        episode_return = 0.0
        done = False
        while not done:
            if model is None:
                action = _heuristic_action(env)
            elif masked:
                action, _state = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            else:
                action, _state = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).reshape(-1)[0])
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            speeds.append(float(info["speed"]))
            decisions += 1
            interventions += int(info.get("intervened", False))
            safety_interventions += int(info.get("safety_intervention", False))
            crashed = crashed or bool(info.get("crashed", False))
            done = bool(terminated or truncated)
            if done and (not info.get("crashed", False)) and (not env.unwrapped.vehicle.on_road):
                offroad = True
        rows.append(
            {
                "crashed": crashed,
                "offroad": offroad,
                "success": (not crashed) and (not offroad),
                "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                "distance": float(np.sum(speeds)),
                "steps": decisions,
                "intervention_rate": interventions / max(decisions, 1),
                "safety_intervention_rate": safety_interventions / max(decisions, 1),
                "return": float(episode_return),
            }
        )
    env.close()
    return _aggregate(rows)


def evaluate_heuristic(config, seed: int, episodes: int, filter_name: str | None = None) -> dict:
    return evaluate_model(None, config, seed, episodes, filter_name, masked=False)


def _aggregate(rows: list[dict]) -> dict:
    def mean(key):
        return float(np.mean([row[key] for row in rows]))

    return {
        "episodes": len(rows),
        "collision_rate": mean("crashed"),
        "offroad_rate": mean("offroad"),
        "success_rate": mean("success"),
        "mean_speed": mean("mean_speed"),
        "mean_distance": mean("distance"),
        "mean_steps": mean("steps"),
        "intervention_rate": mean("intervention_rate"),
        "safety_intervention_rate": mean("safety_intervention_rate"),
        "mean_return": mean("return") if "return" in rows[0] else float("nan"),
    }


def load_model(name: str, seed: int, masked: bool):
    path = MODELS / f"{name}_seed{seed}.zip"
    algo = MaskablePPO if masked else PPO
    return algo.load(path, device="cpu")


def invariance_check(episodes: int = 5) -> dict:
    config = base_config()
    env = gym.make("highway-v0")
    env.unwrapped.configure(config)
    shield = PredictiveSafetyFilter(horizon=2.0, margin=1.0, predictor="brake")
    errors = []
    max_drift = 0.0
    for episode in range(episodes):
        env.reset(seed=episode)
        shield.bind(env)
        for _ in range(5):
            ego = env.unwrapped.vehicle
            before = np.array(ego.position, dtype=np.float64)
            others = shield._neighbors(ego)
            before_others = [np.array(vehicle.position, dtype=np.float64) for vehicle in others]
            shield.assess()
            drift = float(np.linalg.norm(ego.position - before))
            max_drift = max(max_drift, drift)
            for vehicle, start in zip(shield._neighbors(ego), before_others):
                max_drift = max(max_drift, float(np.linalg.norm(vehicle.position - start)))
            action = int(env.action_space.sample())
            predicted_hit, predicted_off, _distance = shield._rollout(action)
            snap_pos = np.array(env.unwrapped.vehicle.position)
            env.step(action)
            actual = np.array(env.unwrapped.vehicle.position)
            # One policy step is 1 second. Compare only the stored prediction horizon if it equals 1.
            errors.append(
                {
                    "predicted_hit": predicted_hit,
                    "predicted_off": predicted_off,
                    "moved": float(np.linalg.norm(actual - snap_pos)),
                }
            )
            if env.unwrapped.vehicle.crashed or not env.unwrapped.vehicle.on_road:
                break
    env.close()
    return {"max_state_drift_m": max_drift, "samples": len(errors)}


def prediction_error(episodes: int = 30) -> dict:
    """Ego position error of a 1 second constant-velocity-other rollout versus the true step.

    The action is IDLE, so the ego controller is the quantity being checked.
    Other vehicles differ on purpose whenever they brake or change lane.
    """
    config = base_config()
    env = gym.make("highway-v0")
    env.unwrapped.configure(config)
    shield = PredictiveSafetyFilter(horizon=1.0, margin=0.0, predictor="cv", dt=0.2)
    ego_errors = []
    for episode in range(episodes):
        env.reset(seed=100 + episode)
        shield.bind(env)
        ego = env.unwrapped.vehicle
        others = shield._neighbors(ego)
        tracked = [ego, *others]
        from safety_filter import _restore, _snapshot

        snaps = [_snapshot(vehicle) for vehicle in tracked]
        dt = 0.2
        for index in range(5):
            ego.act(None if index else "IDLE")
            for other in others:
                other.action = {"steering": 0.0, "acceleration": 0.0}
            ego.step(dt)
            for other in others:
                other.step(dt)
        predicted = np.array(ego.position, dtype=np.float64)
        for vehicle, snap in zip(tracked, snaps):
            _restore(vehicle, snap)
        env.step(1)  # IDLE
        if env.unwrapped.vehicle.crashed:
            continue
        actual = np.array(env.unwrapped.vehicle.position, dtype=np.float64)
        ego_errors.append(float(np.linalg.norm(actual - predicted)))
    env.close()
    values = np.array(ego_errors) if ego_errors else np.array([np.nan])
    return {
        "episodes": int(len(ego_errors)),
        "ego_position_error_mean_m": float(np.mean(values)),
        "ego_position_error_std_m": float(np.std(values)),
        "ego_position_error_p95_m": float(np.quantile(values, 0.95)),
    }


def _snapshot_public(vehicle):
    from safety_filter import _snapshot

    return _snapshot(vehicle)


JOBS = [
    ("speed", 0.0, None),
    ("penalty", -1.0, None),
    ("heavy", -5.0, None),
    ("shielded", 0.0, "brake_h3_m1"),
]
EXTRA_JOBS = [
    ("severe", -30.0, None),
]
COLLISION = {name: collision for name, collision, _filter in JOBS + EXTRA_JOBS}


def filter_specs() -> dict:
    specs = {}
    for horizon in (1.0, 2.0, 3.0):
        for margin in (0.0, 1.0, 2.0):
            for predictor in ("cv", "brake"):
                key = f"{predictor}_h{horizon:g}_m{margin:g}"
                specs[key] = dict(
                    horizon=horizon,
                    margin=margin,
                    predictor=predictor,
                    leader_brake=3.0,
                )
    return specs


FILTERS = filter_specs()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["check", "worker", "full", "extra", "sweep", "sweep_seed"],
        default="check",
    )
    parser.add_argument("--name", default="speed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=60_000)
    args = parser.parse_args()
    if args.mode == "check":
        print("invariance", json.dumps(invariance_check()), flush=True)
        print("prediction", json.dumps(prediction_error()), flush=True)
        return
    if args.mode == "worker":
        collision = COLLISION[args.name]
        filter_name = "brake_h3_m1" if args.name == "shielded" else None
        train_one(args.name, args.seed, base_config(collision_reward=collision), filter_name, args.timesteps, n_envs=1)
        return
    if args.mode == "full":
        _run_pool(args.timesteps, JOBS)
        return
    if args.mode == "extra":
        _run_pool(args.timesteps, EXTRA_JOBS, workers=2)
        return
    if args.mode == "sweep_seed":
        _run_sweep_seed(args.seed)
        return
    if args.mode == "sweep":
        _run_sweep()
        return


def _run_pool(timesteps: int, jobs, workers: int = 4) -> None:
    import subprocess
    import sys

    commands = []
    for seed in SEEDS:
        for name, _collision, _filter in jobs:
            commands.append(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "--mode",
                    "worker",
                    "--name",
                    name,
                    "--seed",
                    str(seed),
                    "--timesteps",
                    str(timesteps),
                ]
            )
    active = []
    for command in commands:
        while len(active) >= workers:
            finished = [proc for proc in active if proc.poll() is not None]
            if not finished:
                try:
                    active[0].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                finished = [proc for proc in active if proc.poll() is not None]
            for proc in finished:
                if proc.returncode != 0:
                    raise RuntimeError(f"worker failed with code {proc.returncode}")
            active = [proc for proc in active if proc.poll() is None]
        print("launch", command[-6:], flush=True)
        active.append(subprocess.Popen(command, cwd=str(ROOT)))
    for proc in active:
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"worker failed with code {proc.returncode}")


def _sweep_jobs():
    return JOBS + EXTRA_JOBS


def _run_sweep_seed(seed: int, episodes: int = 60) -> None:
    partial = RESULTS / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    out = partial / f"sweep_seed{seed}.json"
    final = []
    for name, collision_reward, trained_filter in _sweep_jobs():
        model_path = MODELS / f"{name}_seed{seed}.zip"
        if not model_path.exists():
            print(f"skip missing {model_path.name}", flush=True)
            continue
        model = load_model(name, seed, masked=trained_filter is not None)
        variants = []
        if name == "speed":
            variants.append((None, False))
            for key in FILTERS:
                variants.append((key, False))
        elif name == "shielded":
            variants.append(("brake_h3_m1", True))
            variants.append((None, False))
        else:
            variants.append((None, False))
        for filter_name, masked in variants:
            densities = (12, 20, 30) if filter_name in {None, "brake_h3_m1", "cv_h3_m1"} else (20,)
            for vehicles in densities:
                config = base_config(vehicles_count=vehicles, collision_reward=collision_reward)
                metrics = evaluate_model(
                    model,
                    config,
                    seed=50_000 + seed,
                    episodes=episodes,
                    filter_name=filter_name,
                    masked=masked,
                )
                record = {
                    "policy": name,
                    "seed": seed,
                    "eval_filter": filter_name,
                    "masked": masked,
                    "vehicles_count": vehicles,
                    **metrics,
                }
                final.append(record)
                print("eval", json.dumps(record), flush=True)
                out.write_text(json.dumps(final, indent=2), encoding="utf-8")
    for vehicles in (12, 20, 30):
        heuristic = evaluate_heuristic(
            base_config(vehicles_count=vehicles, collision_reward=0.0),
            seed=50_000 + seed,
            episodes=episodes,
        )
        heuristic.update(
            {
                "policy": "heuristic",
                "seed": seed,
                "eval_filter": None,
                "masked": False,
                "vehicles_count": vehicles,
            }
        )
        final.append(heuristic)
        print("eval", json.dumps(heuristic), flush=True)
        out.write_text(json.dumps(final, indent=2), encoding="utf-8")


def _run_sweep() -> None:
    import subprocess
    import sys

    partial = RESULTS / "partial"
    partial.mkdir(parents=True, exist_ok=True)
    commands = [
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--mode",
            "sweep_seed",
            "--seed",
            str(seed),
        ]
        for seed in SEEDS
    ]
    active = []
    for command in commands:
        while len(active) >= 3:
            finished = [proc for proc in active if proc.poll() is not None]
            if not finished:
                active[0].wait()
                finished = [proc for proc in active if proc.poll() is not None]
            for proc in finished:
                if proc.returncode != 0:
                    raise RuntimeError(f"sweep worker failed with code {proc.returncode}")
            active = [proc for proc in active if proc.poll() is None]
        print("launch", command[-2:], flush=True)
        active.append(subprocess.Popen(command, cwd=str(ROOT)))
    for proc in active:
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"sweep worker failed with code {proc.returncode}")
    final = []
    for seed in SEEDS:
        path = partial / f"sweep_seed{seed}.json"
        final.extend(json.loads(path.read_text(encoding="utf-8")))
    (RESULTS / "sweep.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(f"sweep records {len(final)}", flush=True)


if __name__ == "__main__":
    main()
