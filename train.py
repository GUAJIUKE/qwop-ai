"""QWOP PPO 训练脚本 (stable-baselines3)。

用法:
    python train.py                          # 新训练, 默认 4 个并行环境
    python train.py --n-envs 8               # 8 个并行浏览器
    python train.py --resume                 # 从最新 checkpoint 续训
    python train.py --timesteps 5000000      # 指定训练步数

监控:
    tensorboard --logdir logs
"""
from __future__ import annotations

# ---- 依赖自检: 若当前 Python 未安装 RL 库, 自动用隔离 venv 重跑 ----
import sys, os, subprocess
_VENV = r"C:\Users\24479\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
try:
    import stable_baselines3  # noqa: F401
except ImportError:
    if os.path.exists(_VENV) and sys.executable != _VENV:
        sys.exit(subprocess.call([_VENV, *sys.argv]))
    raise

import argparse
import glob
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from qwop_env import QWOPEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
LOG_DIR = os.path.join(HERE, "logs")
METRICS_PATH = os.path.join(HERE, "metrics.json")
CSV_PATH = os.path.join(HERE, "episodes.csv")


def make_env(rank: int):
    def _init():
        env = QWOPEnv(headless=True, frames_per_step=3, max_steps=2000)
        return Monitor(env)
    return _init


class BestMetresCallback(BaseCallback):
    """记录并打印最远距离。"""

    def __init__(self):
        super().__init__()
        self.best = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            m = info.get("metres")
            if m is not None and m > self.best:
                announce = m - getattr(self, "_last_print", 0.0) >= 1.0
                self.best = m
                if announce and self.best > 1:
                    self._last_print = self.best
                    print(f"[milestone] 新纪录: {self.best:.1f} 米 "
                          f"(step={self.num_timesteps})")
            if info.get("success"):
                print(f"[SUCCESS] 到达终点! step={self.num_timesteps}")
        self.logger.record("qwop/best_metres", self.best)
        return True


class MetricsCallback(BaseCallback):
    """周期性把训练指标写入 metrics.json，供实时仪表盘读取。"""

    def __init__(self, total_timesteps: int, write_every: float = 10.0):
        super().__init__()
        self.total = total_timesteps
        self.write_every = write_every
        self._last_write = 0.0
        self._t0 = time.time()
        self._step0 = 0
        self.episodes = []          # 最近若干回合: {metres,reward,length,crossed,success,standing}
        self.action_counts = {}     # 动作分布计数
        self.history = []           # {step, best, reward_mean}
        self.best = 0.0
        self._cap = 200
        self._csv_written = False

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        for info in infos:
            a = info.get("action")
            if a is not None:
                self.action_counts[str(int(a))] = self.action_counts.get(str(int(a)), 0) + 1
            if "episode" in info:  # 回合结束（来自 Monitor 包装）
                ep = {
                    "metres": float(info.get("metres", 0.0)),
                    "reward": float(info["episode"]["r"]),
                    "length": int(info["episode"]["l"]),
                    "crossed": bool(info.get("crossed_hurdle", False)),
                    "success": bool(info.get("success", False)),
                    "standing": bool(info.get("standing_at_hurdle", False)),
                }
                self.episodes.append(ep)
                if len(self.episodes) > self._cap:
                    self.episodes.pop(0)
                if ep["metres"] > self.best:
                    self.best = ep["metres"]
                # 追加写入 episodes.csv（供训练后分析）
                try:
                    import csv
                    new_file = not self._csv_written
                    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        if new_file:
                            w.writerow(["step", "metres", "reward", "length",
                                        "crossed", "success", "standing"])
                            self._csv_written = True
                        w.writerow([self.num_timesteps, ep["metres"], ep["reward"],
                                    ep["length"], int(ep["crossed"]),
                                    int(ep["success"]), int(ep["standing"])])
                except Exception:
                    pass

        now = time.time()
        if now - self._last_write >= self.write_every:
            self._last_write = now
            fps = 0.0
            if now - self._t0 > 0:
                fps = (self.num_timesteps - self._step0) / (now - self._t0)
            recent = self.episodes[-60:]
            n = max(len(recent), 1)
            reward_mean = sum(e["reward"] for e in recent) / n
            cross_rate = sum(1 for e in recent if e["crossed"]) / n
            succ_rate = sum(1 for e in recent if e["success"]) / n
            if recent:
                self.history.append(
                    {"step": self.num_timesteps, "best": self.best,
                     "reward_mean": reward_mean})
                if len(self.history) > 400:
                    self.history.pop(0)
            payload = {
                "step": self.num_timesteps,
                "total_steps": self.total,
                "best_metres": round(self.best, 2),
                "episodes": len(self.episodes),
                "fps": round(fps, 1),
                "reward_mean": round(reward_mean, 2),
                "hurdle_cross_rate": round(cross_rate, 3),
                "success_rate": round(succ_rate, 3),
                "recent": [
                    {"m": round(e["metres"], 1), "r": round(e["reward"], 1),
                     "crossed": e["crossed"], "success": e["success"]}
                    for e in recent[-60:]
                ],
                "action_dist": self.action_counts,
                "history": self.history,
            }
            try:
                with open(METRICS_PATH, "w", encoding="utf-8") as f:
                    json.dump(payload, f)
            except Exception:
                pass
        return True


def latest_checkpoint() -> str | None:
    files = glob.glob(os.path.join(MODEL_DIR, "ppo_qwop_*_steps.zip"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    venv = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    vn_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")
    if args.resume and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ckpt = latest_checkpoint() if args.resume else None
    if ckpt:
        print(f"从 checkpoint 续训: {ckpt}")
        model = PPO.load(ckpt, env=venv, tensorboard_log=LOG_DIR)
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            policy_kwargs=dict(net_arch=[256, 256]),
            n_steps=2048,
            batch_size=512,
            learning_rate=3e-4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=LOG_DIR,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_qwop",
        ),
        BestMetresCallback(),
        MetricsCallback(total_timesteps=args.timesteps, write_every=10.0),
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
    finally:
        model.save(os.path.join(MODEL_DIR, "ppo_qwop_final"))
        venv.save(vn_path)
        venv.close()
        print("模型已保存到 models/ppo_qwop_final.zip")


if __name__ == "__main__":
    main()
