"""PPO 可学习性 sanity check：在物理后端上训练，确认 forward metres 能上升。"""
from __future__ import annotations
import sys, os, time
_VENV = r"C:\Users\24479\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
try:
    import stable_baselines3  # noqa
except ImportError:
    if os.path.exists(_VENV) and sys.executable != _VENV:
        sys.exit(__import__("subprocess").call([_VENV, *sys.argv]))
    raise

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from qwop_phys import QWOPPhysEnv

HERE = os.path.dirname(os.path.abspath(__file__))

class BestMetres(BaseCallback):
    def __init__(self):
        super().__init__(); self.best = 0.0
    def _on_step(self):
        for info in self.locals.get("infos", []):
            m = info.get("metres")
            if m is not None and m > self.best:
                self.best = m
                if self.best > 1:
                    print(f"  [里程碑] {self.best:.1f}m @ step {self.num_timesteps}")
        return True

def make_env(rank):
    def _init():
        return QWOPPhysEnv(frames_per_step=3, max_steps=2000)
    return _init

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=150_000)
    args = ap.parse_args()
    t0 = time.time()
    venv = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model = PPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=[256, 256]),
                n_steps=2048, batch_size=512, learning_rate=3e-4, gamma=0.995,
                gae_lambda=0.95, ent_coef=0.01, clip_range=0.2, verbose=0)
    model.learn(total_timesteps=args.timesteps, callback=BestMetres())
    venv.save(os.path.join(HERE, "models", "vecnormalize_phys_sanity.pkl"))
    model.save(os.path.join(HERE, "models", "ppo_phys_sanity"))
    print(f"训练完成: {args.timesteps} 步, 用时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
