"""用已有 checkpoint 在物理环境里随机 rollout, 重建正确的 vecnormalize.pkl。

问题: 之前的训练被 kill 时未执行 finally 里的 venv.save(), 导致 vecnormalize.pkl
是陈旧后端(浏览器)留下的错误观测归一化统计, 让物理模型评估/续训全部失效。
这里用模型自身 rollout 收集物理 obs 统计, 使 pkl 与该模型训练时的分布一致。
"""
import os, glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from qwop_phys import QWOPPhysEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
VN_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")

ckpt = max(glob.glob(os.path.join(MODEL_DIR, "ppo_qwop_*_steps.zip")), key=os.path.getmtime)
print("用 checkpoint 重建归一化统计:", ckpt)
model = PPO.load(ckpt)

env = DummyVecEnv([lambda: QWOPPhysEnv(frames_per_step=3, max_steps=2000)])
venv = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
venv.training = True  # 关键: 让 obs_rms 持续累积统计

obs = venv.reset()
WARMUP = 150_000
for i in range(WARMUP):
    action, _ = model.predict(obs, deterministic=False)
    obs, r, done, info = venv.step(action)

venv.training = False
venv.norm_reward = False  # 评估时不需要 reward 归一化
venv.save(VN_PATH)
print(f"已保存正确 vecnormalize.pkl (obs_rms count={int(venv.obs_rms.count)}, "
      f"mean[:3]={np.round(venv.obs_rms.mean[:3],3)})")
