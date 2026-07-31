"""诊断: 加载最新 checkpoint + vecnormalize.pkl, 检查归一化统计是否合理,
并分别用随机/确定性策略评估, 判断是否 pkl 陈旧导致评估失效。"""
import os, glob, pickle
import numpy as np
from stable_baselines3 import PPO
from qwop_phys import QWOPPhysEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
VN = os.path.join(MODEL_DIR, "vecnormalize.pkl")
ckpt = max(glob.glob(os.path.join(MODEL_DIR, "ppo_qwop_*_steps.zip")), key=os.path.getmtime)
print("ckpt:", ckpt)
with open(VN, "rb") as f:
    vn = pickle.load(f)
rms = vn.obs_rms
print("obs_rms count =", rms.count, "  shape =", rms.mean.shape)
print("mean[:6] =", np.round(rms.mean[:6], 3))
print("var[:6]  =", np.round(rms.var[:6], 3))

model = PPO.load(ckpt)
env = QWOPPhysEnv(frames_per_step=3, max_steps=2000)


def norm(o):
    o = np.asarray(o, dtype=np.float32)
    return np.clip((o - rms.mean) / (np.sqrt(rms.var) + 1e-8), -10.0, 10.0).reshape(1, -1)


for det in (False, True):
    print(f"\n=== deterministic={det} ===")
    for ep in range(3):
        o, _ = env.reset()
        on = norm(o)
        done = False
        steps = 0
        while not done:
            a, _ = model.predict(on, deterministic=det)
            o2, r, term, trunc, info = env.step(int(a[0]))
            on = norm(o2)
            done = term or trunc
            steps += 1
        print(f"  ep{ep}: metres={info['metres']:.2f} steps={steps} time={steps*0.1:.1f}s success={info['success']}")
