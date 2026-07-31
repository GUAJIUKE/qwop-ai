"""确定性评估当前最佳模型的最快耗时 (速度基准)。

自动选最新 checkpoint + vecnormalize.pkl 的观测归一化,
用 deterministic 策略跑 N 局, 打印每局米数/步数/耗时/是否通关,
并汇总最快与平均耗时。

用法:
    python eval_speed.py                # 自动选最新 checkpoint
    python eval_speed.py 路径.zip       # 指定模型
"""
import os, sys, glob, pickle
import numpy as np
from stable_baselines3 import PPO
from qwop_phys import QWOPPhysEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
VN_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")
if not os.path.exists(VN_PATH):
    _alt = os.path.join(MODEL_DIR, "vecnormalize_phys_sanity.pkl")
    if os.path.exists(_alt):
        VN_PATH = _alt
FRAMES_PER_STEP = 3
DT = 1.0 / 30.0
SIM_PER_STEP = FRAMES_PER_STEP * DT  # 每环境步对应的仿真秒数


def latest_ckpt(model_dir):
    # 优先 final.zip，否则取最新的 *_steps.zip
    f = os.path.join(model_dir, "ppo_qwop_final.zip")
    if os.path.exists(f):
        return f
    fs = glob.glob(os.path.join(model_dir, "ppo_qwop_*_steps.zip"))
    return max(fs, key=os.path.getmtime) if fs else None


def main():
    # 解析可选参数: [模型.zip] [N] [--modeldir DIR]
    model_arg = None
    n_arg = 10
    modeldir = MODEL_DIR
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--modeldir":
            modeldir = sys.argv[i + 1]
            i += 2
        elif model_arg is None and not a.startswith("--"):
            # 第一个非 flags 参数: 模型路径
            model_arg = a
            i += 1
        elif a.isdigit():
            n_arg = int(a)
            i += 1
        else:
            i += 1
    vn_path = os.path.join(modeldir, "vecnormalize.pkl")
    if not os.path.exists(vn_path):
        _alt = os.path.join(modeldir, "vecnormalize_phys_sanity.pkl")
        if os.path.exists(_alt):
            vn_path = _alt

    ckpt = model_arg if model_arg else latest_ckpt(modeldir)
    assert ckpt, "找不到 checkpoint"
    print("模型:", ckpt)
    print("归一化:", vn_path)
    with open(vn_path, "rb") as f:
        vn = pickle.load(f)
    rms = vn.obs_rms

    def norm(o):
        o = np.asarray(o, dtype=np.float32)
        o = (o - rms.mean) / (np.sqrt(rms.var) + 1e-8)
        return np.clip(o, -10.0, 10.0)

    model = PPO.load(ckpt)
    env = QWOPPhysEnv(frames_per_step=FRAMES_PER_STEP, max_steps=2000)
    N = n_arg

    times, metres = [], []
    succ = 0
    for ep in range(N):
        o, _ = env.reset()
        on = norm(o).reshape(1, -1)
        done = False
        steps = 0
        while not done:
            a, _ = model.predict(on, deterministic=True)
            o2, r, term, trunc, info = env.step(int(a[0]))
            on = norm(o2).reshape(1, -1)
            done = term or trunc
            steps += 1
        m = info["metres"]
        s = info["success"]
        t = steps * SIM_PER_STEP
        metres.append(m)
        if s:
            succ += 1
            times.append(t)
        print(f"  ep{ep:2d}: metres={m:7.2f}  steps={steps:5d}  time={t:7.2f}s  success={s}")

    best = min(times) if times else 0.0
    avg = float(np.mean(times)) if times else 0.0
    best_m = max(metres) if metres else 0.0
    if succ > 0:
        print(f"\n汇总: 通关 {succ}/{N}  最快耗时={best:.2f}s ({100.0/best:.2f} m/s)  平均耗时={avg:.2f}s  平均米数={float(np.mean(metres)):.2f}")
    else:
        print(f"\n汇总: 通关 0/{N} (全失败)  最佳距离={best_m:.2f}m  —— 确定性策略未收敛, 需更久训练或调参")


if __name__ == "__main__":
    main()
