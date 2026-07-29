"""训练后分析: 读取 episodes.csv 与 metrics.json, 生成 PNG 曲线。

用法:
    python analyze.py                # 读取最新数据, 保存到 analysis/
    python analyze.py --out myplot  # 指定输出目录
"""
from __future__ import annotations

# ---- 依赖自检: 若当前 Python 未安装 matplotlib, 自动用隔离 venv 重跑 ----
import sys, os, subprocess
_VENV = r"C:\Users\24479\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
try:
    import matplotlib  # noqa: F401
except ImportError:
    if os.path.exists(_VENV) and sys.executable != _VENV:
        sys.exit(subprocess.call([_VENV, *sys.argv]))
    raise

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "episodes.csv")
METRICS_PATH = os.path.join(HERE, "metrics.json")

# 注册一个系统中文字体（解决 CJK 缺字方框）
for _cand in (r"C:\Windows\Fonts\simhei.ttf",
              r"C:\Windows\Fonts\simsun.ttc",
              r"C:\Windows\Fonts\msyh.ttc"):
    if os.path.exists(_cand):
        try:
            fm.fontManager.addfont(_cand)
        except Exception:
            pass

# 暗色风格
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#181b24",
    "axes.edgecolor": "#2a2f3c",
    "axes.labelcolor": "#e6e8ee",
    "text.color": "#e6e8ee",
    "xtick.color": "#8b93a7",
    "ytick.color": "#8b93a7",
    "grid.color": "#2a2f3c",
    "font.size": 11,
    "font.family": "SimHei",
    "axes.unicode_minus": False,
})


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("0", "1") or k in
                             ("metres", "reward", "length", "step") else int(v))
                         for k, v in r.items()})
    return rows


def rolling(vals, w=20):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        out.append(sum(vals[lo:i + 1]) / (i - lo + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = load_csv(CSV_PATH) if os.path.exists(CSV_PATH) else []
    if not rows:
        raise SystemExit(f"没有找到 {CSV_PATH}，请先训练（python train.py）")

    n = len(rows)
    idx = list(range(1, n + 1))
    metres = [r["metres"] for r in rows]
    rewards = [r["reward"] for r in rows]
    lengths = [r["length"] for r in rows]
    successes = [r["success"] for r in rows]
    crossed = [r["crossed"] for r in rows]

    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    fig.suptitle(f"QWOP 训练分析  ·  {n} 回合  ·  最远 {max(metres):.1f} 米",
                 fontsize=14, fontweight="bold")

    # 1) 米数 vs 回合
    ax = axs[0, 0]
    ax.plot(idx, metres, color="#4f8cff", alpha=0.35, lw=1)
    ax.plot(idx, rolling(metres), color="#39d98a", lw=2.2, label="滚动均值(20)")
    ax.axhline(100, color="#ff5d6c", ls="--", lw=1, label="目标 100m")
    ax.set_title("距离 vs 回合"); ax.set_xlabel("回合"); ax.set_ylabel("米")
    ax.legend(fontsize=9); ax.grid(True)

    # 2) 奖励 vs 回合
    ax = axs[0, 1]
    ax.plot(idx, rewards, color="#ffb454", alpha=0.35, lw=1)
    ax.plot(idx, rolling(rewards), color="#ffb454", lw=2.2, label="滚动均值(20)")
    ax.set_title("回合奖励 vs 回合"); ax.set_xlabel("回合"); ax.set_ylabel("reward")
    ax.legend(fontsize=9); ax.grid(True)

    # 3) 回合长度
    ax = axs[0, 2]
    ax.plot(idx, lengths, color="#b07bff", alpha=0.5, lw=1)
    ax.plot(idx, rolling(lengths), color="#b07bff", lw=2.2, label="滚动均值(20)")
    ax.set_title("回合长度 vs 回合"); ax.set_xlabel("回合"); ax.set_ylabel("步数")
    ax.legend(fontsize=9); ax.grid(True)

    # 4) 跨栏率 / 通关率 (窗口 30)
    ax = axs[1, 0]
    w = min(30, n)
    cr = rolling(crossed, w); sr = rolling(successes, w)
    ax.plot(idx, cr, color="#39d98a", lw=2, label=f"跨栏率(窗口{w})")
    ax.plot(idx, sr, color="#ff5d6c", lw=2, label=f"通关率(窗口{w})")
    ax.set_ylim(-0.05, 1.05); ax.set_title("跨栏率 / 通关率"); ax.set_xlabel("回合"); ax.set_ylabel("比例")
    ax.legend(fontsize=9); ax.grid(True)

    # 5) 通关回合距离分布
    ax = axs[1, 1]
    succ_m = [r["metres"] for r in rows if r["success"]]
    if succ_m:
        ax.hist(succ_m, bins=min(20, max(5, len(succ_m) // 2)),
                color="#39d98a", edgecolor="#0f1117")
        ax.set_title(f"通关回合距离分布 (n={len(succ_m)})")
    else:
        ax.text(0.5, 0.5, "尚无通关回合", ha="center", va="center", color="#8b93a7")
        ax.set_title("通关回合距离分布")
    ax.set_xlabel("米"); ax.set_ylabel("次数"); ax.grid(True)

    # 6) 动作分布
    ax = axs[1, 2]
    if os.path.exists(METRICS_PATH):
        m = json.load(open(METRICS_PATH, encoding="utf-8"))
        ad = m.get("action_dist", {})
        if ad:
            keys = sorted(int(k) for k in ad)
            vals = [ad[str(k)] for k in keys]
            ax.bar([f"a{k}" for k in keys], vals, color="#4f8cff")
            ax.set_title("动作分布 (最近快照)"); ax.set_xlabel("动作"); ax.set_ylabel("次数")
            ax.grid(True, axis="y")
        else:
            ax.set_title("动作分布 (无数据)")
    else:
        ax.set_title("动作分布 (无 metrics)")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.out, "qwop_analysis.png")
    fig.savefig(out, dpi=130)
    print(f"已保存: {out}")

    # 单独导出最远距离学习曲线（来自 metrics 历史）
    if os.path.exists(METRICS_PATH):
        m = json.load(open(METRICS_PATH, encoding="utf-8"))
        hist = m.get("history", [])
        if len(hist) > 1:
            fig2, ax = plt.subplots(figsize=(11, 5))
            xs = [h["step"] for h in hist]; ys = [h["best"] for h in hist]
            ax.plot(xs, ys, color="#39d98a", lw=2.5)
            ax.set_title("最远纪录 vs 训练步数"); ax.set_xlabel("步数"); ax.set_ylabel("米")
            ax.grid(True)
            out2 = os.path.join(args.out, "learning_curve.png")
            fig2.savefig(out2, dpi=130)
            print(f"已保存: {out2}")


if __name__ == "__main__":
    main()
