"""深度分析训练不稳定性"""
import json
import csv

# 1. 当前奖励配置
rc = json.load(open("reward_config.json"))
print("=== 当前奖励配置 ===")
for k, v in rc.items():
    print(f"  {k}: {v}")

# 2. 动作映射
print("\n=== 动作映射 ===")
ACTIONS = [
    (0, 0, 0, 0),  # 0: NONE
    (1, 0, 0, 0),  # 1: Q
    (0, 1, 0, 0),  # 2: W
    (0, 0, 1, 0),  # 3: O
    (0, 0, 0, 1),  # 4: P
    (1, 1, 0, 0),  # 5: QW
    (1, 0, 0, 1),  # 6: QP
    (0, 1, 1, 0),  # 7: WO
    (0, 1, 0, 1),  # 8: WP
]
names = ["NONE", "Q", "W", "O", "P", "QW", "QP", "WO", "WP"]
for i, (q, w, o, p) in enumerate(ACTIONS):
    keys = []
    if q:
        keys.append("Q")
    if w:
        keys.append("W")
    if o:
        keys.append("O")
    if p:
        keys.append("P")
    print(f"  {i}: {names[i]:4s} -> {' '.join(keys) or '(空)'}")

# 3. 全量 CSV 分析
print("\n=== 全量 episodes.csv 分析（按训练阶段）===")
rows = []
with open("episodes.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        try:
            m = float(r.get("metres", 0))
            rows.append({
                "metres": m,
                "reward": float(r.get("reward", 0)),
                "length": int(r.get("length", 0)),
                "crossed": r.get("crossed", "False") == "True",
                "success": r.get("success", "False") == "True",
            })
        except (ValueError, TypeError):
            pass

print(f"有效数据: {len(rows)} 回合")

if len(rows) >= 20:
    chunk_size = max(len(rows) // 6, 10)
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]
    for i, chunk in enumerate(chunks):
        ms = [r["metres"] for r in chunk]
        bad = sum(1 for m in ms if m <= 2)
        good = sum(1 for m in ms if m >= 100)
        avg_r = sum(r["reward"] for r in chunk) / len(chunk)
        avg_l = sum(r["length"] for r in chunk) / len(chunk)
        print(
            f"  阶段{i+1}({len(chunk)}eps): "
            f"avg={sum(ms)/len(ms):.1f}m  "
            f"bad(<=2m)={bad}({bad/len(chunk)*100:.0f}%)  "
            f"good(>=100m)={good}({good/len(chunk)*100:.0f}%)  "
            f"avg_r={avg_r:.0f}  avg_len={avg_l:.0f}s"
        )

# 4. 失败模式分析
print(f"\n=== 失败模式: <=2m 回合的 step 长度 ===")
failures = [r for r in rows if r["metres"] <= 2]
if failures:
    lengths = [r["length"] for r in failures]
    print(f"共 {len(failures)} 个失败回合 / 总 {len(rows)} ({len(failures)/len(rows)*100:.1f}%)")
    print(f"  平均长度: {sum(lengths)/len(lengths):.0f} steps")
    sorted_len = sorted(lengths)
    print(f"  中位长度: {sorted_len[len(sorted_len)//2]:.0f} steps")
    print(f"  最短: {min(lengths)}  最长: {max(lengths)}")
    instant = sum(1 for l in lengths if l <= 30)
    print(f"  立刻摔(<=30steps): {instant} ({instant/len(failures)*100:.0f}%)")
else:
    print("无失败回合")

# 5. 成功 vs 失败 reward 对比
print(f"\n=== 成功 vs 失败 reward 对比 ===")
succ = [r for r in rows if r["success"]]
fail = [r for r in rows if not r["success"] and r["metres"] < 50]
if succ:
    sr = [r["reward"] for r in succ]
    print(f"成功(>=100m): n={len(succ)}  avg_reward={sum(sr)/len(sr):.0f}")
if fail:
    fr = [r["reward"] for r in fail]
    print(f"失败(<50m):   n={len(fail)}  avg_reward={sum(fr)/len(fr):.0f}")

# 6. PPO 超参（从 train.py 推断）
print(f"\n=== PPO 超参与稳定性因素 ===")
print("  默认 ent_coef=0.01 -> 训练后期仍有 ~1% 随机探索动作")
print("  默认 lr=3e-4     -> 后期学习率未衰减，策略仍在震荡")
print("  离散9动作       -> NONE(action0)被选中时角色立刻失去平衡摔倒")
print("  奖励: forward=1.0 fall=-5 timeout=-5 -> 摔倒惩罚不够'痛'")
