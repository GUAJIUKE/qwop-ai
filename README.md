# QWOP-AI · 用 PPO 教会小人跨栏并跳远

一个用 **PPO（Proximal Policy Optimization）** 强化学习训练 **QWOP** 的端到端项目。
通过浏览器内桥接脚本直接读取游戏物理状态、合成按键、同步快进，让智能体在
**带跨栏 + 沙坑跳远** 的 QWOP 变体里尽快跑到终点。

> 🎮 这是 QWOP 的**跨栏 + 沙坑跳远**变体：先跑过 x≈500（约 50 米）处的栏架，
> 再在沙坑完成跳远，落地即通关（"X metres in Y seconds"）；中途摔倒则失败。

---

## ✨ 特性

- **状态直读，不靠像素**：直接读取 Box2D 的 12 个刚体状态（位置/角度/速度），用小型 MLP 即可训练，无需 CNN。
- **同步快进**：劫持 `requestAnimationFrame` + 虚拟时钟，单环境可达 ~20× 实时速度，多环境并行更快。
- **可调奖励**：`reward_config.json` 实时改权重（前进 / 摔倒 / 超时 / 跨栏站立 / 跳远距离），训练中保存即生效。
- **4 套可视化**：实时仪表盘、观战 + HUD 数据叠加、训练后分析图、姿态骨架。

---

## 🧰 环境要求

- Python 3.10+
- [Playwright](https://playwright.dev/) 浏览器内核（Chromium / Edge / Chrome 任一）
- 一个现代浏览器（观战 / 仪表盘用）

---

## 🚀 快速开始

```bash
# 1. 克隆
git clone https://github.com/GUAJIUKE/qwop-ai.git
cd qwop-ai

# 2. 创建虚拟环境并安装依赖
python -m venv venv
venv\Scripts\activate        # Windows
#   source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 3. 安装 Playwright 浏览器
playwright install chromium

# 4. 拉取并打补丁游戏运行时 (自动从官方源下载, 见下方版权说明)
python fetch_game.py

# 5. 开始训练
python train.py --n-envs 8 --timesteps 3000000

# 6. 随时观战 (另开终端)
python watch.py --mode pose          # 叠加人形骨架看跑姿
python dashboard.py                  # 启动实时仪表盘 http://127.0.0.1:8765
```

> 💡 脚本内置"依赖自检"：若当前 Python 缺少依赖，会自动用虚拟环境重跑，无需手动切换解释器。

---

## 🏃 游戏机制

| 要素 | 说明 |
|---|---|
| 终点 | 100 米（躯干世界坐标 x / 10） |
| 跨栏 | 栏架位于 `x ≈ 500`（约 **50 米**），需站立通过，否则只能爬行 |
| 跳远 | 沙坑处 `jumpLanded` 标志置位即完成跳远，落地距离计入成绩 |
| 失败 | `fallenEnding`（摔倒）；成功为 `jumpEnding`（"X metres in Y seconds"） |

---

## 🎯 奖励设计（`reward_config.json` 可调）

| 组件 | 公式 | 默认权重 |
|---|---|---|
| 前进奖励 | `Δmetres × forward_per_metre` | `1.0` |
| 摔倒惩罚 | 摔倒时 `−fall_penalty` | `5.0` |
| 超时惩罚 | 达 `max_steps` 未完成 `−timeout_penalty` | `5.0` |
| 跨栏站立加成 | 过栏且躯干近乎直立 `＋hurdle_stand_bonus`，否则爬行 `＋hurdle_crawl_bonus` | `8.0 / 1.0` |
| 跳远距离加成 | `jumpLanded` 时 `＋jump_distance_bonus × 最终米数` | `1.0` |
| 通关奖励 | 跳远落地额外 `＋success_terminal_bonus` | `50.0` |

训练中直接编辑 `reward_config.json` 保存即可生效（环境按文件修改时间自动重载），
或在仪表盘底部拖滑块实时调整。

---

## 📁 仓库结构

```
qwop-ai/
├── qwop_env.py        # Gymnasium 环境 (Playwright 驱动浏览器)
├── train.py           # PPO 训练 (并行环境 / 断点续训 / TensorBoard)
├── watch.py           # 观战 + HUD/骨架叠加
├── analyze.py         # 训练后分析图 (matplotlib)
├── dashboard.py       # 实时训练仪表盘 (本地 HTTP 服务)
├── dashboard.html     # 仪表盘前端
├── bridge.js          # 注入游戏的桥接脚本 (读状态/按键/快进/重置)
├── reward_config.json # 可调奖励权重
├── fetch_game.py      # 拉取并打补丁游戏运行时
├── game/
│   ├── qwop.html      # 游戏页 (本仓库代码)
│   └── bridge.js      # 同名桥接 (本仓库代码)
└── requirements.txt
```

> `game/QWOP.min.js`、`game/assets/`、`game/lib/howler.js` 由 `fetch_game.py`
> 生成，**不入库**（见版权说明）。

---

## 📊 可视化

| 可视化 | 启动方式 | 说明 |
|---|---|---|
| 实时仪表盘 | `python dashboard.py` | 学习曲线 / 奖励 / 距离柱状图 / 动作分布 + 实时调奖励滑块 |
| 观战 + HUD | `python watch.py` | 游戏画面叠加距离/动作/跨栏/跳远 HUD |
| 观战 + 骨架 | `python watch.py --mode pose` | 额外叠加 12 刚体人形骨架看跑姿 |
| 训练后分析图 | `python analyze.py` | 输出 `analysis/qwop_analysis.png` 等 6 子图 |

---

## ⚠️ 版权与免责声明

- **QWOP 游戏本体（包括 `QWOP.min.js`、`assetbundle.parcel` 等）版权归 Bennett Foddy 所有**。
  本仓库**不打包、不重新分发**任何游戏二进制文件。
- 请通过 `python fetch_game.py` 从官方源 `https://www.foddy.net/legacy/` 获取原版文件；
  脚本仅应用 4 处**最小补丁**以使自动化训练可行（暴露实例 / 禁用引擎自毁 / 跳帧渲染），
  不改变游戏玩法或美术资源。
- 本项目仅用于**学习与研究强化学习**，请遵守游戏作者的使用条款。

---

## 📚 参考

- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [Gymnasium](https://gymnasium.farama.org/)
- [Playwright for Python](https://playwright.dev/python/)
- 原版游戏：<https://www.foddy.net/Athletics.html>
