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
- **默认物理后端**：`box2d`（纯 Python Box2D，`pip install box2d`），无需浏览器，速度极快
- **浏览器后端（可选）**：[Playwright](https://playwright.dev/) 浏览器内核（Chromium / Edge / Chrome 任一）
- 一个现代浏览器（浏览器后端观战 / 仪表盘用）

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

## 🧱 两种训练后端（速度关键）

训练瓶颈**不在神经网络，而在环境仿真**：原版环境用 Playwright 驱动浏览器跑 Box2D 物理，
每次 `step()` 都要一次跨进程 `page.evaluate` 往返，单环境约 **20× 实时**。智能体的 MLP
（`[256,256]`）算力相对可忽略。

本项目提供两种后端，由 `train.py --backend` 切换：

| 后端 | 原理 | 速度（单环境） | 保真度 |
|---|---|---|---|
| `browser`（原版） | Playwright 驱动真实 QWOP 游戏 | ~20× 实时 | 100% 原版游戏 |
| **`phys`（默认）** | 纯 Python Box2D 复刻 12 刚体 ragdoll，进程内运行 | **~数百× 实时** | QWOP 风格复刻（标准做法） |

`phys` 后端去掉了浏览器 + Playwright 的全部开销，单环境可达 **约 870× 实时**（实测约 2.6 万物理帧/秒），
比原版快 **40 倍以上**，且 obs(88)/动作(9)/奖励/跨栏/沙坑接口与原版**完全一致**，所以
`train.py` / `watch.py` / `analyze.py` / 仪表盘都无需改动即可切换。

> ⚠️ **保真度说明**：`phys` 后端是用 Box2D 从头复刻的 QWOP 风格双足 ragdoll（相同的 12 刚体、
> Q/W/O/P 肌肉控制、跨栏与沙坑机制），而非官方游戏二进制。这是 QWOP-AI 研究的**通用做法**
> （原版游戏代码混淆、强耦合浏览器，难以高速批量仿真）。若需与原版 1:1 对照，用 `--backend browser`。
> 为让底座可学，`phys` 后端对躯干施加了"核心稳定"PD 力矩（模拟核心张力），因此平衡难度低于原版，
> 但前进/跨栏/跳远仍是需学习的操控任务。

```bash
# 默认物理后端（极快，推荐）
python train.py --n-envs 8 --timesteps 3000000

# 对照：原版浏览器后端（忠实但慢）
python train.py --backend browser --n-envs 8 --timesteps 3000000
```

观战同样支持两种后端：

```bash
python watch.py --backend phys   --mode pose   # matplotlib 实时骨架（无需浏览器）
python watch.py --backend browser --mode pose  # 浏览器画面 + HUD/骨架叠加
```

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
├── qwop_env.py        # Gymnasium 环境 (Playwright 驱动浏览器, --backend browser)
├── qwop_phys.py       # Gymnasium 环境 (纯 Python Box2D 复刻, 默认 --backend phys, 极快)
├── train.py           # PPO 训练 (两种后端 / 并行环境 / 断点续训 / TensorBoard)
├── train_phys_sanity.py  # 物理后端可学习性验证脚本
├── watch.py           # 观战 + HUD/骨架 (两种后端)
├── analyze.py         # 训练后分析图 (matplotlib)
├── dashboard.py       # 实时训练仪表盘 (本地 HTTP 服务)
├── dashboard.html     # 仪表盘前端
├── bridge.js          # 注入游戏的桥接脚本 (读状态/按键/快进/重置)
├── reward_config.json # 可调奖励权重
├── fetch_game.py      # 拉取并打补丁游戏运行时
├── game/
│   ├── qwop.html      # 游戏页 (本仓库代码)
│   └── bridge.js      # 同名桥接 (本仓库代码)
├── docs/              # 文档与截图
│   └── dashboard_100m_screenshot.png
├── analysis/          # 训练后自动生成的图表
│   ├── qwop_analysis.png
│   └── learning_curve.png
└── requirements.txt
```

> `game/QWOP.min.js`、`game/assets/`、`game/lib/howler.js` 由 `fetch_game.py`
> 生成，**不入库**（见版权说明）。
> `models/`（训练好的模型 checkpoint）也**不入库**（体积大且可复现训练）。

---

## 📊 可视化

| 可视化 | 启动方式 | 说明 |
|---|---|---|
| 实时仪表盘 | `python dashboard.py` | 学习曲线 / 奖励 / 距离柱状图 / 动作分布 + 实时调奖励滑块 |
| 观战 + HUD | `python watch.py` | 游戏画面叠加距离/动作/跨栏/跳远 HUD |
| 观战 + 骨架 | `python watch.py --mode pose` | 额外叠加 12 刚体人形骨架看跑姿 |
| 训练后分析图 | `python analyze.py` | 输出 `analysis/qwop_analysis.png` 等 6 子图 |

---

## 🏆 训练成果

### 首次训练 · 2026-07-29（21 万步通关）

> **21 万步即通关 100 米**，PPO 智能体从零开始学会了跨栏 + 沙坑跳远完整流程。

| 指标 | 数值 |
|---|---|
| **训练步数** | ~215,000 / 3,000,000（仅用 7%） |
| **最远距离** | **100.97 米** ✅ 超过终点线 |
| **跨栏成功率** | **91.7%** |
| **通关率** | **18.3%** |
| **平均奖励** | 98.57 |

### 最终训练 · 2026-07-30（30.8 万步停止）

> 训练在 308k 步时手动停止进行分析。智能体已稳定通关，但存在少量 0m 回合。

| 指标 | 数值 |
|---|---|
| **训练步数** | **308,400 / 3,000,000（10.3%）** |
| **最远距离** | **103.54 米** ✅ 超终点 3.5m |
| **跨栏成功率** | **96.7%** |
| **通关率** | **88.3%**（最近 200 回合） |
| **平均奖励** | 224.83 |
| **训练 FPS** | ~87（8 并行） |
| **总回合数** | 777 |

### 最终仪表盘截图（103.54m 时刻）

![最终仪表盘](docs/dashboard_294k_final.png)

### 不稳定性分析（重要！）

> **为什么训练后期还会出现 0~1 米的成绩？** 完整分析见 [docs/INSTABILITY_ANALYSIS.md](docs/INSTABILITY_ANALYSIS.md)

**一句话结论：不是退步，是 PPO 探索机制的副作用。**

| 因素 | 影响 | 修复难度 |
|---|---|---|
| Action 0（空操作）被随机探索采样到 → 立即摔倒 | 占后期失败的 **80%+** | 改 1 行代码 |
| 熵系数 ent_coef=0.01 未衰减 → 后期仍有 ~1% 随机动作 | 占 **15%** | 加 1 个参数 |
| 学习率未衰减 → 策略震荡无法收敛到保守策略 | 占 **5%** | 加 1 个参数 |

详细数据、分段分析、修复方案和预期效果见完整报告。

### 学习曲线

![学习曲线](analysis/learning_curve.png)

距离从随机乱跑的 ~0m 稳步增长到 100m+，在约 15 万步处出现**快速跃升**（学会跨栏后的爆发式进步），之后稳定在 99~101 米区间。

### 综合分析

![综合分析](analysis/qwop_analysis.png)

6 个子图展示：
1. **距离趋势**：滚动均值从 0→100m，目标线 100m 已突破
2. **奖励曲线**：从 -14 飙升到 ~250，信号健康无震荡
3. **回合长度**：后期稳定在 1700~1900 步/回合（接近 max_steps=2000 上限）
4. **跨栏率/通关率**：跨栏率接近 100%，通关率持续爬升
5. **通关距离分布**：集中在 99~101m（13 个已通关回合）
6. **动作分布**：9 个动作均有使用，a2/a7/a8（前进组合键）使用最多

### 实时仪表盘截图（100.97m 时刻）

![仪表盘](docs/dashboard_100m_screenshot.png)

仪表盘实时显示：训练步数、最远纪录、跨栏成功率、FPS、学习曲线、最近 60 回合距离柱状图、动作分布热力图，以及底部 8 个可拖动的奖励调节滑块。

### 关键发现

- **收敛极快**：仅用计划步数的 ~7% 就达到 100m，10% 达到 103.54m，说明状态直读 + 组合键动作空间的设计非常高效
- **跨栏是关键分水岭**：约 12 万步前卡在 20~30m（"跪地蹭"局部最优），一旦学会过栏就迅速冲到 80m+
- **奖励设计有效**：跨栏站立加成(+8) vs 爬行加成(+1) 的差异化让智能体优先学站立过栏
- **后期 0m 回合 = 探索噪声**：777 回合中 98% 的失败集中在早期（阶段 1-4），后期实际失败率仅 7%。根因是 Action 0（空操作）被 PPO 随机探索采样 → 立即摔倒。详见[不稳定性分析](docs/INSTABILITY_ANALYSIS.md)
- **修复方案明确**：移除 Action 0 + 熵衰减 + 加重摔倒惩罚，预计可将后期失败率从 3~7% 降到 <0.5%

---

## 🔬 不稳定性深度分析

完整报告：**[docs/INSTABILITY_ANALYSIS.md](docs/INSTABILITY_ANALYSIS.md)**

包含：全量 777 回合的分段数据、失败模式统计、4 个根因的量化分析、3 个修复方案及预期效果对比。

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
