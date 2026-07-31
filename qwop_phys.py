"""QWOP 物理环境（无浏览器，纯 Python Box2D）。

用 Box2D 在进程内复刻 QWOP 的 12 刚体 ragdoll 与 Q/W/O/P 肌肉控制，
接口（obs 88 / 动作 9 / 奖励 / 跨栏 / 沙坑）与原浏览器版完全一致，
因此 train.py / watch.py / dashboard.py 几乎无需改动即可切换。

obs 布局（与 game/bridge.js 一致）:
    12 个刚体 x (相对躯干x/10, y/10, sin角, cos角, vx/10, vy/10, 角速度/10)
    + 4 个按键状态 (Q,W,O,P)
    = 12*7 + 4 = 88 维

坐标尺度: 1 米 = 10 世界单位（与 bridge.js 的 metres = x/10 对应）。
课程: 起点 x=0，跨栏 x=500(50m)，终点沙坑 x=1000(100m)。
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

try:
    from Box2D import (
        b2World, b2BodyDef, b2PolygonShape, b2CircleShape, b2FixtureDef,
        b2RevoluteJointDef, b2_dynamicBody, b2_staticBody,
    )
    _HAS_BOX2D = True
except Exception:  # noqa: BLE001
    _HAS_BOX2D = False

import gymnasium as gym
from gymnasium import spaces

# ---- 与浏览器版保持一致的常量 ----
OBS_DIM = 12 * 7 + 4
FINISH_METRES = 100.0
HURDLE_X = 500.0          # 世界单位(decimetre) = 50m
FINISH_X = 1000.0         # = 100m
DT = 1.0 / 30.0

# 12 个刚体，顺序必须和 bridge.js 的 PARTS 完全一致
PARTS = [
    "torso", "head",
    "leftArm", "leftForearm", "leftThigh", "leftCalf", "leftFoot",
    "rightArm", "rightForearm", "rightThigh", "rightCalf", "rightFoot",
]

# 9 个离散动作（与 qwop_env.py 完全一致）
ACTIONS = [
    (False, False, False, False),
    (True, False, False, False),   # Q
    (False, True, False, False),   # W
    (False, False, True, False),   # O
    (False, False, False, True),   # P
    (True, False, True, False),    # Q+O
    (True, False, False, True),    # Q+P
    (False, True, True, False),    # W+O
    (False, True, False, True),    # W+P
]

REWARD_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_config.json")

# ---------------------------------------------------------------------------
# 刚体几何（decimetre 单位，1m=10u，地面 y=0，角色面朝 +x）
#   hx,hy: 半宽/半高（圆用 r）; cx,cy: 初始中心; density; friction
# ---------------------------------------------------------------------------
GEOM = {
    "torso":       dict(shape="box", hx=1.5, hy=3.5, cx=0.0,  cy=13.5, density=0.8, friction=0.6),
    "head":        dict(shape="circle", r=1.2,        cx=0.0,  cy=18.2, density=0.5, friction=0.4),
    "leftArm":     dict(shape="box", hx=0.6, hy=1.75, cx=-1.8, cy=15.25, density=0.6, friction=0.4),
    "leftForearm": dict(shape="box", hx=0.5, hy=1.75, cx=-2.4, cy=11.75, density=0.5, friction=0.4),
    "leftThigh":   dict(shape="box", hx=1.0, hy=2.25, cx=-0.7, cy=7.75, density=1.2, friction=0.6),
    "leftCalf":    dict(shape="box", hx=0.8, hy=2.25, cx=-1.0, cy=3.25, density=1.0, friction=0.6),
    "leftFoot":    dict(shape="box", hx=2.5, hy=0.5,  cx=-1.5, cy=0.5,  density=1.0, friction=1.6),
    "rightArm":    dict(shape="box", hx=0.6, hy=1.75, cx=1.8,  cy=15.25, density=0.6, friction=0.4),
    "rightForearm":dict(shape="box", hx=0.5, hy=1.75, cx=2.4,  cy=11.75, density=0.5, friction=0.4),
    "rightThigh":  dict(shape="box", hx=1.0, hy=2.25, cx=0.7,  cy=7.75, density=1.2, friction=0.6),
    "rightCalf":   dict(shape="box", hx=0.8, hy=2.25, cx=1.0,  cy=3.25, density=1.0, friction=0.6),
    "rightFoot":   dict(shape="box", hx=2.5, hy=0.5,  cx=1.5,  cy=0.5,  density=1.0, friction=1.6),
}

# 关节: (bodyA, bodyB, 世界锚点(x,y), 电机标识或None, (下限角, 上限角))
# 电机标识: hipL/kneeL/hipR/kneeR 对应 Q/W/O/P
JOINTS = [
    ("torso", "head",        (0.0, 17.0),  None,    (-0.6, 0.6)),
    ("torso", "leftArm",     (-1.0, 17.0), None,    (-2.6, 0.6)),
    ("leftArm", "leftForearm",(-1.8, 13.5),None,    (-0.3, 2.6)),
    ("torso", "leftThigh",   (-0.7, 10.0), "hipL",  (-1.1, 1.1)),
    ("leftThigh", "leftCalf",(-0.7, 5.5),  "kneeL", (0.0, 2.2)),
    ("leftCalf", "leftFoot", (-1.0, 1.0),  None,    (-0.7, 0.7)),
    ("torso", "rightArm",    (1.0, 17.0),  None,    (-0.6, 2.6)),
    ("rightArm", "rightForearm",(1.8, 13.5),None,   (-2.6, 0.3)),
    ("torso", "rightThigh",  (0.7, 10.0),  "hipR",  (-1.1, 1.1)),
    ("rightThigh", "rightCalf",(0.7, 5.5), "kneeR", (0.0, 2.2)),
    ("rightCalf", "rightFoot",(1.0, 1.0),  None,    (-0.7, 0.7)),
]

# 控制参数（电机 / 重力），集中放这里方便调
# 设计:
#   髋部: 按住键 -> 以 ±HIP_SWING_SPEED 摆动(模拟大腿肌肉收缩); 松开键 -> 用"核心 PD"
#         主动把躯干拉回直立(模拟核心张力, 使站姿可稳住、可被学会)。
#   膝部: 按住键 -> ±KNEE_SWING_SPEED; 松开键 -> 以 KNEE_HOLD 刹车锁直。
GRAVITY = -90.0
HIP_SWING_SPEED = 5.0
KNEE_SWING_SPEED = 6.0
# 重构(冲33s): 放开关节驱动扭矩, 让腿能真正快速摆动形成奔跑
HIP_PRESS = 65000.0     # 按 Q/O 时髋部驱动扭矩（52s 最优配置）
HIP_HOLD = 20000.0      # 松键时髋部刹车/保持扭矩（锁住大腿姿态, 需撑住体重）
KNEE_PRESS = 38000.0    # 按 W/P 时膝部驱动扭矩（52s 最优配置）
KNEE_HOLD = 10000.0     # 松键时膝部刹车/保持扭矩（防止膝盖在体重下弯折）
# 躯干核心稳定: 每帧直接对躯干施加 PD 扶正力矩, 保证底座可学。
# 注: 实验已证(2026-07-30) 改为"前倾/软核心"奔跑重构均退化为 ~71s; 直立+CLAMP=26 即 52s
# 是本位置式控制 + 始终扶正设计的速度上限(无腾空相, 只能快走, 见 README)。
CORE_KP = 40000.0
CORE_KD = 5000.0
CORE_MAX = 70000.0

# ===========================================================================
# 深度动力学重构 (冲 33s / 真正奔跑): RUN_RECONSTRUCT = True
# ---------------------------------------------------------------------------
# 根因: 52s 天花板来自"核心始终锁直立"——躯干是根刚性竖直杆, 质心永远在双脚
# 支撑面之上, 永远不会出现"质心超前→腾空→下一脚接住"的奔跑循环, 只能快走。
# 重构让 ragdoll 真正能跑(带腾空相):
#   1) 条件式核心: 小角度时只给弱弹簧(允许前倾), 超过防摔阈值才上强 PD。
#      制造"可被学习的不稳定可恢复态"= 奔跑所需的动力学。
#   2) 放开关节速度钳制 CLAMP 26 -> RUN_CLAMP, 腿能更快摆动/蹬伸, 抬高步频上限。
#   3) 降低躯干/头转动惯量(密度下调), 躯干更易前倾摆动, 奔跑步态更易学。
# 设为 False 时, 全部回退到 52s 最优行为(与原版逐字节一致), 用于保留备份。
#
# 【实验结论 2026-07-30/31】重构未能突破 52s, 故**默认关闭**:
#   - v1 条件式核心 + 放开 CLAMP (3M 步) -> 57.1s / 1.75 m/s
#   - v2 再叠加力矩驱动肌肉 (2M 步)      -> 67.8s / 1.47 m/s
#   两者均慢于 52s 基准。详见 README「深度动力学重构」一节。
# 用环境变量 QWOP_RECON=1 开启以复现该实验。
# ===========================================================================
RUN_RECONSTRUCT = os.environ.get("QWOP_RECON", "0") == "1"
CORE_LEAN_TARGET = 0.25     # 平时弱弹簧目标前倾角(rad), 默认略前倾更利于奔跑
CORE_WEAK_KP = 8000.0       # 平时弱弹簧刚度(允许前倾/奔跑, 不至于一碰就倒)
CORE_WEAK_KD = 1500.0       # 平时弱弹簧阻尼
CORE_PANIC_ANGLE = 0.70     # |躯干角|超过此值(约40°)进入强 PD 防摔
RUN_CLAMP = 55.0            # 重构模式关节速度上限(放开以提速)
# 重构模式下调低躯干/头密度, 降低转动惯量
if RUN_RECONSTRUCT:
    GEOM["torso"]["density"] = 0.55
    GEOM["head"]["density"] = 0.40

# ---------------------------------------------------------------------------
# 深度重构 v2 (冲33s 真正奔跑): RUN_TORQUE = True —— 肌肉改为力矩驱动
# ---------------------------------------------------------------------------
# 关键发现(2026-07-30): 位置伺服下达的电机速度 = K_POS*(target-angle) ≈ 25*0.95 ≈ 24,
# 真正限速的是 K_POS 而非 CLAMP; 故仅放开 CLAMP 几乎不影响腿速, v1 仍卡在~60s 快走。
# 要真正提速/蹬地腾空, 必须让肌肉"用力": 按下键时以最大扭矩朝目标方向猛推到关节限位
# (爆发性蹬伸 + 更大步幅); 松开键时仍用位置伺服温和回中(受控恢复, 不至于失控)。
# 设 False 时(即便 RUN_RECONSTRUCT=True)仍用位置伺服, 回退到 v1 行为。
# 用环境变量 QWOP_TORQUE 控制(默认关闭), 便于在不同物理实验间 A/B 而不互相污染
# (v1 评估不设变量->False 与训练一致; v2 训练/评估设 QWOP_TORQUE=1)。
RUN_TORQUE = os.environ.get("QWOP_TORQUE", "0") == "1"
TORQUE_DRIVE_SPEED = 60.0   # 按下键时的电机目标速度(远超可达->等效恒定最大扭矩驱动)


class QWOPPhysics:
    """纯 Box2D 的 QWOP 物理世界。"""

    def __init__(self):
        if not _HAS_BOX2D:
            raise RuntimeError("Box2D 未安装，无法使用物理后端 (pip install box2d)")
        self.world = b2World(gravity=(0, GRAVITY))
        self.bodies = {}
        self.motors = {}          # hipL/kneeL/hipR/kneeR -> joint
        self._prev_angles = None
        self._build()

    def _build(self):
        w = self.world
        self._keys = {"hipL": False, "kneeL": False, "hipR": False, "kneeR": False}
        # 地面（长条，覆盖整条赛道）
        ground = w.CreateStaticBody(position=(FINISH_X / 2.0, -1.0),
                                    shapes=b2PolygonShape(box=(FINISH_X / 2.0 + 200, 1.0)))
        ground.friction = 1.2
        self.ground = ground

        # 跨栏（传感器，仅用于可视化与 hurdleX 取值，不阻挡）
        hurdle = w.CreateStaticBody(position=(HURDLE_X, 1.5))
        hd = b2FixtureDef(shape=b2PolygonShape(box=(0.6, 1.5)), isSensor=True)
        hurdle.CreateFixture(hd)
        self.hurdle = hurdle

        for name in PARTS:
            g = GEOM[name]
            bd = b2BodyDef(type=b2_dynamicBody, position=(g["cx"], g["cy"]))
            body = w.CreateBody(bd)
            if g["shape"] == "box":
                shape = b2PolygonShape(box=(g["hx"], g["hy"]))
            else:
                shape = b2CircleShape(radius=g["r"])
            fd = b2FixtureDef(shape=shape, density=g["density"],
                              friction=g["friction"], restitution=0.0)
            fd.filter_groupIndex = -1   # 同组负索引 -> 刚体之间不互相碰撞
            body.CreateFixture(fd)
            self.bodies[name] = body

        for a, b, anchor, motor, (lo, hi) in JOINTS:
            ba, bb = self.bodies[a], self.bodies[b]
            jd = b2RevoluteJointDef()
            jd.bodyA = ba
            jd.bodyB = bb
            jd.localAnchorA = (anchor[0] - ba.position.x, anchor[1] - ba.position.y)
            jd.localAnchorB = (anchor[0] - bb.position.x, anchor[1] - bb.position.y)
            jd.collideConnected = False
            jd.enableLimit = True
            jd.lowerAngle = lo
            jd.upperAngle = hi
            jd.enableMotor = False
            jd.maxMotorTorque = 0.0
            jd.motorSpeed = 0.0
            joint = w.CreateJoint(jd)
            if motor:
                self.motors[motor] = joint

    def set_keys(self, q, w, o, p):
        """根据 Q/W/O/P 开关对应电机。

        默认(位置式伺服): 髋前伸=负角, 膝屈曲=正角; 松开回中立。
        重构 v2(RUN_TORQUE=True): 按下键时以最大扭矩朝目标方向猛推到关节限位
        (爆发性蹬伸), 松开键时用位置伺服温和回中。见文件顶部 RUN_TORQUE 说明。
        躯干稳定由 _stabilize_core 处理。
        """
        want = {"hipL": q, "kneeL": w, "hipR": o, "kneeR": p}
        # 位置控制目标角: 髋前伸=负角(标定为前进方向), 膝屈曲=正角
        HIP_TARGET = -0.95   # 52s 最优配置: 较大前伸步幅
        KNEE_TARGET = 1.2    # 较大屈膝以获得蹬地推力
        K_POS = 25.0
        # 重构模式放开速度钳制(26->RUN_CLAMP), 让腿能更快摆动/蹬伸以形成奔跑
        CLAMP = RUN_CLAMP if RUN_RECONSTRUCT else 26.0
        self._keys = want
        for key, joint in self.motors.items():
            is_hip = "hip" in key
            if is_hip:
                target = HIP_TARGET if want[key] else 0.0
                torque = HIP_PRESS if want[key] else HIP_HOLD
                # 髋前伸方向: 负角=负方向; 松键回中立=正方向
                drive_sign = -1.0 if (want[key]) else 1.0
            else:
                target = KNEE_TARGET if want[key] else 0.0
                torque = KNEE_PRESS if want[key] else KNEE_HOLD
                # 膝屈曲方向: 正角=正方向; 松键伸直=负方向
                drive_sign = 1.0 if (want[key]) else -1.0
            if RUN_TORQUE and want[key]:
                # 按下键: 力矩驱动, 最大扭矩猛推到限位(爆发性蹬伸/前摆)
                motor_speed = drive_sign * TORQUE_DRIVE_SPEED
            else:
                # 位置伺服: 温和趋近目标角(或松键回中)
                motor_speed = max(-CLAMP, min(CLAMP, K_POS * (target - joint.angle)))
            joint.motorEnabled = True
            joint.motorSpeed = motor_speed
            joint.maxMotorTorque = torque

    def _stabilize_core(self):
        """每帧直接对躯干施加 PD 扶正力矩（模拟核心张力）。

        非重构模式(52s 最优): 锁直立(angle=0)。
        重构模式(冲33s): 条件式核心——小角度只给弱弹簧(允许前倾/奔跑),
        超过 CORE_PANIC_ANGLE 才上强 PD 防摔, 制造可被学习的"不稳定可恢复态",
        这是出现腾空相、真正奔跑的前提(见文件顶部 RUN_RECONSTRUCT 说明)。
        """
        torso = self.bodies["torso"]
        a = torso.angle
        av = torso.angularVelocity
        if RUN_RECONSTRUCT and abs(a) > CORE_PANIC_ANGLE:
            # 接近摔倒: 强 PD 把躯干拉回直立, 防止彻底扑街
            t = -(CORE_KP * a + CORE_KD * av)
        elif RUN_RECONSTRUCT:
            # 平时: 弱弹簧维持略前倾姿态, 允许躯干随步姿前倾产生奔跑腾空相
            t = -(CORE_WEAK_KP * (a - CORE_LEAN_TARGET) + CORE_WEAK_KD * av)
        else:
            # 52s 最优: 始终锁直立
            t = -(CORE_KP * a + CORE_KD * av)
        t = max(-CORE_MAX, min(CORE_MAX, t))
        torso.ApplyTorque(t, True)

    def step(self, n_frames: int):
        for _ in range(n_frames):
            self._stabilize_core()
            self.world.Step(DT, 8, 3)

    def reset(self):
        # 直接重建世界比逐一复位更稳
        self.world = b2World(gravity=(0, GRAVITY))
        self.bodies = {}
        self.motors = {}
        self._prev_angles = None
        self._build()
        # 让初始姿态稳定几帧
        self.step(3)

    def _body_center(self, name):
        return self.bodies[name].position

    def get_state(self):
        torso = self.bodies["torso"]
        tc = torso.position
        obs = []
        angles = []
        for name in PARTS:
            b = self.bodies[name]
            c = b.position
            a = b.angle
            v = b.linearVelocity
            angles.append(a)
            av = 0.0
            if self._prev_angles is not None:
                av = (a - self._prev_angles[PARTS.index(name)]) / DT
            obs.extend([
                (c.x - tc.x) / 10.0,
                c.y / 10.0,
                math.sin(a),
                math.cos(a),
                v.x / 10.0,
                v.y / 10.0,
                av / 10.0,
            ])
        self._prev_angles = angles
        # 当前按键状态（真实按键，而非电机使能）
        keys = [self._keys[m] for m in ("hipL", "kneeL", "hipR", "kneeR")]
        obs.extend([1.0 if k else 0.0 for k in keys])

        headY = self.bodies["head"].position.y
        x = tc.x
        metres = x / 10.0
        fallen = (tc.y < 5.0) or (abs(torso.angle) > 1.4) or (headY < 2.0)
        jumped = x >= HURDLE_X
        jumpLanded = x >= FINISH_X
        return {
            "obs": obs,
            "x": x,
            "metres": metres,
            "score": metres,
            "torsoY": tc.y,
            "torsoAngle": torso.angle,
            "headY": headY,
            "hasHurdle": True,
            "hurdleX": HURDLE_X,
            "gameOver": bool(fallen),
            "gameEnded": bool(fallen),
            "fallen": bool(fallen),
            "jumped": bool(jumped),
            "jumpLanded": bool(jumpLanded),
        }

    def get_pose(self):
        pts = {}
        for name in PARTS:
            b = self.bodies[name]
            pts[name] = {"x": b.position.x, "y": b.position.y, "a": b.angle}
        return pts


# ---------------------------------------------------------------------------
# Gymnasium 环境：奖励逻辑与 qwop_env.py 完全一致
# ---------------------------------------------------------------------------
class QWOPPhysEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, frames_per_step: int = 3, max_steps: int = 2000, **_kw):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.frames_per_step = frames_per_step
        self.max_steps = max_steps
        self._phys = QWOPPhysics()
        self._steps = 0
        self._last_metres = 0.0
        self._best_metres = 0.0
        self._hurdle_x = None
        self._crossed_hurdle = False
        self._standing_at_hurdle = False
        self._finished = False
        self._reward_cfg = None
        self._reward_mtime = 0.0

    def _load_reward_config(self):
        try:
            mtime = os.path.getmtime(REWARD_CONFIG_PATH)
        except OSError:
            mtime = 0
        if self._reward_cfg is None or mtime != self._reward_mtime:
            try:
                with open(REWARD_CONFIG_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
            except Exception:
                cfg = {
                    "forward_per_metre": 1.0, "time_penalty_per_step": 0.002,
                    "fall_penalty": 5.0, "timeout_penalty": 5.0,
                    "hurdle_stand_bonus": 8.0, "hurdle_crawl_bonus": 1.0,
                    "hurdle_stand_sin_threshold": 0.7, "hurdle_stand_headY_threshold": 0.0,
                    "jump_distance_bonus": 1.0, "success_terminal_bonus": 50.0,
                    "speed_bonus_coef": 0.0, "velocity_coeff": 0.0,
                }
            self._reward_cfg = cfg
            self._reward_mtime = mtime
        return self._reward_cfg

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._phys.reset()
        state = self._phys.get_state()
        self._steps = 0
        self._last_metres = state["metres"]
        self._last_x = state["x"]
        self._hurdle_x = state.get("hurdleX")
        self._crossed_hurdle = False
        self._standing_at_hurdle = False
        self._finished = False
        obs = np.asarray(state["obs"], dtype=np.float32)
        return obs, {"metres": state["metres"]}

    def step(self, action):
        q, w, o, p = ACTIONS[int(action)]
        self._phys.set_keys(q, w, o, p)
        self._phys.step(self.frames_per_step)
        state = self._phys.get_state()
        self._steps += 1
        metres = float(state["metres"])
        x = float(state["x"])
        finished = bool(state["jumpLanded"])
        fell = (bool(state["fallen"]) or bool(state["gameEnded"])
                or bool(state["gameOver"])) and not finished
        cfg = self._load_reward_config()

        reward = cfg["forward_per_metre"] * (metres - self._last_metres)
        reward -= cfg["time_penalty_per_step"]

        # 密集前向速度奖励：每步直接奖励"此刻前进有多快"，把目标从"到达终点"
        # 扭成"尽快前进"（对比只有终点速度奖励，这个信号每步都在，学习快得多）。
        vx = (x - self._last_x) / 10.0 / (self.frames_per_step * DT)  # m/s (1u=0.1m)
        reward += cfg.get("velocity_coeff", 0.0) * vx
        self._last_x = x

        if self._hurdle_x is not None and not self._crossed_hurdle and x >= self._hurdle_x:
            self._crossed_hurdle = True
            sin_a = abs(math.sin(state.get("torsoAngle", 0.0)))
            head_y = state.get("headY", 0.0)
            standing = (sin_a < cfg["hurdle_stand_sin_threshold"]
                        and head_y > cfg["hurdle_stand_headY_threshold"])
            self._standing_at_hurdle = standing
            reward += cfg["hurdle_stand_bonus"] if standing else cfg["hurdle_crawl_bonus"]

        if finished and not self._finished:
            self._finished = True
            final_m = max(metres, float(state.get("score") or 0))
            reward += cfg["jump_distance_bonus"] * final_m + cfg["success_terminal_bonus"]
            # 速度奖励: 终点速度越高(用时越短)奖励越大, 直接把目标从"到达终点"扭成"尽快到达"
            coef = cfg.get("speed_bonus_coef", 0.0)
            if coef:
                time_sec = self._steps * self.frames_per_step * DT
                reward += coef * (100.0 / max(time_sec, 1e-3))

        if fell:
            reward -= cfg["fall_penalty"]

        self._last_metres = metres
        self._best_metres = max(self._best_metres, metres)

        terminated = finished or fell
        truncated = (not terminated) and self._steps >= self.max_steps
        if truncated and not finished and not fell:
            reward -= cfg["timeout_penalty"]

        obs = np.asarray(state["obs"], dtype=np.float32)
        info = {
            "metres": metres, "best_metres": self._best_metres,
            "fallen": fell, "success": finished,
            "crossed_hurdle": self._crossed_hurdle,
            "standing_at_hurdle": self._standing_at_hurdle,
            "action": int(action),
        }
        return obs, reward, terminated, truncated, info

    def get_pose(self):
        return self._phys.get_pose()

    def close(self):
        self._phys = None


if __name__ == "__main__":
    # 冒烟测试: 随机动作，统计前进距离 / 摔倒率 / 速度
    env = QWOPPhysEnv()
    obs, info = env.reset()
    print(f"obs shape: {obs.shape}, start metres: {info['metres']:.2f}")
    import time
    t0 = time.time()
    n = 0
    best = 0.0
    falls = 0
    eps = 0
    while eps < 30:
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        n += 1
        best = max(best, info["metres"])
        if term or trunc:
            if info["fallen"]:
                falls += 1
            print(f"ep{eps}: metres={info['metres']:.2f} fallen={info['fallen']} "
                  f"crossed={info['crossed_hurdle']} success={info['success']}")
            env.reset()
            eps += 1
    dt = time.time() - t0
    fps = n * env.frames_per_step / dt
    print(f"{n} env steps, {n*env.frames_per_step} 物理帧, 用时 {dt:.1f}s")
    print(f"≈ {fps:.0f} 物理帧/秒  ≈ {fps/30:.1f}x 实时 (单环境)")
    print(f"随机策略最远 {best:.2f}m, 摔倒率 {falls}/{eps}")
