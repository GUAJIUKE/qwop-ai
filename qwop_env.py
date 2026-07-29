"""QWOP Gymnasium 环境（Playwright 驱动浏览器，同步快进物理）。

观测: 88 维 float32
    12 个刚体 x (相对躯干x, y, sin角, cos角, vx, vy, 角速度) + 4 个按键状态
动作: 9 个离散动作（QWOP 组合键）
奖励: 前进距离(米) - 时间惩罚 - 摔倒惩罚 + 到达终点奖励
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import threading
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

GAME_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game")
REWARD_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reward_config.json")

import math

# 9 个动作: 无 / 单键 / 常用组合（Q+O, Q+P, W+O, W+P）
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

OBS_DIM = 12 * 7 + 4
FINISH_METRES = 100.0


def _start_http_server(directory: str) -> int:
    """在随机端口起一个静态文件服务，返回端口号。"""
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=directory
    )
    handler.log_message = lambda *a, **k: None  # 静音
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd.server_address[1]


class QWOPEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        headless: bool = True,
        frames_per_step: int = 3,
        max_steps: int = 2000,
        realtime: bool = False,
        browser_channel: str | None = None,
    ):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.headless = headless
        self.frames_per_step = frames_per_step
        self.max_steps = max_steps
        self.realtime = realtime
        self.browser_channel = browser_channel
        self._pw = None
        self._browser = None
        self._page = None
        self._steps = 0
        self._last_metres = 0.0
        self._best_metres = 0.0
        # 跨栏 / 跳远 / 奖励配置跟踪
        self._hurdle_x = None
        self._crossed_hurdle = False
        self._standing_at_hurdle = False
        self._finished = False
        self._reward_cfg = None
        self._reward_mtime = 0.0

    # ---------- 浏览器管理 ----------

    def _launch(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        args = ["--disable-gpu-vsync", "--mute-audio", "--autoplay-policy=no-user-gesture-required"]
        candidates = (
            [self.browser_channel] if self.browser_channel else ["msedge", "chrome", None]
        )
        last_err = None
        for ch in candidates:
            try:
                self._browser = self._pw.chromium.launch(
                    headless=self.headless, channel=ch, args=args
                )
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        if self._browser is None:
            raise RuntimeError(f"无法启动浏览器: {last_err}")

        port = _start_http_server(GAME_DIR)
        self._page = self._browser.new_page(viewport={"width": 660, "height": 420})
        self._page.goto(f"http://127.0.0.1:{port}/qwop.html")
        # 等待资源包加载完毕、游戏就绪
        self._page.wait_for_function(
            "window.QWOPBridge && window.QWOPBridge.ready()", timeout=60_000
        )
        # 点掉开场画面（firstClick）
        self._page.mouse.click(320, 200)
        self._page.wait_for_timeout(100)
        # 切换到手动步进模式，并等待游戏循环回调被捕获
        self._page.evaluate("QWOPBridge.startManual()")
        # 注意: 必须用定时轮询, RAF 轮询已被 bridge 劫持
        self._page.wait_for_function(
            "QWOPBridge.cb !== null", timeout=10_000, polling=100
        )

    def _ensure_page(self):
        if self._page is None:
            self._launch()

    # ---------- Gym 接口 ----------

    def _load_reward_config(self):
        """按 mtime 实时重载奖励权重（训练时可被仪表盘/手动修改）。"""
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
                    "forward_per_metre": 1.0,
                    "time_penalty_per_step": 0.002,
                    "fall_penalty": 5.0,
                    "timeout_penalty": 5.0,
                    "hurdle_stand_bonus": 8.0,
                    "hurdle_crawl_bonus": 1.0,
                    "hurdle_stand_sin_threshold": 0.7,
                    "hurdle_stand_headY_threshold": 0.0,
                    "jump_distance_bonus": 1.0,
                    "success_terminal_bonus": 50.0,
                }
            self._reward_cfg = cfg
            self._reward_mtime = mtime
        return self._reward_cfg

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_page()
        state = self._page.evaluate("(QWOPBridge.reset(), QWOPBridge.getState())")
        self._steps = 0
        self._last_metres = state["metres"]
        self._hurdle_x = state.get("hurdleX")
        self._crossed_hurdle = False
        self._standing_at_hurdle = False
        self._finished = False
        obs = np.asarray(state["obs"], dtype=np.float32)
        return obs, {"metres": state["metres"]}

    def step(self, action):
        q, w, o, p = ACTIONS[int(action)]
        state = self._page.evaluate(
            "([q,w,o,p,f]) => QWOPBridge.act(q,w,o,p,f)",
            [q, w, o, p, self.frames_per_step],
        )
        self._steps += 1
        metres = float(state["metres"])
        x = float(state["x"])
        finished = bool(state["jumpLanded"])
        fell = bool(state["fallen"]) or bool(state["gameEnded"]) or bool(state["gameOver"])
        fell = fell and not finished
        cfg = self._load_reward_config()

        # ---- 奖励设计（按用户要求）----
        reward = cfg["forward_per_metre"] * (metres - self._last_metres)  # 前进 1m 奖励
        reward -= cfg["time_penalty_per_step"]                            # 时间惩罚（鼓励快）

        # 跨栏：首次躯干越过栏架 x 坐标
        if self._hurdle_x is not None and not self._crossed_hurdle and x >= self._hurdle_x:
            self._crossed_hurdle = True
            sin_a = abs(math.sin(state.get("torsoAngle", 0.0)))
            head_y = state.get("headY", 0.0)
            standing = (sin_a < cfg["hurdle_stand_sin_threshold"]
                        and head_y > cfg["hurdle_stand_headY_threshold"])
            self._standing_at_hurdle = standing
            reward += cfg["hurdle_stand_bonus"] if standing else cfg["hurdle_crawl_bonus"]

        # 终点跳远：完成跳远落地，成绩越高奖励越大
        if finished and not self._finished:
            self._finished = True
            final_m = max(metres, float(state.get("score") or 0))
            reward += cfg["jump_distance_bonus"] * final_m + cfg["success_terminal_bonus"]

        # 摔倒惩罚
        if fell:
            reward -= cfg["fall_penalty"]

        self._last_metres = metres
        self._best_metres = max(self._best_metres, metres)

        terminated = finished or fell
        truncated = (not terminated) and self._steps >= self.max_steps
        if truncated and not finished and not fell:
            reward -= cfg["timeout_penalty"]   # 超时惩罚（没跑完）

        obs = np.asarray(state["obs"], dtype=np.float32)
        info = {
            "metres": metres,
            "best_metres": self._best_metres,
            "fallen": fell,
            "success": finished,
            "crossed_hurdle": self._crossed_hurdle,
            "standing_at_hurdle": self._standing_at_hurdle,
            "action": int(action),
        }
        if self.realtime:
            time.sleep(self.frames_per_step / 30.0)
        return obs, reward, terminated, truncated, info

    def close(self):
        for closer in (
            lambda: self._browser.close() if self._browser else None,
            lambda: self._pw.stop() if self._pw else None,
        ):
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        self._browser = None
        self._page = None
        self._pw = None


if __name__ == "__main__":
    # 冒烟测试: 随机动作
    env = QWOPEnv(headless=True)
    obs, info = env.reset()
    print(f"obs shape: {obs.shape}, start metres: {info['metres']:.2f}")
    total = 0.0
    t0 = time.time()
    n = 0
    for ep in range(3):
        obs, _ = env.reset()
        ep_r = 0.0
        while True:
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            n += 1
            if term or trunc:
                print(
                    f"ep{ep}: steps={n} metres={info['metres']:.2f} "
                    f"fallen={info['fallen']} reward={ep_r:.2f}"
                )
                break
    dt = time.time() - t0
    print(f"{n} env steps ({n * env.frames_per_step} 物理帧) 用时 {dt:.1f}s "
          f"≈ {n * env.frames_per_step / 30 / dt:.1f}x 实时速度")
    env.close()
