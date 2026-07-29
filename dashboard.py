"""QWOP 训练实时仪表盘。

用法:
    python dashboard.py            # 起服务，浏览器打开提示的地址
    # 然后另开一个终端跑 python train.py，仪表盘每 1 秒自动刷新

功能:
    GET  /                -> dashboard.html
    GET  /api/metrics     -> 读取 train.py 写出的 metrics.json
    POST /api/reward      -> 写入 reward_config.json（实时调整奖励权重）
"""
from __future__ import annotations

import functools
import http.server
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "metrics.json")
REWARD_PATH = os.path.join(HERE, "reward_config.json")
HTML_PATH = os.path.join(HERE, "dashboard.html")


class Handler(http.server.BaseHTTPRequestHandler):
    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, b"", "text/plain")

    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    data = f.read()
            except FileNotFoundError:
                data = json.dumps({"step": 0, "total_steps": 0, "best_metres": 0,
                                   "episodes": 0, "fps": 0, "recent": [],
                                   "action_dist": {}, "history": [],
                                   "hurdle_cross_rate": 0, "success_rate": 0})
            self._send(200, data.encode("utf-8"))
        elif self.path.startswith("/api/reward"):
            try:
                with open(REWARD_PATH, "r", encoding="utf-8") as f:
                    data = f.read()
            except FileNotFoundError:
                data = "{}"
            self._send(200, data.encode("utf-8"))
        else:
            try:
                with open(HTML_PATH, "r", encoding="utf-8") as f:
                    html = f.read()
            except FileNotFoundError:
                html = "<h1>dashboard.html 缺失</h1>"
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        if self.path.startswith("/api/reward"):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send(400, b'{"error":"bad json"}')
                return
            # 只保留数值权重
            clean = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
            try:
                with open(REWARD_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                for k, v in clean.items():
                    cfg[k] = v
                with open(REWARD_PATH, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                self._send(200, json.dumps({"ok": True, "cfg": cfg}).encode("utf-8"))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self._send(404, b"")

    def log_message(self, *a):
        pass


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"仪表盘已启动: http://127.0.0.1:{args.port}")
    print("提示: 另开终端运行  python train.py  ，本页每 1 秒自动刷新。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
