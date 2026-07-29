"""下载 QWOP 游戏运行时并自动打补丁 (训练所需)。

QWOP 游戏本体版权归 Bennett Foddy 所有。本仓库不打包游戏二进制文件,
而是通过本脚本从官方源 (https://www.foddy.net/legacy/) 获取原版文件,
并应用 4 处最小补丁以: 暴露游戏实例 / 禁用引擎自毁 / 支持跳帧渲染。

用法:
    python fetch_game.py
"""
from __future__ import annotations

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
GAME_DIR = os.path.join(HERE, "game")

BASE = "https://www.foddy.net/legacy"
FILES = {
    "QWOP.min.js": f"{BASE}/QWOP.min.js",
    "lib/howler.js": f"{BASE}/lib/howler.js",
    "assets/assetbundle.parcel": f"{BASE}/assets/assetbundle.parcel",
}

# 4 处补丁: (原字符串, 替换字符串) —— 均在官方原版中唯一出现
PATCHES = [
    (
        'ready:function(){var t=window.document.getElementById("gameContent")',
        'ready:function(){window.QWOPGame=this;var t=window.document.getElementById("gameContent")',
    ),
    ('case 7:case 8:this.shutdown();break;', 'case 7:case 8:break;'),
    (
        'shutdown:function(){this.shutting_down=!0,this.host.ondestroy()',
        'shutdown:function(){return;this.shutting_down=!0,this.host.ondestroy()',
    ),
    (
        'shutdown:function(){this.shutting_down=!0,C.Snow.next(',
        'shutdown:function(){return;this.shutting_down=!0,C.Snow.next(',
    ),
    (
        'snow_core_loop:function(t){return null==t&&(t=.016),this.update(),this.app.on_event({type:3}),this.request_update(),!0}',
        'snow_core_loop:function(t){return null==t&&(t=.016),this.update(),window.QWOPBridge&&window.QWOPBridge.skipRender||this.app.on_event({type:3}),this.request_update(),!0}',
    ),
]


def download(rel_path: str, url: str) -> str:
    dest = os.path.join(GAME_DIR, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"下载 {url} -> {dest}")
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def apply_patches(path: str) -> None:
    with open(path, "r", encoding="utf-8", errors="surrogateescape") as f:
        s = f.read()
    for original, replacement in PATCHES:
        n = s.count(original)
        if n != 1:
            raise SystemExit(
                f"[错误] 补丁锚点未精确匹配 (出现 {n} 次):\n{original[:60]}...\n"
                "官方游戏文件可能已更新, 请提 issue 或手动更新补丁。"
            )
        s = s.replace(original, replacement)
    with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
        f.write(s)
    print(f"已对 {path} 应用 {len(PATCHES)} 处补丁")


def main() -> None:
    os.makedirs(GAME_DIR, exist_ok=True)
    for rel, url in FILES.items():
        download(rel, url)
    apply_patches(os.path.join(GAME_DIR, "QWOP.min.js"))
    print("\n完成。游戏运行时已就绪, 现在可以运行 train.py / watch.py。")


if __name__ == "__main__":
    main()
