"""加载训练好的模型, 打开有头浏览器实时观看 AI 跑步, 并叠加 HUD/骨架。

用法:
    python watch.py                    # 游戏画面 + HUD 数据浮层
    python watch.py --mode pose        # 额外叠加人形骨架(看跑姿)
    python watch.py --model 路径.zip --episodes 5
"""
from __future__ import annotations

# ---- 依赖自检: 若当前 Python 未安装 RL 库, 自动用隔离 venv 重跑 ----
import sys, os, subprocess
_VENV = r"C:\Users\24479\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
try:
    import stable_baselines3  # noqa: F401
except ImportError:
    if os.path.exists(_VENV) and sys.executable != _VENV:
        sys.exit(subprocess.call([_VENV, *sys.argv]))
    raise

import argparse
import glob
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from qwop_env import QWOPEnv

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

ACT_NAMES = {
    0: "静止", 1: "Q", 2: "W", 3: "O", 4: "P",
    5: "Q+O", 6: "Q+P", 7: "W+O", 8: "W+P",
}

OVERLAY_JS = r"""
window.__mode = window.__mode || 'game';
window.__hist = [];
function el(id,tag){var e=document.getElementById(id);if(!e){e=document.createElement(tag||'div');e.id=id;document.body.appendChild(e);}return e;}
function initHUD(){
  var s=el('qwopHUD','div');
  s.style.cssText='position:fixed;left:12px;top:12px;z-index:9999;background:rgba(15,17,23,.82);'+
    'border:1px solid #2a2f3c;border-radius:10px;padding:10px 12px;color:#e6e8ee;'+
    'font:13px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;min-width:230px;backdrop-filter:blur(4px)';
  s.innerHTML='<div id="hud_dist" style="font-size:22px;font-weight:700">0.0 米</div>'+
    '<div id="hud_sub" style="color:#8b93a7;font-size:12px;margin-bottom:6px">目标 100 米</div>'+
    '<div id="hud_row" style="display:flex;gap:10px;font-size:12px"></div>'+
    '<canvas id="hud_cv" width="220" height="64" style="margin-top:8px;border-radius:6px;background:#0f1117"></canvas>';
  var sc=el('qwopSkeletonCanvas','canvas');
  sc.style.cssText='position:fixed;left:0;top:0;z-index:9998;pointer-events:none;display:none';
  sc.width=innerWidth;sc.height=innerHeight;
  if(window.__mode==='pose'){sc.style.display='block';}
  window.addEventListener('resize',function(){sc.width=innerWidth;sc.height=innerHeight;});
}
function drawHUD(){
  var g=window.QWOPGame, b=window.QWOPBridge; if(!g||!b||!b.getState)return;
  var st; try{st=b.getState();}catch(e){return;}
  if(!st)return;
  var dist=st.metres||0;
  el('hud_dist').textContent=dist.toFixed(1)+' 米';
  var keys=[];if(g.QDown)keys.push('Q');if(g.WDown)keys.push('W');if(g.ODown)keys.push('O');if(g.PDown)keys.push('P');
  var act=(window.__actName)||keys.join('+')||'—';
  var cross=st.hasHurdle?(window.__crossed?'已跨栏':'未跨栏'):'无栏';
  var jump=st.jumpLanded?'已跳远':'—';
  el('hud_row').innerHTML=
    '<span>动作:<b style="color:#4f8cff">'+act+'</b></span>'+
    '<span>跨栏:<b style="color:#39d98a">'+cross+'</b></span>'+
    '<span>跳远:<b style="color:#b07bff">'+jump+'</b></span>'+
    '<span>奖励:<b style="color:#ffb454">'+(window.__r!=null?window.__r.toFixed(2):'0')+'</b></span>';
  // 距离曲线
  window.__hist.push(dist); if(window.__hist.length>160)window.__hist.shift();
  var cv=el('hud_cv','canvas').getContext('2d'); var W=cv.canvas.width,H=cv.canvas.height;
  cv.clearRect(0,0,W,H); var hs=window.__hist; if(hs.length>1){
    var lo=Math.min.apply(null,hs),hi=Math.max.apply(null,hs); if(hi-lo<1)hi=lo+1;
    cv.strokeStyle='#39d98a';cv.lineWidth=2;cv.beginPath();
    for(var i=0;i<hs.length;i++){var x=4+i/(hs.length-1)*(W-8);var y=H-6-(hs[i]-lo)/(hi-lo)*(H-12);if(i===0)cv.moveTo(x,y);else cv.lineTo(x,y);}
    cv.stroke();
    cv.fillStyle='#8b93a7';cv.font='10px sans-serif';cv.fillText(hi.toFixed(1)+'m',4,12);cv.fillText(lo.toFixed(1)+'m',4,H-2);
  }
  if(window.__mode==='pose') drawSkeleton();
}
function drawSkeleton(){
  var b=window.QWOPBridge; if(!b||!b.getPose)return;
  var p; try{p=b.getPose();}catch(e){return;}
  if(!p||!p.torso)return;
  var cv=el('qwopSkeletonCanvas','canvas'); var ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  // 计算包围盒
  var xs=[],ys=[];for(var k in p){if(p[k]){xs.push(p[k].x);ys.push(p[k].y);}}
  if(!xs.length)return;
  var minx=Math.min.apply(null,xs),maxx=Math.max.apply(null,xs),miny=Math.min.apply(null,ys),maxy=Math.max.apply(null,ys);
  var pad=40; var bw=(maxx-minx)||1, bh=(maxy-miny)||1;
  var sc=Math.min((cv.width-2*pad)/bw,(cv.height-2*pad)/bh);
  function tx(x){return pad+(x-minx)*sc+(cv.width-2*pad-bw*sc)/2;}
  function ty(y){return cv.height-(pad+(y-miny)*sc+(cv.height-2*pad-bh*sc)/2);} // 翻转Y
  var edges=[['head','torso'],['torso','leftArm'],['leftArm','leftForearm'],
    ['torso','rightArm'],['rightArm','rightForearm'],
    ['torso','leftThigh'],['leftThigh','leftCalf'],['leftCalf','leftFoot'],
    ['torso','rightThigh'],['rightThigh','rightCalf'],['rightCalf','rightFoot']];
  ctx.strokeStyle='#4f8cff';ctx.lineWidth=4;ctx.lineCap='round';
  for(var i=0;i<edges.length;i++){var a=p[edges[i][0]],c=p[edges[i][1]];if(!a||!c)continue;
    ctx.beginPath();ctx.moveTo(tx(a.x),ty(a.y));ctx.lineTo(tx(c.x),ty(c.y));ctx.stroke();}
  for(var k in p){if(!p[k])continue;ctx.fillStyle='#ffb454';ctx.beginPath();ctx.arc(tx(p[k].x),ty(p[k].y),4,0,7);ctx.fill();}
  // 标签
  ctx.fillStyle='#8b93a7';ctx.font='11px sans-serif';ctx.fillText(distLabel(),10,18);
}
function distLabel(){var b=window.QWOPBridge;try{return '距离 '+b.getState().metres.toFixed(1)+' 米';}catch(e){return '';}}
initHUD(); setInterval(drawHUD,120);
"""


def pick_model(path: str | None) -> str:
    if path:
        return path
    final = os.path.join(MODEL_DIR, "ppo_qwop_final.zip")
    if os.path.exists(final):
        return final
    files = glob.glob(os.path.join(MODEL_DIR, "ppo_qwop_*_steps.zip"))
    if not files:
        raise SystemExit("没有找到模型, 先运行 train.py")
    return max(files, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--mode", type=str, default="game",
                        choices=["game", "pose"])
    args = parser.parse_args()

    model_path = pick_model(args.model)
    print(f"加载模型: {model_path}  模式: {args.mode}")

    env = DummyVecEnv([
        lambda: QWOPEnv(headless=False, frames_per_step=3, realtime=True)
    ])
    vn_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")
    if os.path.exists(vn_path):
        env = VecNormalize.load(vn_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(model_path)

    obs = env.reset()
    page = env.envs[0]._page
    page.evaluate(f"window.__mode='{args.mode}';")
    page.add_script_tag(content=OVERLAY_JS)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            info = infos[0]
            page.evaluate(
                "window.__r=__R__;window.__m=__M__;window.__a=__A__;"
                "window.__actName=__AN__;window.__crossed=__C__;window.__succ=__S__"
                .replace("__R__", f"{float(reward)}")
                .replace("__M__", f"{float(info['metres'])}")
                .replace("__A__", f"{int(action)}")
                .replace("__AN__", f"'{ACT_NAMES[int(action)]}'")
                .replace("__C__", f"{int(info.get('crossed_hurdle', False))}")
                .replace("__S__", f"{int(info.get('success', False))}")
            )
            done = bool(dones[0])
        tag = "到达终点!" if info.get("success") else ("摔倒" if info.get("fallen") else "超时")
        print(f"第 {ep + 1} 局: {info['metres']:.2f} 米 ({tag})")

    env.close()


if __name__ == "__main__":
    main()
