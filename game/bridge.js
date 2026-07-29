/*
 * QWOP RL bridge
 * 必须在 QWOP.min.js 之前加载。
 * 功能：
 *  1. 劫持 requestAnimationFrame + performance.now，实现"手动步进"模式，
 *     训练时以 CPU 极限速度快进物理（游戏固定 30fps 时间步，确定性好）。
 *  2. 读取 12 个刚体的物理状态（位置/角度/速度/角速度）。
 *  3. 直接调用游戏的 oninputdown/oninputup 控制 Q/W/O/P。
 *  4. 调用游戏内置 reset() 重开。
 */
(function () {
  var realRAF = window.requestAnimationFrame.bind(window);
  var realNow = performance.now.bind(performance);

  // 屏蔽页面可见性事件：headless/切后台时引擎会收到"窗口关闭"事件并自毁
  var BLOCKED = {
    visibilitychange: 1,
    mozvisibilitychange: 1,
    msvisibilitychange: 1,
    webkitvisibilitychange: 1,
    pagehide: 1,
    blur: 1,
  };
  var docAdd = document.addEventListener.bind(document);
  document.addEventListener = function (type, fn, opts) {
    if (BLOCKED[type]) return;
    return docAdd(type, fn, opts);
  };
  try {
    Object.defineProperty(Document.prototype, "hidden", { get: function () { return false; } });
    Object.defineProperty(Document.prototype, "visibilityState", { get: function () { return "visible"; } });
  } catch (e) {}

  var B = (window.QWOPBridge = {
    manual: false,
    vt: 0, // 虚拟时钟（毫秒）
    cb: null, // 游戏挂起的 RAF 回调
    frameMs: 1000 / 30,
    _prevAngles: null,
  });

  performance.now = function () {
    return B.manual ? B.vt : realNow();
  };

  window.requestAnimationFrame = function (cb) {
    if (B.manual) {
      B.cb = cb;
      return 0;
    }
    return realRAF(cb);
  };

  var PARTS = [
    "torso",
    "head",
    "leftArm",
    "leftForearm",
    "leftThigh",
    "leftCalf",
    "leftFoot",
    "rightArm",
    "rightForearm",
    "rightThigh",
    "rightCalf",
    "rightFoot",
  ];

  function game() {
    return window.QWOPGame;
  }

  function body(part) {
    return part._components.get("physicsBody", false);
  }

  B.ready = function () {
    var g = game();
    return !!(g && g.doneLoading && g.torso);
  };

  // 进入手动步进模式（虚拟时钟从当前真实时间无缝衔接）
  B.startManual = function () {
    if (B.manual) return;
    B.vt = realNow();
    B.manual = true;
  };

  // 快进 n 帧（每帧 = 1/30 秒的固定物理步）
  // 为提速, 只在最后一帧渲染（skipRender 由 min.js 补丁读取）
  B.step = function (n) {
    var done = 0;
    for (var i = 0; i < n; i++) {
      B.skipRender = i < n - 1;
      B.vt += B.frameMs;
      var cb = B.cb;
      B.cb = null;
      if (!cb) break;
      cb(B.vt);
      done++;
    }
    B.skipRender = false;
    return done;
  };

  B.setKeys = function (q, w, o, p) {
    var g = game();
    var want = { Q: q, W: w, O: o, P: p };
    var cur = { Q: g.QDown, W: g.WDown, O: g.ODown, P: g.PDown };
    for (var k in want) {
      if (want[k] && !cur[k]) g.oninputdown(k);
      if (!want[k] && cur[k]) g.oninputup(k);
    }
  };

  B.reset = function () {
    var g = game();
    B.setKeys(false, false, false, false);
    g.reset();
    B._prevAngles = null;
    // 跑几帧让重建后的物理世界稳定
    if (B.manual) B.step(3);
  };

  // 读取状态：每个刚体 7 维 + 4 个按键 = 88 维
  B.getState = function () {
    var g = game();
    var tb = body(g.torso);
    var tc = tb.getWorldCenter();
    var obs = [];
    var angles = [];
    var dt = 1 / 30;
    for (var i = 0; i < PARTS.length; i++) {
      var b = body(g[PARTS[i]]);
      var c = b.getWorldCenter();
      var v = b.getLinearVelocity();
      var a = b.getAngle();
      angles.push(a);
      var av = 0;
      if (B._prevAngles) av = (a - B._prevAngles[i]) / dt;
      obs.push(
        (c.x - tc.x) / 10,
        c.y / 10,
        Math.sin(a),
        Math.cos(a),
        v.x / 10,
        v.y / 10,
        av / 10
      );
    }
    B._prevAngles = angles;
    obs.push(g.QDown ? 1 : 0, g.WDown ? 1 : 0, g.ODown ? 1 : 0, g.PDown ? 1 : 0);

    // 跨栏世界坐标（用于跨栏奖励判定）
    var hb = g.hurdleBase;
    var hbp = hb && hb._components ? hb._components.get("physicsBody", false) : null;
    var hurdleX = hbp ? hbp.getWorldCenter().x : null;
    var headC = body(g.head).getWorldCenter();

    return {
      obs: obs,
      x: tc.x,
      metres: tc.x / 10,
      score: g.score,
      torsoY: tc.y,
      torsoAngle: tb.getAngle(),
      headY: headC.y,
      hasHurdle: !!hbp,
      hurdleX: hurdleX,
      gameOver: !!g.gameOver,
      gameEnded: !!g.gameEnded,
      fallen: !!g.fallen,
      jumped: !!g.jumped,
      jumpLanded: !!g.jumpLanded,
    };
  };

  // 人形骨架：返回 12 刚体的世界中心（供姿态可视化）
  B.PARTS = PARTS;
  B.getPose = function () {
    var g = game();
    var pts = {};
    for (var i = 0; i < PARTS.length; i++) {
      var b = body(g[PARTS[i]]);
      if (!b) { pts[PARTS[i]] = null; continue; }
      var c = b.getWorldCenter();
      pts[PARTS[i]] = { x: c.x, y: c.y, a: b.getAngle() };
    }
    return pts;
  };

  // 一次往返完成 动作 -> 快进 -> 读状态，减少 evaluate 次数
  B.act = function (q, w, o, p, frames) {
    B.setKeys(q, w, o, p);
    B.step(frames);
    return B.getState();
  };
})();
