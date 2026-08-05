/**
 * 实时演示。两种音频来源，走**完全相同**的下游路径：
 *
 *   麦克风   → AudioWorklet(16k) ─┐
 *   虚拟麦克风 → 文件按实时节奏切块 ─┴→ WS 二进制 → 服务端逐帧管线
 *                                    → WS 文本(指标/频谱) + WS 二进制(增强音频)
 *                                    → 画布 + 可选回放
 *
 * 虚拟麦克风存在的理由：本机没有可用的麦克风端点（docs/ISSUES.md I-16），
 * 而且作品集演示时也未必总有麦克风。它按真实时间节奏（每 128 ms 一块）推送，
 * 所以 RTF、p95/p99 延迟这些指标测出来与真麦克风是同一回事 ——
 * 它不是"离线处理"，是把数据源换掉了而已。
 *
 * 采样率的坑：AudioContext 构造时可以请求 16000 Hz，但**浏览器不保证给你**。
 * 拿到后必须回读 ctx.sampleRate 校验，不一致就明确报错，
 * 而不是让服务端收到 48 kHz 却当成 16 kHz 处理（频谱会整体拉伸 3 倍）。
 */

import { Spectrogram, VadStrip, setStat, fmt } from "/static/spectro.js";

const TARGET_SR = 16000;
const CHUNK = 2048; // 128 ms @ 16 kHz，与 capture-worklet.js 一致

const el = (id) => document.getElementById(id);
const banner = el("banner");

let ctx = null, node = null, stream = null, ws = null;
let virtualTimer = null, virtualData = null, virtualPos = 0;
let playhead = 0, monitoring = false;

const specNoisy = new Spectrogram(el("cvNoisy"));
const specEnh = new Spectrogram(el("cvEnh"));
const vadStrip = new VadStrip(el("cvVad"));

function note(msg, cls = "") {
  banner.className = "note" + (cls ? " " + cls : "");
  banner.innerHTML = msg;
}

// ------------------------------------------------------------------ 初始化

fetch("/api/info")
  .then((r) => r.json())
  .then((d) => {
    setStat("stBudget", fmt(d.frame_ms, 1));
    setStat("stAlgo", fmt(d.algo_latency_ms, 0));
    const sel = el("selMethod");
    if (d.nn_ready) {
      for (const m of d.nn_models) {
        const o = document.createElement("option");
        o.value = m;
        o.textContent = `${m} · 神经网络`;
        sel.appendChild(o);
      }
    } else {
      const o = document.createElement("option");
      o.value = "";
      o.disabled = true;
      o.textContent = "神经模型 · 待 Colab 训练产出";
      sel.appendChild(o);
    }
  })
  .catch(() => note("无法连接服务端 /api/info", "bad"));

/**
 * 开场就探测有没有音频输入设备，而不是等用户点了「开始」才失败。
 *
 * 注意：未授权时 enumerateDevices() 返回的条目 label 为空，但**条目本身存在**。
 * 所以能靠 kind==='audioinput' 的数量判断设备有无，不需要先申请权限。
 */
async function probeDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return { ok: false, n: 0, reason: "浏览器不支持 enumerateDevices" };
  }
  try {
    const list = await navigator.mediaDevices.enumerateDevices();
    const n = list.filter((d) => d.kind === "audioinput").length;
    return { ok: n > 0, n, reason: "" };
  } catch (e) {
    return { ok: false, n: 0, reason: e.message };
  }
}

probeDevices().then(({ ok, n }) => {
  if (ok) {
    note(`检测到 ${n} 个音频输入设备。点「开始」授权麦克风，或切到「虚拟麦克风」用文件演示。`);
  } else {
    // 没有麦克风是完全可以工作的状态，直接切到虚拟麦克风并说清楚原因
    el("selSource").value = "virtual";
    el("selSource").dispatchEvent(new Event("change"));
    note(
      `<b>系统里没有可用的音频输入设备</b>，已自动切换到「虚拟麦克风」。<br>` +
        `它把音频文件按<b>真实时间节奏</b>喂进完全相同的处理路径，` +
        `所以 RTF 与延迟指标测出来与真麦克风是同一回事。<br>` +
        `<span class="muted">想用真麦克风：插上麦克风或耳机麦，然后刷新页面。</span>`,
      "warn"
    );
  }
});

// ------------------------------------------------------------------ 启停

el("btnStart").onclick = start;
el("btnStop").onclick = stop;

async function start() {
  el("btnStart").disabled = true;
  specNoisy.clear();
  specEnh.clear();
  vadStrip.clear();
  resetLevels();

  const ok = el("selSource").value === "virtual" ? await startVirtual() : await startMic();
  if (!ok) {
    el("btnStart").disabled = false;
    return;
  }
  el("btnStop").disabled = false;
}

// ---------------------------------------------------------- 来源 A：真麦克风

async function startMic() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // 用 ideal 而不是精确值。把 16000 写成必须满足的约束时，
        // 若没有设备支持该采样率，浏览器会直接抛错而不是让 AudioContext 去重采样。
        sampleRate: { ideal: TARGET_SR },
        // 这三项必须**全部关掉**：浏览器自带的降噪/AEC/AGC 会先把音频处理一遍，
        // 那样评测的就是"Chrome 的降噪 + 我们的降噪"，频谱图上也看不到真实噪声。
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (e) {
    note(await diagnoseMicError(e), "bad");
    return false;
  }

  ctx = new AudioContext({ sampleRate: TARGET_SR });
  if (Math.abs(ctx.sampleRate - TARGET_SR) > 1) {
    note(
      `浏览器拒绝了 ${TARGET_SR} Hz 的请求，实际为 ${ctx.sampleRate} Hz。<br>` +
        `服务端管线按 16 kHz 设计，继续下去所有频率都会错位。` +
        `请在系统声音设置里把麦克风采样率改为 16000 Hz，或改用「虚拟麦克风」。`,
      "bad"
    );
    await cleanup();
    return false;
  }

  await ctx.audioWorklet.addModule("/static/capture-worklet.js");
  node = new AudioWorkletNode(ctx, "capture", { numberOfInputs: 1, numberOfOutputs: 0 });
  ctx.createMediaStreamSource(stream).connect(node);

  if (!(await openSocket())) return false;
  node.port.onmessage = (ev) => {
    if (ws?.readyState === WebSocket.OPEN) ws.send(ev.data);
  };

  playhead = ctx.currentTime + 0.1;
  note("正在采集。上下两幅频谱图是处理前/后，切换方法可以直接对比。", "");
  return true;
}

/**
 * 把 getUserMedia 的错误映射成**准确**的诊断。
 *
 * 早先这里不分错误类型，一律提示"只在 localhost 或 HTTPS 下可用"——
 * 这句话对 NotFoundError 是完全错误的引导，会让人去折腾根本没问题的证书和端口。
 * DOMException 的 name 已经把原因说清楚了，照实翻译即可。
 */
async function diagnoseMicError(e) {
  const { n } = await probeDevices();
  const tail =
    `<br><span class="muted">不想折腾的话，把「音频来源」切到<b>虚拟麦克风</b>` +
    `就能立刻看到完整演示 —— 走的是同一条处理路径。</span>`;

  switch (e.name) {
    case "NotFoundError":
    case "OverconstrainedError":
      return (
        `<b>没有找到可用的麦克风</b>（${e.name}）。当前系统报告的音频输入设备数：<b>${n}</b>。<br>` +
        `这<b>不是</b>权限或 HTTPS 的问题 —— 是系统里根本没有可用的录音设备。<br>` +
        `常见原因：台式机没插麦克风；耳机麦没插到位；设备在「声音设置 → 输入」里被禁用。` +
        tail
      );
    case "NotAllowedError":
    case "SecurityError":
      return (
        `<b>麦克风权限被拒绝</b>（${e.name}）。<br>` +
        `点地址栏左侧的图标重新允许；或确认是用 <code>http://127.0.0.1:8000</code> 打开的 ——` +
        `浏览器只在 localhost 或 HTTPS 下允许访问麦克风。` +
        tail
      );
    case "NotReadableError":
      return (
        `<b>麦克风被占用或硬件出错</b>（${e.name}）。<br>` +
        `其他程序（会议软件、录音软件）可能正独占该设备，关掉后重试。` + tail
      );
    default:
      return `<b>麦克风打开失败</b>：${e.name} — ${e.message}${tail}`;
  }
}

// ------------------------------------------------------- 来源 B：虚拟麦克风

async function startVirtual() {
  const q = new URLSearchParams({
    noise: el("selVnoise").value,
    snr_db: el("selVsnr").value,
    t60: el("selVt60").value,
    seconds: "20",
  });
  note("正在合成虚拟输入…");

  let buf;
  try {
    const r = await fetch(`/api/demo/mixed?${q}`);
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      note(
        `合成失败：${j.error || r.status}。<br>` +
          `请确认项目的 <code>data/demo/</code> 下有音频文件。`,
        "bad"
      );
      return false;
    }
    const ab = await r.arrayBuffer();
    ctx = new AudioContext({ sampleRate: TARGET_SR });
    buf = await ctx.decodeAudioData(ab);
  } catch (e) {
    note(`虚拟输入准备失败：${e.message}`, "bad");
    await cleanup();
    return false;
  }

  virtualData = buf.getChannelData(0);
  virtualPos = 0;

  if (!(await openSocket())) return false;
  playhead = ctx.currentTime + 0.1;

  // 按真实时间节奏推送：每 CHUNK/16000 秒发一块。
  // **不能一次性全发** —— 那样服务端会在几十毫秒内处理完 20 秒音频，
  // 测出来的 RTF 和延迟就变成了批处理吞吐量，与实时部署毫无关系。
  const periodMs = (CHUNK / TARGET_SR) * 1000;
  virtualTimer = setInterval(() => {
    if (ws?.readyState !== WebSocket.OPEN) return;
    if (virtualPos >= virtualData.length) {
      virtualPos = 0; // 循环播放，方便边看边切方法
    }
    const end = Math.min(virtualPos + CHUNK, virtualData.length);
    const slice = new Float32Array(CHUNK);
    slice.set(virtualData.subarray(virtualPos, end));
    virtualPos = end;
    ws.send(slice.buffer);
  }, periodMs);

  const desc = el("selVnoise").value === "none"
    ? "无噪声"
    : `${el("selVnoise").value} @ ${el("selVsnr").value} dB`;
  note(
    `虚拟麦克风运行中（${desc}${el("selVt60").value > 0 ? ` · T60 ${el("selVt60").value}s` : ""}，循环播放）。<br>` +
      `按真实时间节奏推送，因此 RTF 与延迟指标与真麦克风等价。` +
      `切换降噪方法可以直接对比两幅频谱图。`,
    ""
  );
  return true;
}

// ------------------------------------------------------------------ 停止

async function stop() {
  await cleanup();
  el("btnStart").disabled = false;
  el("btnStop").disabled = true;
  note("已停止。", "");
}

async function cleanup() {
  if (virtualTimer) { clearInterval(virtualTimer); virtualTimer = null; }
  virtualData = null;
  ws?.close(); ws = null;
  node?.disconnect(); node = null;
  stream?.getTracks().forEach((t) => t.stop()); stream = null;
  if (ctx) { await ctx.close(); ctx = null; }
}

// ------------------------------------------------------------------ WebSocket

function openSocket() {
  return new Promise((resolve) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { pushConfig(); resolve(true); };
    ws.onerror = () => { note("WebSocket 连接失败，服务端可能没在跑。", "bad"); resolve(false); };
    ws.onclose = () => { if (!el("btnStop").disabled) note("连接已断开。", "warn"); };
    ws.onmessage = onMessage;
  });
}

function onMessage(ev) {
  if (typeof ev.data === "string") {
    const m = JSON.parse(ev.data);
    if (m.op === "error") { note(`服务端错误：${m.msg}`, "bad"); return; }
    if (m.op) return; // config_ack 等控制回执
    render(m);
    return;
  }
  if (monitoring && ctx) playback(new Float32Array(ev.data));
}

/**
 * 电平表的时间平滑状态。
 *
 * 服务端给的 in_db / out_db 是**每 128 ms 一块的瞬时 RMS**。直接显示的话，
 * 数字会随当前这一块恰好是语音还是静音剧烈摆动 —— 同一个方法，
 * 读数可能一会儿 0.1 dB 一会儿 16 dB，看起来像坏了。
 * 真实的音频电平表都带时间常数，这里同样处理：约 1.5 秒的一阶平滑。
 *
 * 注意平滑要在 **dB 域**做。在线性域平滑再取对数的话，
 * 少数几个高能量块会主导结果，静音段的信息几乎被抹掉。
 */
// 每块 128 ms，时间常数 = 128ms / (1 - TAU)。0.92 → 约 1.6 s。
// 别调到 0.98（≈6.4 s）：读数是稳，但切换方法后要等六七秒才爬到新值，
// 用起来像"切了没反应"。1.6 s 足够压掉语音/静音的摆动，又跟得上手动切换。
const LEVEL_TAU = 0.92;
let smIn = null, smOut = null;

function resetLevels() { smIn = smOut = null; }

function render(m) {
  specNoisy.push(m.noisy_spec);
  specEnh.push(m.enh_spec);
  vadStrip.push(m.vad, m.speech);

  setStat("stRtf", fmt(m.rtf, 3), m.rtf < 0.5 ? "good" : m.rtf < 1 ? "warn" : "bad");
  setStat("stP95", fmt(m.p95_ms, 2), m.p95_ms < m.budget_ms ? "good" : "warn");
  // p99 才是实时性的判据：它一旦越过帧预算，缓冲就会持续累积
  setStat("stP99", fmt(m.p99_ms, 2), m.p99_ms < m.budget_ms ? "good" : "bad");

  // 静音块（-100 dB 那种）不参与平滑，否则一段静音就能把读数拖到底
  if (m.in_db > -80) {
    smIn = smIn === null ? m.in_db : LEVEL_TAU * smIn + (1 - LEVEL_TAU) * m.in_db;
    smOut = smOut === null ? m.out_db : LEVEL_TAU * smOut + (1 - LEVEL_TAU) * m.out_db;
  }
  if (smIn === null) return;

  setStat("stIn", fmt(smIn, 0));
  setStat("stOut", fmt(smOut, 0));
  const nr = smIn - smOut;
  setStat("stNr", (nr >= 0 ? "" : "+") + fmt(nr, 1), nr > 1 ? "good" : "");
}

// ------------------------------------------------------------------ 回放

function playback(pcm) {
  if (!pcm.length) return;
  const buf = ctx.createBuffer(1, pcm.length, TARGET_SR);
  buf.copyToChannel(pcm, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  // 排队播放：块之间必须首尾相接。直接用 currentTime 播会因为网络抖动
  // 在块间产生空隙或重叠，听起来是持续的咔哒声。
  const now = ctx.currentTime;
  if (playhead < now + 0.02) playhead = now + 0.06; // 落后太多就重新对齐
  src.start(playhead);
  playhead += buf.duration;
}

// ------------------------------------------------------------------ 配置

function pushConfig() {
  if (ws?.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    op: "config",
    method: el("selMethod").value || "none",
    vad: el("selVad").value,
    vad_gate: el("chkGate").checked,
  }));
}

for (const id of ["selMethod", "selVad", "chkGate"]) {
  // 切换方法后电平会阶跃变化，平滑状态必须一起重置，
  // 否则新方法的读数要一两秒才从旧值爬过来，看着像"切了没反应"
  el(id).onchange = () => { resetLevels(); pushConfig(); };
}

el("selSource").onchange = (e) => {
  el("ctlVirtual").hidden = e.target.value !== "virtual";
};

el("chkMonitor").onchange = (e) => {
  monitoring = e.target.checked;
  el("monitorHint").hidden = !monitoring;
  if (ctx) playhead = ctx.currentTime + 0.1;
};

addEventListener("beforeunload", () => { ws?.close(); });
