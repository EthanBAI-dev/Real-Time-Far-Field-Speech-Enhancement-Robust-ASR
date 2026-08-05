/** 文件实验台。 */

const el = (id) => document.getElementById(id);
const STOPS = [
  [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
  [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25],
  [252, 165, 10], [246, 215, 70], [252, 255, 164],
];

function inferno(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t));
  const f = t - i, a = STOPS[i], b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** 把 [列][频段] 的 dBFS 矩阵画成一幅静态频谱图。范围理由见 spectro.js。 */
function drawSpec(canvas, cols, dbMin = -90, dbMax = -20) {
  if (!cols?.length) return;
  const nBins = cols[0].length;
  const ctx = canvas.getContext("2d");
  canvas.width = cols.length;
  canvas.height = nBins;
  const img = ctx.createImageData(cols.length, nBins);
  const range = dbMax - dbMin;
  for (let x = 0; x < cols.length; x++) {
    for (let k = 0; k < nBins; k++) {
      const [r, g, b] = inferno((cols[x][k] - dbMin) / range);
      // 低频画在下方
      const o = ((nBins - 1 - k) * cols.length + x) * 4;
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function status(msg, cls = "") {
  const s = el("status");
  s.hidden = false;
  s.className = "note" + (cls ? " " + cls : "");
  s.innerHTML = msg;
}

el("run").onclick = async () => {
  const f = el("file").files[0];
  const fd = new FormData();
  if (f) fd.append("file", f);
  fd.append("noise_kind", el("noise").value);
  fd.append("snr_db", el("snr").value);
  fd.append("t60", el("t60").value);
  fd.append("methods", "none,specsub,wiener,mmse-lsa");
  fd.append("vad", "energy");

  el("run").disabled = true;
  status("处理中…（每种方法都走逐帧流式路径，与实时演示完全相同的代码）");

  try {
    const r = await fetch("/api/lab/run", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || d.error) { status(d.error || `请求失败 (${r.status})`, "bad"); return; }
    render(d);
    status(`完成 · ${d.config.duration_s}s 音频 · ${d.config.noise_kind} @ ${d.config.snr_db} dB` +
           (d.config.t60 > 0 ? ` · T60 ${d.config.t60}s` : " · 无混响"), "");
  } catch (e) {
    status(`请求出错：${e.message}`, "bad");
  } finally {
    el("run").disabled = false;
  }
};

function audioTag(b64) {
  return `<audio controls preload="none" style="height:30px;width:170px" src="data:audio/wav;base64,${b64}"></audio>`;
}

function render(d) {
  el("resultPanel").hidden = false;
  el("specPanel").hidden = false;

  const base = d.rows.find((r) => r.method === "none")?.metrics;
  const tb = el("tbl").querySelector("tbody");
  tb.innerHTML = "";

  for (const row of d.rows) {
    const m = row.metrics;
    const dS = base ? m.si_sdr - base.si_sdr : NaN;
    const dT = base ? m.stoi - base.stoi : NaN;
    const cls = (v) => (Number.isFinite(v) ? (v > 0.001 ? "up" : v < -0.001 ? "down" : "") : "");
    const sign = (v, d2) => (Number.isFinite(v) ? (v >= 0 ? "+" : "") + v.toFixed(d2) : "—");

    tb.insertAdjacentHTML("beforeend", `<tr>
      <td><b>${row.method === "none" ? "noisy（未处理）" : row.method}</b></td>
      <td class="num">${m.si_sdr.toFixed(2)}</td>
      <td class="num ${cls(dS)}">${row.method === "none" ? "—" : sign(dS, 2)}</td>
      <td class="num">${m.seg_snr.toFixed(2)}</td>
      <td class="num">${m.stoi.toFixed(3)}</td>
      <td class="num ${cls(dT)}">${row.method === "none" ? "—" : sign(dT, 3)}</td>
      <td class="num">${m.estoi.toFixed(3)}</td>
      <td class="num">${m.pesq ?? '<span class="muted">n/a</span>'}</td>
      <td class="num">${row.latency.rtf.toFixed(4)}</td>
      <td class="num">${row.latency.proc_p99_ms.toFixed(2)}</td>
      <td>${audioTag(row.audio)}</td>
    </tr>`);
  }

  el("readNote").innerHTML = `
    <b>怎么读</b>：ΔSI-SDR 与 ΔSTOI 经常<b>方向相反</b> ——
    单通道降噪能明显改善信噪比与听感，却几乎不改善、甚至损害可懂度。
    这不是 bug，是单通道降噪的经典结论。
    <b>所以最终必须直接测 ASR 字错率，而不是拿语音质量指标去代理它。</b>
    ${d.pesq_available ? "" : "PESQ 一列为 n/a：该包在 Windows 上需 MSVC 编译，已改由 Colab 侧计算（见 ISSUES.md I-05）。"}
  `;

  const wrap = el("specs");
  wrap.innerHTML = "";
  const items = [
    ["干净参考", d.reference],
    [`带噪输入（${d.config.noise_kind} @ ${d.config.snr_db} dB）`, d.noisy],
    ...d.rows.filter((r) => r.method !== "none").map((r) => [r.method, r]),
  ];
  for (const [label, obj] of items) {
    wrap.insertAdjacentHTML("beforeend",
      `<div class="specwrap"><span class="tag">${label}</span>
       <canvas style="height:150px;image-rendering:pixelated"></canvas></div>`);
    drawSpec(wrap.lastElementChild.querySelector("canvas"), obj.spec);
  }
}
