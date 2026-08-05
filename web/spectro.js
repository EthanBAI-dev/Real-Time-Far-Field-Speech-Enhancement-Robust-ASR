/**
 * 滚动频谱图渲染器 + 共用小工具。
 *
 * 渲染策略：每来一批帧，先把画布内容整体左移 N*COL_W 像素（drawImage 自拷贝），
 * 再在右边缘画新列。比"每帧重绘整幅图"快一个数量级，
 * 因为浏览器的 canvas 自拷贝走的是 GPU blit。
 */

const COL_W = 2; // 每帧占的像素宽度

/** Inferno 色图的分段线性近似。深色背景下动态范围看起来最舒服。 */
const STOPS = [
  [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
  [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25],
  [252, 165, 10], [246, 215, 70], [252, 255, 164],
];

function inferno(t) {
  t = Math.max(0, Math.min(1, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t));
  const f = t - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [
    (a[0] + (b[0] - a[0]) * f) | 0,
    (a[1] + (b[1] - a[1]) * f) | 0,
    (a[2] + (b[2] - a[2]) * f) | 0,
  ];
}

export class Spectrogram {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{dbMin?: number, dbMax?: number}} opts
   *   色标范围，单位 **dBFS**（满幅正弦 = 0 dBFS，由服务端的 magnitude_db 定义）。
   *   默认 -90~-20 是实测定出来的：正常音量语音的频点分布 p10≈-90、p99≈-30。
   *   这个范围必须与服务端的归一化配套 —— 早期版本按"时域在 ±1 之间所以 dB 应该
   *   接近 0"来猜范围，结果整幅图压在色标顶端，全是均匀亮黄色（见 ISSUES.md I-12）。
   */
  constructor(canvas, { dbMin = -90, dbMax = -20 } = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.dbMin = dbMin;
    this.dbMax = dbMax;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.cv.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.cv.clientHeight * dpr));
    if (w === this.cv.width && h === this.cv.height) return;
    // 保留已有内容：改尺寸会清空画布，不救一下滚动历史就没了
    const old = this.cv.width ? this.ctx.getImageData(0, 0, this.cv.width, this.cv.height) : null;
    this.cv.width = w;
    this.cv.height = h;
    this.ctx.fillStyle = "#05070c";
    this.ctx.fillRect(0, 0, w, h);
    if (old) this.ctx.putImageData(old, 0, 0);
    this.colW = Math.max(1, Math.round(COL_W * dpr));
  }

  clear() {
    this.ctx.fillStyle = "#05070c";
    this.ctx.fillRect(0, 0, this.cv.width, this.cv.height);
  }

  /**
   * 追加若干列。
   * @param {number[][]} cols 每列是一组 dB 值，索引 0 = 最低频
   */
  push(cols) {
    if (!cols?.length) return;
    const { ctx, cv } = this;
    const dx = this.colW * cols.length;

    ctx.drawImage(cv, -dx, 0);

    const range = this.dbMax - this.dbMin;
    for (let c = 0; c < cols.length; c++) {
      const col = cols[c];
      const x = cv.width - dx + c * this.colW;
      const bandH = cv.height / col.length;
      for (let k = 0; k < col.length; k++) {
        const [r, g, b] = inferno((col[k] - this.dbMin) / range);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        // 低频画在下方：与常规频谱图的方向一致
        ctx.fillRect(x, cv.height - (k + 1) * bandH, this.colW, Math.ceil(bandH) + 1);
      }
    }
  }
}

/** VAD 判决 / 概率的滚动时间轴。 */
export class VadStrip {
  constructor(canvas) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.cv.width = Math.max(1, Math.round(this.cv.clientWidth * dpr));
    this.cv.height = Math.max(1, Math.round(this.cv.clientHeight * dpr));
    this.colW = Math.max(1, Math.round(COL_W * dpr));
    this.ctx.fillStyle = "#05070c";
    this.ctx.fillRect(0, 0, this.cv.width, this.cv.height);
  }

  clear() {
    this.ctx.fillStyle = "#05070c";
    this.ctx.fillRect(0, 0, this.cv.width, this.cv.height);
  }

  push(probs, speech) {
    if (!probs?.length) return;
    const { ctx, cv } = this;
    const dx = this.colW * probs.length;
    ctx.drawImage(cv, -dx, 0);
    for (let i = 0; i < probs.length; i++) {
      const x = cv.width - dx + i * this.colW;
      ctx.fillStyle = "#05070c";
      ctx.fillRect(x, 0, this.colW, cv.height);
      // 概率画成柱高，硬判决画成底部色条 —— 两者不一致的地方
      // 正好是迟滞/挂起在起作用，界面上能直接看出来
      const h = Math.max(1, probs[i] * cv.height);
      ctx.fillStyle = speech[i] ? "#3ddc97" : "#39415a";
      ctx.fillRect(x, cv.height - h, this.colW, h);
      if (speech[i]) {
        ctx.fillStyle = "#4ea1ff";
        ctx.fillRect(x, cv.height - 3, this.colW, 3);
      }
    }
  }
}

/** 更新一个 .stat 数码管。 */
export function setStat(id, value, cls) {
  const el = document.getElementById(id);
  if (!el) return;
  el.querySelector(".v").firstChild.nodeValue = value;
  const box = el.closest(".stat");
  if (box && cls !== undefined) box.className = "stat" + (cls ? " " + cls : "");
}

export function fmt(x, d = 1) {
  return Number.isFinite(x) ? x.toFixed(d) : "—";
}
