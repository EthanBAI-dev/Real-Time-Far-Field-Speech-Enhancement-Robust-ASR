/**
 * 麦克风采集 AudioWorklet。
 *
 * 为什么用 AudioWorklet 而不是已废弃的 ScriptProcessorNode：
 * ScriptProcessor 在**主线程**上跑回调，页面一做重绘（我们每帧都在画频谱图）
 * 就会丢音频块，表现为周期性的爆音和卡顿。AudioWorklet 跑在独立的音频渲染线程上，
 * 与 UI 渲染完全解耦。
 *
 * Worklet 每次固定收到 128 个样本。直接把 128 个样本发一次 postMessage
 * 会是 125 次/秒的跨线程消息 + WebSocket 帧，开销不划算；
 * 这里攒够 CHUNK 个样本再发一次，在延迟和开销之间取折中。
 */

const CHUNK = 2048; // 128 ms @ 16 kHz

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(CHUNK);
    this._n = 0;
  }

  process(inputs) {
    const ch = inputs[0]?.[0];
    if (!ch) return true; // 输入未就绪时保持存活，不要返回 false（那会永久关闭节点）

    for (let i = 0; i < ch.length; i++) {
      this._buf[this._n++] = ch[i];
      if (this._n === CHUNK) {
        // 转移所有权（transferable）而不是拷贝，避免每 128 ms 产生一次 GC 压力
        const out = this._buf.slice();
        this.port.postMessage(out, [out.buffer]);
        this._n = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
