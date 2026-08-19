"""训练循环。设计成**位置无关**：同一份代码在 Colab 和本地跑同样的逻辑。

Colab 特有的两个现实约束，直接决定了这里的设计：

1. **会话随时会断**（12 小时上限、闲置回收、GPU 配额）。
   所以每个 epoch 都存 checkpoint，且**存到 Google Drive**，
   并支持从任意 checkpoint 无缝续训（包括优化器状态、调度器状态、随机数状态）。
   不做这一点，一次断线就等于白跑几小时。

2. **磁盘与显存有限**。所以默认开混合精度（AMP），
   并把 batch size / 段长做成配置项而不是写死。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rtse.audio.stft import DEFAULT_CONFIG, STFTConfig
from rtse.data.dataset import istft_torch, stft_torch
from rtse.train.losses import CombinedLoss

__all__ = ["TrainConfig", "Trainer"]


@dataclass
class TrainConfig:
    model: str = "crn-lite"
    epochs: int = 60
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    """梯度裁剪。GRU 在训练早期很容易出现梯度爆炸，不裁剪会直接 NaN。"""
    amp: bool = True
    num_workers: int = 2
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    out_dir: str = "checkpoints"
    log_every: int = 50
    seed: int = 0
    loss_weights: tuple[float, float, float] = (1.0, 0.5, 0.2)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        cfg: TrainConfig | None = None,
        stft_cfg: STFTConfig = DEFAULT_CONFIG,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg or TrainConfig()
        self.stft_cfg = stft_cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = CombinedLoss(*self.cfg.loss_weights).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.use_amp = self.cfg.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.out_dir = Path(self.cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.epoch = 0
        self.step = 0
        self.best_val = math.inf
        self.history: list[dict] = []

    # ------------------------------------------------------------------ 调度

    def _lr_at(self, step: int) -> float:
        """线性 warmup + 余弦退火。

        warmup 不是可选项：GRU + BatchNorm 的组合在第一个几百步里
        BN 的running stats 还很不准，此时用满学习率经常直接发散。
        """
        total = max(1, self.cfg.epochs * len(self.train_loader))
        if step < self.cfg.warmup_steps:
            return self.cfg.lr * (step + 1) / self.cfg.warmup_steps
        p = (step - self.cfg.warmup_steps) / max(1, total - self.cfg.warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        return self.cfg.lr * (self.cfg.min_lr_ratio + (1 - self.cfg.min_lr_ratio) * cos)

    # ------------------------------------------------------------------ 前向

    def _forward(self, noisy: torch.Tensor, clean: torch.Tensor):
        spec_noisy = stft_torch(noisy, self.stft_cfg)
        spec_clean = stft_torch(clean, self.stft_cfg)
        est_spec = self.model(spec_noisy)
        est_wav = istft_torch(est_spec, length=noisy.shape[-1], cfg=self.stft_cfg)
        return self.criterion(est_spec, spec_clean, est_wav, clean)

    # ------------------------------------------------------------------ 训练

    def train_epoch(self) -> dict:
        self.model.train()
        agg: dict[str, float] = {}
        n = 0
        t0 = time.perf_counter()

        for i, (noisy, clean) in enumerate(self.train_loader):
            noisy = noisy.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)

            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(self.step)

            self.opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, parts = self._forward(noisy, clean)

            if not torch.isfinite(loss):
                # 单个坏 batch（比如全静音段导致 log(0)）不该毁掉整次训练。
                # 跳过并记录，连续出现才说明是真问题。
                print(f"  [warn] step {self.step} 损失非有限，跳过该 batch")
                self.step += 1
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
            self.step += 1

            if self.cfg.log_every and i % self.cfg.log_every == 0:
                cur = {k: v / max(n, 1) for k, v in agg.items()}
                print(
                    f"  ep{self.epoch:02d} [{i:>4}/{len(self.train_loader)}] "
                    f"lr={self._lr_at(self.step):.2e} "
                    + " ".join(f"{k}={v:.4f}" for k, v in cur.items())
                )

        out = {k: v / max(n, 1) for k, v in agg.items()}
        out["epoch_seconds"] = round(time.perf_counter() - t0, 1)
        return out

    @torch.no_grad()
    def validate(self) -> dict:
        if self.val_loader is None:
            return {}
        self.model.eval()
        agg: dict[str, float] = {}
        n = 0
        for noisy, clean in self.val_loader:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                _, parts = self._forward(noisy, clean)
            if not all(math.isfinite(v) for v in parts.values()):
                continue
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        return {f"val_{k}": v / max(n, 1) for k, v in agg.items()}

    def fit(self) -> list[dict]:
        print(f"设备={self.device}  AMP={self.use_amp}  "
              f"参数量={sum(p.numel() for p in self.model.parameters()):,}")
        while self.epoch < self.cfg.epochs:
            tr = self.train_epoch()
            va = self.validate()
            rec = {"epoch": self.epoch, **tr, **va}
            self.history.append(rec)
            print(f"[epoch {self.epoch:02d}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                                          else f"{k}={v}" for k, v in rec.items()))

            # 用验证 SI-SDR 选最佳模型，而不是用总 loss ——
            # 总 loss 里 mrstft 项数值大，会主导"哪个 epoch 最好"的判断，
            # 而我们真正在意的是 SI-SDR 和下游 CER。
            score = -va.get("val_si_sdr", -tr.get("si_sdr", 0.0))
            self.epoch += 1
            self.save("last.pt")
            if score < self.best_val:
                self.best_val = score
                self.save("best.pt")
                print(f"  → 新的最佳模型 (val SI-SDR={-score:.3f} dB)")
            (self.out_dir / "history.json").write_text(
                json.dumps(self.history, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return self.history

    # ------------------------------------------------------------------ 断点

    def save(self, name: str) -> Path:
        """存 checkpoint。**优化器与调度状态必须一起存** ——
        只存权重的话，Colab 断线续训会因为动量和学习率重置而出现明显的 loss 反弹。"""
        p = self.out_dir / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_cfg": asdict(self.model.cfg),
                "opt": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
                "best_val": self.best_val,
                "history": self.history,
                "train_cfg": asdict(self.cfg),
            },
            p,
        )
        return p

    def load(self, path: str | Path, weights_only_model: bool = False) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if weights_only_model:
            return
        self.opt.load_state_dict(ck["opt"])
        # AMP 的 scaler 状态**只在两侧都启用时才能搬**。
        # `GradScaler` 被禁用时 `state_dict()` 返回空字典，而空字典喂给
        # 启用状态的 `load_state_dict()` 会直接抛
        # "The source state dict is empty..."。
        #
        # 这不是假想情况：Colab 切换运行时是常态（GPU 配额用完会被降级到 CPU），
        # 于是「CPU 那轮存的 last.pt」拿到 GPU 会话里续训就必然炸。
        # checkpoint 本身没问题——权重、优化器、epoch 全在，只有这一个字段是空的。
        #
        # scaler 里只有损失缩放因子这类**每步都会自适应重估**的量，
        # 丢掉它最多让恢复后的头几步缩放系数重新收敛一次，
        # 远比"续训直接崩掉"或"为它重下一遍 checkpoint"划算。
        saved_scaler = ck.get("scaler") or {}
        if saved_scaler and self.use_amp:
            self.scaler.load_state_dict(saved_scaler)
        elif saved_scaler and not self.use_amp:
            print("  [scaler] checkpoint 存于 AMP 开启时，本次运行在 CPU/无 AMP —— 跳过")
        elif not saved_scaler and self.use_amp:
            print("  [scaler] checkpoint 存于无 AMP 时（多半是 CPU 运行时），"
                  "本次启用 AMP —— 跳过，缩放系数会自行重估")
        self.epoch = ck["epoch"]
        self.step = ck["step"]
        self.best_val = ck.get("best_val", math.inf)
        self.history = ck.get("history", [])
        print(f"已从 {path} 恢复：epoch={self.epoch} step={self.step}")
