import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import NeuroSenseConfig
from src.losses import FocalLoss
from src.metrics import compute_metrics
from src.models.neurosense_net import NeuroSenseNet
from src.utils import MELD_EMOTIONS, setup_logger

logger = setup_logger("neurosense.engine")


class Trainer:
    """Production training loop with AMP, gradient accumulation/clipping, and early stopping."""

    def __init__(
        self,
        model: NeuroSenseNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: NeuroSenseConfig,
        run_dir: Path,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = next(model.parameters()).device
        self.model.to(self.device)

        # Loss Function
        if config.loss.name == "focal":
            self.criterion = FocalLoss(
                gamma=config.loss.gamma,
                weight=class_weights.to(self.device) if class_weights is not None else None,
                reduction="mean",
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights.to(self.device) if class_weights is not None else None
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # LR Scheduler with Warmup + Cosine Decay
        total_steps = len(train_loader) * config.training.epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # AMP Scaler
        use_amp = config.training.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.use_amp = use_amp

        self.best_macro_f1 = -1.0
        self.patience_counter = 0
        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.training.epochs} [Train]")
        for step, batch in enumerate(pbar):
            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=device_type, enabled=self.use_amp):
                outputs = self.model(batch)
                logits = outputs["logits"]
                targets = batch["label"].to(self.device)
                loss = self.criterion(logits, targets)
                loss = loss / self.config.training.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            total_loss += loss.item() * self.config.training.gradient_accumulation_steps

            if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"})

        return total_loss / len(self.train_loader)

    def train(self) -> Dict[str, Any]:
        logger.info(f"Starting training for {self.config.training.epochs} epochs on device {self.device}...")

        for epoch in range(self.config.training.epochs):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)

            # Validation
            val_evaluator = Evaluator(self.model, self.val_loader, self.device)
            val_results = val_evaluator.evaluate()

            val_loss = val_results["metrics"]["loss"]
            val_macro_f1 = val_results["metrics"]["macro_f1"]
            val_acc = val_results["metrics"]["accuracy"]
            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_macro_f1": float(val_macro_f1),
                "val_accuracy": float(val_acc),
                "lr": float(current_lr),
                "time_sec": float(elapsed),
            }
            self.history.append(epoch_record)

            logger.info(
                f"Epoch {epoch+1:02d}/{self.config.training.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f} | ({elapsed:.1f}s)"
            )

            # Save Checkpoints
            is_best = val_macro_f1 > self.best_macro_f1
            if is_best:
                self.best_macro_f1 = val_macro_f1
                self.patience_counter = 0
                self.save_checkpoint(self.checkpoint_dir / "best.pt", epoch + 1, val_macro_f1)
                logger.info(f"  --> Saved new best checkpoint with Val Macro-F1: {val_macro_f1:.4f}")
            else:
                self.patience_counter += 1

            self.save_checkpoint(self.checkpoint_dir / "last.pt", epoch + 1, val_macro_f1)

            # Save History CSV
            pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)

            # Early Stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load best checkpoint back into model
        best_ckpt = self.checkpoint_dir / "best.pt"
        if best_ckpt.exists():
            logger.info(f"Restoring best checkpoint from {best_ckpt} for final evaluation.")
            self.load_checkpoint(best_ckpt)

        return {"best_macro_f1": self.best_macro_f1, "history": self.history}

    def save_checkpoint(self, path: Path, epoch: int, macro_f1: float) -> None:
        state = {
            "epoch": epoch,
            "best_macro_f1": macro_f1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "feature_dims": self.model.feature_dims,
            "config": self.config,
            "rng_torch": torch.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])


class Evaluator:
    """Evaluation runner collecting predictions, gate weights, and computing metrics."""

    def __init__(self, model: NeuroSenseNet, dataloader: DataLoader, device: torch.device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(
        self, override_availability: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        self.model.eval()
        self.model.to(self.device)

        all_logits = []
        all_targets = []
        all_gate_weights = []
        all_ids = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.dataloader:
                outputs = self.model(batch, override_availability=override_availability)
                logits = outputs["logits"]
                gate_weights = outputs["gate_weights"]
                targets = batch["label"].to(self.device)

                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                all_gate_weights.append(gate_weights.cpu())
                all_ids.extend(batch["ids"])

        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)
        cat_gate_weights = torch.cat(all_gate_weights, dim=0)

        probs = torch.softmax(cat_logits, dim=-1).numpy()
        preds = cat_logits.argmax(dim=-1).numpy()
        targets_np = cat_targets.numpy()
        avg_loss = total_loss / len(self.dataloader)

        metrics = compute_metrics(targets_np, preds, class_names=MELD_EMOTIONS)
        metrics["loss"] = float(avg_loss)

        return {
            "metrics": metrics,
            "y_true": targets_np,
            "y_pred": preds,
            "probs": probs,
            "gate_weights": cat_gate_weights.numpy(),
            "ids": all_ids,
        }


def run_modality_ablation_study(
    model: NeuroSenseNet,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    """Run full modality ablation protocol under 7 availability conditions."""
    evaluator = Evaluator(model, test_loader, device)

    ablation_conditions = {
        "All (T+A+V)": None,
        "No Text (A+V)": {"text": False, "audio": True, "video": True},
        "No Audio (T+V)": {"text": True, "audio": False, "video": True},
        "No Video (T+A)": {"text": True, "audio": True, "video": False},
        "Text Only": {"text": True, "audio": False, "video": False},
        "Audio Only": {"text": False, "audio": True, "video": False},
        "Video Only": {"text": False, "audio": False, "video": True},
    }

    results = {}
    logger.info("Executing Modality Ablation Protocol...")
    for cond_name, override in ablation_conditions.items():
        res = evaluator.evaluate(override_availability=override)
        m = res["metrics"]
        mean_gates = res["gate_weights"].mean(axis=0).tolist()
        results[cond_name] = {
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mean_gate_text": mean_gates[0],
            "mean_gate_audio": mean_gates[1],
            "mean_gate_video": mean_gates[2],
        }
        logger.info(
            f"  - [{cond_name:<16}] Macro-F1: {m['macro_f1']:.4f} | Acc: {m['accuracy']:.4f} | "
            f"Gates: T={mean_gates[0]:.2f}, A={mean_gates[1]:.2f}, V={mean_gates[2]:.2f}"
        )

    return results
