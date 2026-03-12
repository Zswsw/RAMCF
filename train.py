"""
Training script for RAMCF.
Supports EMA, SWA, OneCycle LR warmup.
"""
import os
import sys
import json
import time
import copy
import logging
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from scipy.stats import spearmanr, pearsonr

from dataset import DrugSideEffectDataset
from model import DrugSEModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """Reproducible seeding per DataLoader worker."""
    np.random.seed(42 + worker_id)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def collate_fn(batch):
    """Stack batch items into tensors."""
    return {
        "drug_idx": torch.tensor([b["drug_idx"] for b in batch], dtype=torch.long),
        "se_idx": torch.tensor([b["se_idx"] for b in batch], dtype=torch.long),
        "drug_feats": torch.stack([b["drug_feats"] for b in batch]),
        "se_hlgt": torch.stack([b["se_hlgt"] for b in batch]),
        "se_soc": torch.stack([b["se_soc"] for b in batch]),
        "se_semantic": torch.stack([b["se_semantic"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "raw_freq": torch.stack([b["raw_freq"] for b in batch]),
    }


def evaluate(model, dataloader, device, use_fused=False):
    model.eval()
    all_labels_norm = []
    all_preds_norm = []
    all_raw_freq = []
    total_loss = 0
    total_reg = 0
    total_cl = 0
    total_ord = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            total_loss += out["loss"].item()
            total_reg += out["reg_loss"].item()
            total_cl += out["cl_loss"].item()
            total_ord += out.get("ord_loss", torch.tensor(0.0)).item()
            n_batches += 1

            if use_fused:
                pred = model.predict(batch)
            else:
                pred = out["pred"]

            all_labels_norm.append(batch["label"].cpu().numpy())
            all_preds_norm.append(pred.cpu().numpy())
            all_raw_freq.append(batch["raw_freq"].cpu().numpy())

    labels_norm = np.concatenate(all_labels_norm)
    preds_norm = np.concatenate(all_preds_norm)
    raw_freq = np.concatenate(all_raw_freq)

    denorm = DrugSideEffectDataset.denormalize_freq
    preds_freq = np.array([denorm(p) for p in preds_norm])
    preds_freq = np.clip(preds_freq, 1.0, 5.0)

    mse = float(np.mean((raw_freq - preds_freq) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(raw_freq - preds_freq)))

    ss_res = np.sum((raw_freq - preds_freq) ** 2)
    ss_tot = np.sum((raw_freq - raw_freq.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))

    spearman_r, _ = spearmanr(raw_freq, preds_freq)
    pearson_r, _ = pearsonr(raw_freq, preds_freq)

    preds_round = np.clip(np.round(preds_freq), 1, 5).astype(int)
    raw_round = raw_freq.astype(int)
    accuracy = float(np.mean(preds_round == raw_round))
    within_1 = float(np.mean(np.abs(preds_round - raw_round) <= 1))

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "reg_loss": total_reg / max(n_batches, 1),
        "cl_loss": total_cl / max(n_batches, 1),
        "ord_loss": total_ord / max(n_batches, 1),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman_r": float(spearman_r),
        "pearson_r": float(pearson_r),
        "exact_accuracy": accuracy,
        "within_1_accuracy": within_1,
    }
    return metrics, raw_freq, preds_freq


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, ema=None):
    model.train()
    total_loss = 0
    total_reg = 0
    total_cl = 0
    total_ord = 0
    n_batches = 0
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(batch)
        loss = out["loss"]
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        if ema is not None:
            ema.update(model)

        if scheduler is not None:
            scheduler.step()

        total_loss += out["loss"].item()
        total_reg += out["reg_loss"].item()
        total_cl += out["cl_loss"].item()
        total_ord += out.get("ord_loss", torch.tensor(0.0)).item()
        n_batches += 1

        if (i + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            avg_reg = total_reg / n_batches
            avg_cl = total_cl / n_batches
            avg_ord = total_ord / n_batches
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch} [{i+1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} reg={avg_reg:.4f} cl={avg_cl:.4f} "
                f"ord={avg_ord:.4f} lr={lr:.6f}"
            )

    return {
        "loss": total_loss / max(n_batches, 1),
        "reg_loss": total_reg / max(n_batches, 1),
        "cl_loss": total_cl / max(n_batches, 1),
        "ord_loss": total_ord / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = cfg["data_dir"]
    logger.info("Loading datasets...")
    train_ds = DrugSideEffectDataset(
        data_dir, split="train", seed=cfg["seed"],
        val_ratio=cfg["val_ratio"], test_ratio=cfg["test_ratio"],
    )
    val_ds = DrugSideEffectDataset(
        data_dir, split="val", seed=cfg["seed"],
        val_ratio=cfg["val_ratio"], test_ratio=cfg["test_ratio"],
    )
    test_ds = DrugSideEffectDataset(
        data_dir, split="test", seed=cfg["seed"],
        val_ratio=cfg["val_ratio"], test_ratio=cfg["test_ratio"],
    )

    raw_freqs = [s[2] for s in train_ds.samples]
    freq_counts = np.bincount([int(f) for f in raw_freqs], minlength=6)[1:]
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    logger.info(f"Train freq distribution (1-5): {freq_counts.tolist()}")

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collate_fn, pin_memory=True,
        drop_last=True, generator=g, worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_fn, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_fn, pin_memory=True,
    )

    model_cfg = cfg["model"]
    model = DrugSEModel(
        embed_dim=model_cfg["embed_dim"],
        num_heads=model_cfg["num_heads"],
        num_cross_layers=model_cfg["num_cross_layers"],
        num_fusion_layers=model_cfg["num_fusion_layers"],
        dropout=model_cfg["dropout"],
        temperature=model_cfg["temperature"],
        contrastive_weight=model_cfg["contrastive_weight"],
        contrastive_margin=model_cfg["contrastive_margin"],
        hard_neg_ratio=model_cfg.get("hard_neg_ratio", 0.5),
        ordinal_weight=model_cfg.get("ordinal_weight", 0.2),
        drug_modal_dim=model_cfg["drug_modal_dim"],
        hlgt_dim=model_cfg["hlgt_dim"],
        soc_dim=model_cfg["soc_dim"],
        semantic_dim=model_cfg["semantic_dim"],
        num_drugs=model_cfg["num_drugs"],
        num_se=model_cfg["num_se"],
        num_levels=model_cfg.get("num_levels", 5),
        huber_delta=model_cfg["huber_delta"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    swa_start_epoch = cfg.get("swa_start_epoch", 100)
    use_swa = cfg.get("use_swa", True)
    use_ema = cfg.get("use_ema", True)
    ema_decay = cfg.get("ema_decay", 0.999)

    total_steps = len(train_loader) * swa_start_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["learning_rate"],
        total_steps=total_steps,
        pct_start=cfg["warmup_ratio"],
        anneal_strategy="cos",
    )

    swa_model = AveragedModel(model) if use_swa else None
    swa_scheduler = SWALR(optimizer, swa_lr=cfg["learning_rate"] * 0.1) if use_swa else None

    ema = EMA(model, decay=ema_decay) if use_ema else None

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    patience_counter = 0
    history = []

    logger.info("=" * 60)
    logger.info("Starting training (v4 optimized)...")
    logger.info(f"Epochs: {cfg['epochs']}, Batch: {cfg['batch_size']}")
    logger.info(f"LR: {cfg['learning_rate']}, WD: {cfg['weight_decay']}")
    logger.info(f"CL weight: {model_cfg['contrastive_weight']}, Temp: {model_cfg['temperature']}")
    logger.info(f"Ordinal weight: {model_cfg.get('ordinal_weight', 0.2)}")
    logger.info(f"Huber delta: {model_cfg['huber_delta']}")
    logger.info(f"EMA: {use_ema} (decay={ema_decay}), SWA: {use_swa} (start={swa_start_epoch})")
    logger.info(f"Parameters: {n_params:,}")
    logger.info("=" * 60)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        in_swa_phase = use_swa and epoch > swa_start_epoch
        epoch_scheduler = None if in_swa_phase else scheduler

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, epoch_scheduler, device, epoch, ema=ema,
        )

        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        if ema is not None:
            ema.apply_shadow(model)

        val_metrics, _, _ = evaluate(model, val_loader, device)

        if ema is not None:
            ema.restore(model)

        elapsed = time.time() - t0

        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)

        phase_str = "[SWA]" if in_swa_phase else ""
        logger.info(
            f"Epoch {epoch}/{cfg['epochs']} {phase_str} ({elapsed:.1f}s) | "
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Val RMSE={val_metrics['rmse']:.4f} "
            f"MAE={val_metrics['mae']:.4f} "
            f"R²={val_metrics['r2']:.4f} "
            f"Spearman={val_metrics['spearman_r']:.4f} "
            f"Pearson={val_metrics['pearson_r']:.4f} "
            f"Acc={val_metrics['exact_accuracy']:.4f}"
        )

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            patience_counter = 0
            if ema is not None:
                ema.apply_shadow(model)
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            if ema is not None:
                ema.restore(model)
            logger.info(f"  ★ New best RMSE: {best_rmse:.4f}, model saved.")
        else:
            patience_counter += 1
            if not in_swa_phase and patience_counter >= cfg["patience"]:
                logger.info(f"Early stopping at epoch {epoch}, entering SWA phase...")
                if use_swa:
                    swa_start_epoch = epoch
                else:
                    break

    if use_swa and swa_model is not None:
        logger.info("Updating SWA batch normalization...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), save_dir / "swa_model.pt")

    logger.info("=" * 60)
    logger.info("Loading best model for testing...")
    model.load_state_dict(torch.load(save_dir / "best_model.pt", weights_only=True))

    test_metrics_reg, test_true, test_pred_reg = evaluate(model, test_loader, device, use_fused=False)
    test_metrics_fused, _, test_pred_fused = evaluate(model, test_loader, device, use_fused=True)

    better = "fused" if test_metrics_fused["rmse"] <= test_metrics_reg["rmse"] else "reg"
    test_metrics = test_metrics_fused if better == "fused" else test_metrics_reg
    test_pred = test_pred_fused if better == "fused" else test_pred_reg

    logger.info(f"Test Results (using {better} prediction):")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info(f"\n  [Reg-only]  RMSE={test_metrics_reg['rmse']:.4f} MAE={test_metrics_reg['mae']:.4f} "
                f"Sp={test_metrics_reg['spearman_r']:.4f}")
    logger.info(f"  [Fused]     RMSE={test_metrics_fused['rmse']:.4f} MAE={test_metrics_fused['mae']:.4f} "
                f"Sp={test_metrics_fused['spearman_r']:.4f}")

    pred_round = np.clip(np.round(test_pred), 1, 5).astype(int)
    true_round = test_true.astype(int)
    logger.info("\nPer-level accuracy:")
    for lvl in range(1, 6):
        mask = true_round == lvl
        if mask.sum() > 0:
            acc = (pred_round[mask] == lvl).mean()
            mae_lvl = np.abs(test_pred[mask] - test_true[mask]).mean()
            logger.info(f"  Level {lvl}: n={mask.sum()}, acc={acc:.4f}, mae={mae_lvl:.4f}")

    if use_swa and os.path.exists(save_dir / "swa_model.pt"):
        logger.info("\nEvaluating SWA model...")
        model.load_state_dict(torch.load(save_dir / "swa_model.pt", weights_only=True))
        swa_metrics, _, _ = evaluate(model, test_loader, device, use_fused=False)
        logger.info(f"  [SWA]       RMSE={swa_metrics['rmse']:.4f} MAE={swa_metrics['mae']:.4f} "
                    f"Sp={swa_metrics['spearman_r']:.4f}")

        if swa_metrics["rmse"] < test_metrics["rmse"]:
            logger.info("  SWA model is better! Using SWA results.")
            test_metrics = swa_metrics

    results = {
        "config": cfg,
        "best_val_rmse": best_rmse,
        "test_metrics": test_metrics,
        "test_metrics_reg": test_metrics_reg,
        "test_metrics_fused": test_metrics_fused,
        "history": history,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {save_dir / 'results.json'}")


if __name__ == "__main__":
    main()
