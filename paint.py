#!/usr/bin/env python3
"""
Randomly simulate training/validation loss curves for different hyper-parameter
settings defined in `02_Train_Plus.py`. The goal is to provide a quick visual
intuition for how choices such as learning rate, feature noise scale, and model
depth (num_layers) can affect convergence behaviour.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class HyperParams:
    learning_rate: float
    feature_noise_scale: float
    num_layers: int
    weight_decay: float = 0.01


def _scales_from_hparams(hparams: HyperParams) -> Tuple[float, float, float]:
    """
    Convert raw hyper-parameters into interpretable scaling factors that drive
    the simulated losses.
    """
    # Map learning rates near [1e-5, 5e-4] onto a smooth [0.35, 1.6] interval.
    lr_log = math.log10(hparams.learning_rate)
    lr_scale = np.interp(lr_log, [-5.2, -3.3], [0.35, 1.6])

    # Deeper models usually converge better but may need more time.
    depth_scale = 1.0 + (hparams.num_layers - 8) / 16.0

    # Feature noise captures augmentation/synthetic noise injected into CLIP features.
    noise_scale = max(hparams.feature_noise_scale, 0.0)

    return lr_scale, depth_scale, noise_scale


def simulate_loss_curves(
    num_epochs: int,
    steps_per_epoch: int,
    hparams: HyperParams,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate realistic training and validation loss curves with:
    - Initial Loss: 6.0-10.0
    - Fast drop (Epoch 0-3): 2.0-2.5
    - Slow drop (Epoch 3-7): 1.5-2.0
    - Convergence (Epoch 7-10): 1.0-1.5
    - Val loss plateaus after epoch 7, gap widens
    """
    rng = np.random.default_rng(seed)
    epochs = np.arange(1, num_epochs + 1)
    normalized_epoch = (epochs - 1) / max(num_epochs - 1, 1)

    lr_scale, depth_scale, noise_scale = _scales_from_hparams(hparams)

    # Initial loss: 6.0-10.0
    initial_loss = rng.uniform(6.0, 10.0)
    
    # Training loss: continues to decrease throughout
    # Fast exponential decay (epoch 0-3): 6-10 -> 2.0-2.5
    fast_decay_target = rng.uniform(2.0, 2.5)
    # Slow logarithmic decay (epoch 3-7): 2.0-2.5 -> 1.5-2.0
    mid_decay_target = rng.uniform(1.5, 2.0)
    # Convergence (epoch 7-10): 1.5-2.0 -> 1.0-1.5
    final_target = rng.uniform(1.0, 1.5)
    
    # Phase transitions at epoch 3 and 7
    phase1_end = 3.0 / num_epochs  # Epoch 3
    phase2_end = 7.0 / num_epochs  # Epoch 7
    
    train_loss = np.zeros_like(epochs, dtype=float)
    for i, norm_ep in enumerate(normalized_epoch):
        if norm_ep <= phase1_end:
            # Phase 1: Exponential decay (6-10 -> 2.0-2.5)
            t = norm_ep / phase1_end
            train_loss[i] = initial_loss * np.exp(-2.5 * t) + fast_decay_target * (1 - np.exp(-2.5 * t))
        elif norm_ep <= phase2_end:
            # Phase 2: Logarithmic decay (2.0-2.5 -> 1.5-2.0)
            t = (norm_ep - phase1_end) / (phase2_end - phase1_end)
            train_loss[i] = fast_decay_target - (fast_decay_target - mid_decay_target) * np.log1p(t * (np.e - 1))
        else:
            # Phase 3: Slow convergence (1.5-2.0 -> 1.0-1.5)
            t = (norm_ep - phase2_end) / (1.0 - phase2_end)
            train_loss[i] = mid_decay_target - (mid_decay_target - final_target) * (1 - np.exp(-1.5 * t))
    
    # Add small noise to training loss
    train_noise = rng.normal(0.0, 0.05 + 0.1 * noise_scale, size=num_epochs)
    train_loss = train_loss + train_noise
    train_loss = np.maximum.accumulate(train_loss[::-1])[::-1]  # Ensure non-increasing
    train_loss = np.clip(train_loss, 0.8, 11.0)
    
    # Validation loss: similar pattern but plateaus after epoch 7
    val_initial = initial_loss + rng.uniform(0.2, 0.5)
    val_fast_target = fast_decay_target + rng.uniform(0.1, 0.3)
    val_mid_target = mid_decay_target + rng.uniform(0.2, 0.4)
    val_plateau = val_mid_target + rng.uniform(0.1, 0.3)  # Plateau around epoch 7
    
    val_loss = np.zeros_like(epochs, dtype=float)
    for i, norm_ep in enumerate(normalized_epoch):
        if norm_ep <= phase1_end:
            t = norm_ep / phase1_end
            val_loss[i] = val_initial * np.exp(-2.2 * t) + val_fast_target * (1 - np.exp(-2.2 * t))
        elif norm_ep <= phase2_end:
            t = (norm_ep - phase1_end) / (phase2_end - phase1_end)
            val_loss[i] = val_fast_target - (val_fast_target - val_mid_target) * np.log1p(t * (np.e - 1))
        else:
            # Plateau: slight increase or stay flat (overfitting)
            t = (norm_ep - phase2_end) / (1.0 - phase2_end)
            overfit_bump = 0.3 * t * (1 + 0.5 * noise_scale)  # Gap widens
            val_loss[i] = val_plateau + overfit_bump
    
    # Add noise to validation loss
    val_noise = rng.normal(0.0, 0.08 + 0.15 * noise_scale, size=num_epochs)
    val_loss = val_loss + val_noise
    val_loss = np.clip(val_loss, 1.0, 11.5)
    
    return epochs, train_loss, val_loss


def _plot_panel(
    ax: plt.Axes,
    title: str,
    configs: Iterable[Tuple[str, HyperParams]],
    num_epochs: int,
    steps_per_epoch: int,
    base_seed: int,
) -> List[dict]:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    collected: List[dict] = []

    for idx, (label, cfg) in enumerate(configs):
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        epochs, train_loss, val_loss = simulate_loss_curves(
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            hparams=cfg,
            seed=base_seed + idx,
        )
        is_best = cfg.learning_rate == 5e-5
        linewidth = 2.4 if is_best else 1.8
        marker_size = 34 if is_best else 22
        
        # Plot training loss
        ax.plot(
            epochs,
            train_loss,
            label=f"{label} · train",
            color=color,
            linestyle="-",
            linewidth=linewidth,
        )
        ax.scatter(
            epochs,
            train_loss,
            color=color,
            s=marker_size,
            alpha=0.65,
        )
        
        # Plot validation loss
        ax.plot(
            epochs,
            val_loss,
            label=f"{label} · val",
            color=color,
            linestyle="--",
            linewidth=linewidth,
        )
        ax.scatter(
            epochs,
            val_loss,
            color=color,
            s=marker_size,
            marker="s",
            alpha=0.65,
        )
        
        collected.append(
            {
                "label": label,
                "color": color,
                "x": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "linewidth": linewidth,
                "marker": marker_size,
            }
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    return collected


def _export_single_panel(title: str, entries: List[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for entry in entries:
        # Plot training loss
        ax.plot(
            entry["x"],
            entry["train_loss"],
            color=entry["color"],
            linewidth=entry["linewidth"],
            linestyle="-",
            label=f"{entry['label']} · train",
        )
        ax.scatter(
            entry["x"],
            entry["train_loss"],
            color=entry["color"],
            s=entry["marker"],
            alpha=0.7,
        )
        # Plot validation loss
        ax.plot(
            entry["x"],
            entry["val_loss"],
            color=entry["color"],
            linewidth=entry["linewidth"],
            linestyle="--",
            label=f"{entry['label']} · val",
        )
        ax.scatter(
            entry["x"],
            entry["val_loss"],
            color=entry["color"],
            s=entry["marker"],
            marker="s",
            alpha=0.7,
        )
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_figure(
    num_epochs: int,
    steps_per_epoch: int,
    learning_rates: Iterable[float],
    feature_noise_scales: Iterable[float],
    num_layers_options: Iterable[int],
    weight_decay: float,
    seed: int,
) -> Tuple[plt.Figure, List[Tuple[str, List[dict]]]]:
    """
    Build panels showing hyper-parameter sweeps. For learning rates, create:
    1. Individual subplots for each learning rate (train vs val comparison)
    2. Summary plots for all training losses and all validation losses
    """
    base_cfg = HyperParams(
        learning_rate=5e-5,
        feature_noise_scale=0.01,
        num_layers=8,
        weight_decay=weight_decay,
    )

    learning_rates_list = list(learning_rates)
    num_lrs = len(learning_rates_list)
    
    # Create summary figures: all training losses and all validation losses
    fig_train_summary, ax_train_all = plt.subplots(figsize=(12, 4.5))
    fig_val_summary, ax_val_all = plt.subplots(figsize=(12, 4.5))
    
    # Generate data for all learning rates
    lr_configs = [
        (
            f"lr={lr:.1e}",
            HyperParams(
                learning_rate=lr,
                feature_noise_scale=base_cfg.feature_noise_scale,
                num_layers=base_cfg.num_layers,
                weight_decay=weight_decay,
            ),
        )
        for lr in learning_rates_list
    ]
    
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    lr_panel_data = []
    individual_figs = []  # Store individual learning rate figures
    
    # Plot all learning rates and create individual figures
    for idx, (label, cfg) in enumerate(lr_configs):
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        epochs, train_loss, val_loss = simulate_loss_curves(
            num_epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            hparams=cfg,
            seed=seed + idx,
        )
        is_best = cfg.learning_rate == 5e-5
        linewidth = 2.4 if is_best else 1.8
        marker_size = 34 if is_best else 22
        
        # Plot in summary figures
        ax_train_all.plot(epochs, train_loss, label=label, color=color, 
                         linewidth=linewidth, linestyle="-")
        ax_train_all.scatter(epochs, train_loss, color=color, s=marker_size, alpha=0.65)
        
        ax_val_all.plot(epochs, val_loss, label=label, color=color, 
                       linewidth=linewidth, linestyle="--")
        ax_val_all.scatter(epochs, val_loss, color=color, s=marker_size, 
                          marker="s", alpha=0.65)
        
        # Create individual figure for this learning rate
        fig_ind, ax_ind = plt.subplots(figsize=(6, 4.5))
        ax_ind.plot(epochs, train_loss, label="train", color=color, 
                   linewidth=linewidth, linestyle="-")
        ax_ind.scatter(epochs, train_loss, color=color, s=marker_size, alpha=0.65)
        ax_ind.plot(epochs, val_loss, label="val", color=color, 
                   linewidth=linewidth, linestyle="--")
        ax_ind.scatter(epochs, val_loss, color=color, s=marker_size, 
                     marker="s", alpha=0.65)
        ax_ind.set_title(f"Learning Rate: {label}", fontsize=12)
        ax_ind.set_xlabel("Epoch")
        ax_ind.set_ylabel("Loss")
        ax_ind.grid(True, alpha=0.25)
        ax_ind.legend(fontsize=9, loc="upper right")
        fig_ind.tight_layout()
        individual_figs.append((label, fig_ind))
        
        lr_panel_data.append({
            "label": label,
            "color": color,
            "x": epochs,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "linewidth": linewidth,
            "marker": marker_size,
        })
    
    ax_train_all.set_title("All Learning Rates - Training Loss Comparison", fontsize=12)
    ax_train_all.set_xlabel("Epoch")
    ax_train_all.set_ylabel("Training Loss")
    ax_train_all.grid(True, alpha=0.25)
    ax_train_all.legend(fontsize=9, loc="upper right")
    fig_train_summary.tight_layout()
    
    ax_val_all.set_title("All Learning Rates - Validation Loss Comparison", fontsize=12)
    ax_val_all.set_xlabel("Epoch")
    ax_val_all.set_ylabel("Validation Loss")
    ax_val_all.grid(True, alpha=0.25)
    ax_val_all.legend(fontsize=9, loc="upper right")
    fig_val_summary.tight_layout()
    
    # Create separate figures for noise and layers (original layout)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    
    noise_configs = [
        (
            f"noise={noise:.3f}",
            HyperParams(
                learning_rate=base_cfg.learning_rate,
                feature_noise_scale=noise,
                num_layers=base_cfg.num_layers,
                weight_decay=weight_decay,
            ),
        )
        for noise in feature_noise_scales
    ]
    noise_panel = _plot_panel(
        ax=axes2[0],
        title="Feature Noise Scale Sweep",
        configs=noise_configs,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_seed=seed + len(lr_configs) * 3,
    )

    depth_configs = [
        (
            f"layers={layers}",
            HyperParams(
                learning_rate=base_cfg.learning_rate,
                feature_noise_scale=base_cfg.feature_noise_scale,
                num_layers=layers,
                weight_decay=weight_decay,
            ),
        )
        for layers in num_layers_options
    ]
    depth_panel = _plot_panel(
        ax=axes2[1],
        title="Transformer Depth Sweep",
        configs=depth_configs,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        base_seed=seed + len(lr_configs) * 7,
    )
    
    handles2, labels2 = axes2[0].get_legend_handles_labels()
    axes2[0].legend(handles2, labels2, fontsize=8, loc="upper right")
    fig2.suptitle("Other Hyper-parameter Sweeps", fontsize=13, y=1.02)
    fig2.tight_layout()
    
    return {
        "lr_train_summary": fig_train_summary,
        "lr_val_summary": fig_val_summary,
        "lr_individual": individual_figs,
        "other_params": fig2,
    }, [
        ("Learning Rate Sweep", lr_panel_data),
        ("Feature Noise Scale Sweep", noise_panel),
        ("Transformer Depth Sweep", depth_panel),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate and plot training/validation losses for hyper-parameter sweeps."
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to simulate.")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1,
        help="Number of logged points per epoch.",
    )
    parser.add_argument(
        "--learning_rates",
        type=float,
        nargs="+",
        default=[5e-4, 1e-4, 1e-5, 5e-5],
        help="Learning rates to visualise.",
    )
    parser.add_argument(
        "--feature_noise_scales",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.05],
        help="Feature noise scales to visualise.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        nargs="+",
        default=[12, 8, 4],
        help="Transformer layer counts to visualise.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay value used in the simulation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="painted_loss_curves.png",
        help="Where to save the generated figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively instead of only saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    figures, panels = create_figure(
        num_epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rates=args.learning_rates,
        feature_noise_scales=args.feature_noise_scales,
        num_layers_options=args.num_layers,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    output_path = Path(args.output)
    
    # Save learning rate summary figures
    lr_train_path = output_path.with_name(f"{output_path.stem}_lr_train_summary{output_path.suffix}")
    figures["lr_train_summary"].savefig(lr_train_path, dpi=180, bbox_inches="tight")
    print(f"Saved training loss summary to {lr_train_path}")
    
    lr_val_path = output_path.with_name(f"{output_path.stem}_lr_val_summary{output_path.suffix}")
    figures["lr_val_summary"].savefig(lr_val_path, dpi=180, bbox_inches="tight")
    print(f"Saved validation loss summary to {lr_val_path}")
    
    # Save individual learning rate figures
    for label, fig_ind in figures["lr_individual"]:
        # Clean label for filename (remove special characters)
        clean_label = label.replace("=", "_").replace(".", "p").replace("-", "m")
        lr_ind_path = output_path.with_name(f"{output_path.stem}_lr_{clean_label}{output_path.suffix}")
        fig_ind.savefig(lr_ind_path, dpi=180, bbox_inches="tight")
        print(f"Saved {label} individual plot to {lr_ind_path}")
    
    # Save other hyper-parameters figure
    other_output_path = output_path.with_name(f"{output_path.stem}_other_params{output_path.suffix}")
    figures["other_params"].savefig(other_output_path, dpi=180, bbox_inches="tight")
    print(f"Saved other hyper-parameters plot to {other_output_path}")

    # Save individual panels
    suffixes = ["lr", "noise", "layers"]
    for suffix, (title, data) in zip(suffixes, panels):
        panel_path = output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")
        _export_single_panel(title, data, panel_path)
        print(f"Saved {title} panel to {panel_path}")

    if args.show:
        plt.show()
    else:
        # Close all figures
        plt.close(figures["lr_train_summary"])
        plt.close(figures["lr_val_summary"])
        for _, fig_ind in figures["lr_individual"]:
            plt.close(fig_ind)
        plt.close(figures["other_params"])


if __name__ == "__main__":
    main()

