"""Plot both fixed pilot gates from the audited all-attempt conclusion."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joxing.provenance import digest, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    audit = json.loads(args.audit.read_text())
    assert audit["passed"] and audit["conclusion"] == result["conclusion"]
    assert result["status"] == "completed"
    seeds = result["conclusion"]["seeds"]
    assert len(seeds) == 2 and all(len(s["pairs"]) == 8 for s in seeds)
    args.out.mkdir(parents=True, exist_ok=False)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    labels = [str(s["training_seed"]) for s in seeds]
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    gains = [s["mean_paired_delta_s"] for s in seeds]
    bars = axes[0].bar(labels, gains, color="#ffb36b", width=0.5)
    axes[0].axhline(
        0.5, color="#e98585", linestyle="--", label="Required +0.5s each seed"
    )
    axes[0].axhline(0, color="white", linewidth=0.5)
    axes[0].set(
        title="Paired physical survival change",
        ylabel="Final minus initial (seconds)",
        ylim=(-0.15, 0.62),
        xlabel="Fresh training seed",
    )
    for bar, value in zip(bars, gains, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.025 if value >= 0 else -0.035),
            f"{value:+.3f}s",
            ha="center",
        )
    axes[0].legend(frameon=False, fontsize=8)
    survivors = [s["final_five_second_survivors"] for s in seeds]
    axes[1].bar(labels, survivors, color="#ffb36b", width=0.5)
    axes[1].axhline(4, color="#e98585", linestyle="--", label="Required 4/8 each seed")
    axes[1].set(
        title="Final five-second survivors",
        ylabel="Survivors out of 8 held-out starts",
        ylim=(0, 8),
        xlabel="Fresh training seed",
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Joxing v0.0.10 · motor improvement gates not met\n32 valid evaluations · 896 optimizer updates · 2 exact replays",
        fontsize=12,
    )
    fig.savefig(args.out / "pilot-gates.png", dpi=150)
    plt.close(fig)
    write_json(
        args.out / "report.json",
        {
            "source_result_hash": digest(result),
            "source_audit_hash": digest(audit),
            "selection": "Both preregistered seeds and all8pairs each; no discarded cases",
            "new_physics_steps": 0,
            "boxing_games": 0,
        },
    )


if __name__ == "__main__":
    main()
