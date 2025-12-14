import os
import json
import argparse
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", default="plots")
    return parser.parse_args()


def load_results(results_dir: str) -> List[Dict]:
    """Load all JSON results from a directory."""
    results = []
    for f in os.listdir(results_dir):
        if f.endswith(".json"):
            with open(os.path.join(results_dir, f)) as fh:
                results.append(json.load(fh))
    return results


def experiment_label(res: Dict) -> str:
    """Generate a label for the experiment based on its mode and config."""
    mode = res["mode"]
    if mode == "full_ft":
        return "Full FT"
    if mode == "adapter":
        return "Adapter"
    if mode == "lora":
        return f"LoRA (r={res['config']['r']})"
    return mode


def build_summary(results: List[Dict]) -> pd.DataFrame:
    """Build a summary DataFrame from results."""
    rows = []
    for r in results:
        rows.append({
            "Model": experiment_label(r),
            "Accuracy": r["metrics"]["eval_accuracy"],
            "Trainable params": r["params"]["trainable_parameters"],
            "Trainable %": r["params"]["trainable_percentage"],
            "Time (min)": r["elapsed_sec"] / 60,
            "mode": r["mode"],
            "r": r["config"].get("r", None),
        })

    df = pd.DataFrame(rows)

    def sort_key(row):
        if row["mode"] == "full_ft":
            return (2, 0)
        if row["mode"] == "adapter":
            return (0, 0)
        return (1, row["r"])

    df["_sort"] = df.apply(sort_key, axis=1)
    df = df.sort_values("_sort").drop(columns="_sort")

    return df


def save_latex_table(df: pd.DataFrame, outpath: str):
    """Save summary DataFrame as a LaTeX table."""
    latex = df.drop(columns=["mode", "r"]).to_latex(
        index=False,
        float_format="%.3f",
        caption="Comparison of fine-tuning strategies on SST-2",
        label="tab:ft-comparison",
    )
    with open(outpath, "w") as f:
        f.write(latex)


def plot_accuracy_vs_params(df: pd.DataFrame, outpath: str):
    """Plot accuracy vs trainable parameters."""
    plt.figure(figsize=(6.5, 5))

    for _, row in df.iterrows():
        plt.scatter(
            row["Trainable (%)"],
            row["Accuracy"],
            s=90,
        )
        plt.annotate(
            row["Model"],
            (row["Trainable (%)"], row["Accuracy"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
        )

    plt.xscale("log")
    plt.xlabel("Trainable parameters (%) [log scale]")
    plt.ylabel("Validation accuracy")
    plt.title("Accuracy vs trainable parameters")

    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_training_curves(results: List[Dict], outpath: str):
    """Plot training loss curves from results."""
    plt.figure(figsize=(7, 5))

    for r in results:
        history = r.get("log_history", [])
        steps = [h["step"] for h in history if "loss" in h]
        losses = [h["loss"] for h in history if "loss" in h]

        if not steps:
            continue

        plt.plot(
            steps,
            losses,
            label=experiment_label(r),
            linewidth=2,
            alpha=0.9
        )

    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title("Training loss curves")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    """Main function to generate plots and tables from results."""
    # Parse arguments
    args = parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Load results from directory
    results = load_results(args.results_dir)
    # Build summary DataFrame
    df = build_summary(results)
    # Save table as LaTeX
    save_latex_table(df, os.path.join(args.out_dir, "comparison_table.tex"))

    # Create plots
    plot_accuracy_vs_params(
        df,
        os.path.join(args.out_dir, "accuracy_vs_params.png")
    )
    plot_training_curves(
        results,
        os.path.join(args.out_dir, "training_loss_curves.png")
    )

    print(f"[OK] Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
