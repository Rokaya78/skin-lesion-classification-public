"""
Master runner — executes all four weeks in order.

Week 1 : Train baseline CNN from scratch
Week 2 : Train MobileNetV2 — feature extraction (frozen backbone)
Week 3 : Train MobileNetV2 — progressive unfreezing
         Train MobileNetV2 — full fine-tune
Week 4 : Evaluate all four models → compare → plots

Usage:
    python run_all.py                                      # full pipeline
    python run_all.py --skip_train                        # eval + compare only
    python run_all.py --epochs 30 --batch_size 32
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import os


def run(cmd: list[str]):
    print(f"\n{'='*60}")
    print("CMD: " + " ".join(cmd))
    print("="*60)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         default="data/HAM10000_metadata.csv")
    parser.add_argument("--img_dirs",    nargs="+",
                        default=["data/HAM10000_images_part1",
                                 "data/HAM10000_images_part2"])
    parser.add_argument("--epochs",         type=int, default=30)
    parser.add_argument("--baseline_epochs",type=int, default=40,
                        help="Baseline CNN typically needs more epochs (no pretrained weights)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--ckpt_dir",    default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--unfreeze5_epoch",  type=int, default=10)
    parser.add_argument("--unfreeze10_epoch", type=int, default=20)
    parser.add_argument("--skip_train",  action="store_true",
                        help="Skip all training; run evaluation and comparison only")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip the scratch baseline and only run MobileNetV2 strategies")
    args = parser.parse_args()

    py  = sys.executable
    src = os.path.join(os.path.dirname(__file__), "src")

    common = [
        "--csv",         args.csv,
        "--img_dirs",    *args.img_dirs,
        "--batch_size",  str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--ckpt_dir",    args.ckpt_dir,
        "--results_dir", args.results_dir,
    ]

    mobilenet_strategies = ["feature_extraction", "progressive", "full_finetune"]
    mobilenet_lrs = {
        "feature_extraction": "1e-3",
        "progressive":        "1e-3",
        "full_finetune":      "1e-4",
    }

    # -----------------------------------------------------------------------
    # WEEK 1 — Baseline CNN from scratch
    # -----------------------------------------------------------------------
    if not args.skip_train and not args.skip_baseline:
        print("\n>>> WEEK 1: Training baseline CNN from scratch")
        run([
            py, os.path.join(src, "train_baseline.py"),
            "--epochs", str(args.baseline_epochs),
            *common,
        ])

    # -----------------------------------------------------------------------
    # WEEK 2-3 — MobileNetV2 transfer learning strategies
    # -----------------------------------------------------------------------
    if not args.skip_train:
        for strat in mobilenet_strategies:
            week = "WEEK 2" if strat == "feature_extraction" else "WEEK 3"
            print(f"\n>>> {week}: Training MobileNetV2 — {strat}")
            train_cmd = [
                py, os.path.join(src, "train.py"),
                "--strategy", strat,
                "--epochs",   str(args.epochs),
                "--lr",       mobilenet_lrs[strat],
                *common,
            ]
            if strat == "progressive":
                train_cmd += [
                    "--unfreeze5_epoch",  str(args.unfreeze5_epoch),
                    "--unfreeze10_epoch", str(args.unfreeze10_epoch),
                ]
            run(train_cmd)

    # -----------------------------------------------------------------------
    # WEEK 4 — Evaluate all models
    # -----------------------------------------------------------------------
    print("\n>>> WEEK 4: Evaluating all models")
    all_strategies = ["baseline_cnn"] + mobilenet_strategies
    for strat in all_strategies:
        ckpt = os.path.join(args.ckpt_dir, f"{strat}_best.pth")
        if not os.path.isfile(ckpt):
            print(f"  [skip] No checkpoint for {strat}")
            continue
        run([
            py, os.path.join(src, "evaluate.py"),
            "--strategy", strat,
            *common,
        ])

    # -----------------------------------------------------------------------
    # WEEK 4 — Side-by-side comparison & plots
    # -----------------------------------------------------------------------
    print("\n>>> WEEK 4: Generating comparison report")
    run([py, os.path.join(src, "compare.py"),
         "--results_dir", args.results_dir])

    print("\n" + "="*60)
    print("All done!  Results are in:", args.results_dir)
    print("  comparison_table.txt       — metrics table")
    print("  comparison_overall.png     — accuracy / F1 / recall bar chart")
    print("  comparison_per_class_f1.png— per-class F1 for all 4 models")
    print("  learning_curves.png        — training & val curves")
    print("  bkl_mel_confusion.png      — BKL ↔ MEL danger analysis")
    print("="*60)


if __name__ == "__main__":
    main()
