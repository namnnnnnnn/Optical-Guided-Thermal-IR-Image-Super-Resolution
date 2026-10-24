"""
summarize_results.py
Role: Summarises outputs/metrics/results.csv (written by evaluate.py) without
      re-running the model. Scenes where LR == HR (infinite LR PSNR) are
      counted and excluded so the means stay finite.
Usage: python summarize_results.py [path/to/results.csv]
"""

import os
import sys
import csv
import numpy as np

from config import OUTPUT_DIR


def summarize(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])

    sr = {k: col(k) for k in ("PSNR", "SSIM", "RMSE")}
    lr = {k: col("LR_" + k) for k in ("PSNR", "SSIM", "RMSE")}
    seen = col("seen_in_training").astype(bool)
    identical = ~np.isfinite(lr["PSNR"]) | (lr["RMSE"] == 0)

    def report(title, mask):
        n = int(mask.sum())
        print(f"\n  {title} (n = {n})")
        print(f"  {'Metric':<6} | {'SR mean ± std':>18} | {'LR baseline':>12} | {'Gain':>9} | SR better")
        for k in ("PSNR", "SSIM", "RMSE"):
            a, b = sr[k][mask], lr[k][mask]
            better = (a < b) if k == "RMSE" else (a > b)
            print(f"  {k:<6} | {a.mean():>9.4f} ± {a.std():<6.4f} | {b.mean():>12.4f} | "
                  f"{a.mean() - b.mean():>+9.4f} | {better.mean() * 100:5.1f} %")

    print("=" * 72)
    print(f"  Results summary: {csv_path}")
    print(f"  Total test scenes             : {len(rows)}")
    print(f"  LR identical to HR (excluded) : {int(identical.sum())}")
    print(f"  Seen in training (overlap)    : {int(seen.sum())}")
    report("All comparable scenes", ~identical)
    report("Unseen-only comparable scenes", ~identical & ~seen)

    # Gain by input difficulty: quartiles of LR PSNR (Q1 = hardest scenes)
    valid = ~identical
    edges = np.percentile(lr["PSNR"][valid], [0, 25, 50, 75, 100])
    print("\n  Gain by input difficulty (quartiles of LR PSNR, Q1 = hardest)")
    print(f"  {'Quartile':<8} | {'LR PSNR range':>15} | {'n':>5} | {'PSNR gain':>9} | "
          f"{'SSIM gain':>9} | {'RMSE change':>11} | SR better (PSNR)")
    for q in range(4):
        m = valid & (lr["PSNR"] >= edges[q]) & ((lr["PSNR"] < edges[q + 1]) | (q == 3))
        dp = sr["PSNR"][m] - lr["PSNR"][m]
        ds = sr["SSIM"][m] - lr["SSIM"][m]
        dr = (sr["RMSE"][m].mean() / lr["RMSE"][m].mean() - 1) * 100
        print(f"  Q{q + 1:<7} | {edges[q]:6.2f} - {edges[q + 1]:6.2f} | {int(m.sum()):>5} | "
              f"{dp.mean():>+9.3f} | {ds.mean():>+9.4f} | {dr:>+10.1f}% | {(dp > 0).mean() * 100:5.1f} %")
    print("=" * 72)


if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1
              else os.path.join(OUTPUT_DIR, "metrics", "results.csv"))
