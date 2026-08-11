#!/usr/bin/env python3
import os
import csv
import re
import math
import argparse


def parse_laplacian_metrics(path):
    """Parse L0/L1 time & GS, and Reconstructed SSIM/PSNR/LPIPS from laplacian_metrics_summary.csv."""
    l0_time = float("nan")
    l0_gs = -1
    l1_time = float("nan")
    l1_gs = -1
    rec_ssim = float("nan")
    rec_psnr = float("nan")
    rec_lpips = float("nan")

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = row.get("Exp", "").strip()
            if exp == "L0":
                t = row.get("Time (min)", "N/A")
                g = row.get("GS Number", "N/A")
                if t != "N/A" and t != "":
                    l0_time = float(t)
                if g != "N/A" and g != "":
                    l0_gs = int(g)
            elif exp == "L1":
                t = row.get("Time (min)", "N/A")
                g = row.get("GS Number", "N/A")
                if t != "N/A" and t != "":
                    l1_time = float(t)
                if g != "N/A" and g != "":
                    l1_gs = int(g)
            elif exp == "Reconstructed":
                ssim = row.get("SSIM", "N/A")
                psnr = row.get("PSNR", "N/A")
                lpips = row.get("LPIPS", "N/A")
                if ssim != "N/A" and ssim != "":
                    rec_ssim = float(ssim)
                if psnr != "N/A" and psnr != "":
                    rec_psnr = float(psnr)
                if lpips != "N/A" and lpips != "":
                    rec_lpips = float(lpips)

    return {
        "rec_SSIM": rec_ssim,
        "rec_PSNR": rec_psnr,
        "rec_LPIPS": rec_lpips,
        "L0_time_min": l0_time,
        "L0_gs_number": l0_gs,
        "L1_time_min": l1_time,
        "L1_gs_number": l1_gs,
    }


def parse_fps_info(path):
    """Parse FPS from FPS_INFO."""
    with open(path, "r") as f:
        text = f.read()

    # Example: "[test] Rendered 25 frames in 0.17 seconds. Average FPS: 151.18"
    m = re.search(r"Average\s+FPS:\s*([\d.]+)", text)
    return float(m.group(1)) if m else float("nan")


def parse_experiment_dir(exp_dir):
    """Parse all needed metrics from a single experiment directory."""
    lap_path = os.path.join(exp_dir, "laplacian_metrics_summary.csv")
    fps_path = os.path.join(exp_dir, "FPS_INFO")

    if not os.path.isfile(lap_path) or not os.path.isfile(fps_path):
        print(f"[Skip] Missing laplacian_metrics_summary.csv or FPS_INFO in {exp_dir}")
        return None

    result = {"exp_name": os.path.basename(exp_dir)}

    result.update(parse_laplacian_metrics(lap_path))
    result["fps"] = parse_fps_info(fps_path)

    return result


def mean_ignore_missing(values):
    """Mean of list values ignoring NaN and None."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Laplacian experiment metrics into CSV."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory containing experiment subdirectories.",
    )

    args = parser.parse_args()
    base_dir = args.base_dir

    if not os.path.isdir(base_dir):
        print(f"Error: directory '{base_dir}' not found.")
        return

    subdirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    subdirs.sort()

    all_results = []
    for d in subdirs:
        res = parse_experiment_dir(d)
        if res:
            all_results.append(res)

    if not all_results:
        print("No valid experiment results found.")
        return

    # CSV header
    header = [
        "exp_name",
        "rec_SSIM",
        "rec_PSNR",
        "rec_LPIPS",
        "L0_time_min",
        "L0_gs_number",
        "L1_time_min",
        "L1_gs_number",
        "fps",
    ]

    # ---- Add average row ----
    avg_row = {"exp_name": "AVERAGE"}
    for k in header:
        if k == "exp_name":
            continue

        vals = []
        for r in all_results:
            v = r.get(k, float("nan"))

            # treat missing GS number (-1) as NaN for averaging
            if k in ("L0_gs_number", "L1_gs_number"):
                if isinstance(v, int) and v < 0:
                    v = float("nan")
                else:
                    v = float(v)  # make it numeric for mean

            # ensure numeric
            if isinstance(v, (int, float)):
                vals.append(float(v))
            else:
                vals.append(float("nan"))

        avg_row[k] = mean_ignore_missing(vals)
    # -------------------------

    output_csv = os.path.join(base_dir, "lap_results.csv")
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
        writer.writerow(avg_row)  # append average line

    print(f"Done! Results saved to {output_csv}")


if __name__ == "__main__":
    main()
