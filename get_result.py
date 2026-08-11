import os
import json
import csv
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Summarize laplacian metrics.")
parser.add_argument("--base_dir", type=str, default="./output/", help="Base directory to process")
args = parser.parse_args()

# Base directory
base_dir = args.base_dir

# Target folders to process
folders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
folders.sort()
# folders = [
#     "bicycle",
#     "bicycle_laplacian_g3",
#     "bicycle_laplacian_l0",
#     "bicycle_laplacian_l1",
#     "bicycle_laplacian_l2",
# ]

# Path to save the final CSV file
output_csv = os.path.join(base_dir, "laplacian_metrics_summary.csv")

# Prepare data rows for CSV
rows = [["Exp", "SSIM", "PSNR", "LPIPS", "Time (min)", "GS Number"]]

for folder in folders:
    folder_path = os.path.join(base_dir, folder)

    # Initialize default values
    ssim = psnr = lpips = time_min = gs_number = "N/A"

    # Read results.json for metrics
    results_path = os.path.join(folder_path, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
            metric = list(results.values())[0]
            ssim = metric.get("SSIM", "N/A")
            psnr = metric.get("PSNR", "N/A")
            lpips = metric.get("LPIPS", "N/A")

    # Read training info from TRAIN_INFO
    train_info_path = os.path.join(folder_path, "TRAIN_INFO")
    if os.path.exists(train_info_path):
        with open(train_info_path, "r") as f:
            for line in f:
                if "Time" in line:
                    parts = line.strip().split(",")
                    for part in parts:
                        if "minute" in part:
                            time_min = float(part.strip().split()[0])
                if "GS Number" in line:
                    gs_number = int(line.strip().split(":")[-1])

    # Append collected data to CSV rows
    rows.append([folder, ssim, psnr, lpips, time_min, gs_number])

# Write the final CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Saved summary to {output_csv}")
