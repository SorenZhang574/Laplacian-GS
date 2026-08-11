# compare_l0_vis.py
import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_img(path: str, mode: str = "RGB") -> np.ndarray:
    img = Image.open(path).convert(mode)
    return np.asarray(img).astype(np.float32) / 255.0  # [H,W,C] in [0,1]


def save_img(arr01: np.ndarray, path: str):
    arr01 = np.clip(arr01, 0.0, 1.0)
    img = (arr01 * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img).save(path)


def to_gray(arr: np.ndarray) -> np.ndarray:
    # arr: [H,W,3] in [0,1]
    return 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]


def psnr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse < eps:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=str, required=True, help="path to L_0_vis.png (e.g., out_no_blur/L_0_vis.png)")
    parser.add_argument("--b", type=str, required=True, help="path to L_0_vis.png (e.g., out_blur/L_0_vis.png)")
    parser.add_argument("--outdir", type=str, default="cmp_out", help="output directory")
    parser.add_argument("--crop", action="store_true", help="if sizes differ, center-crop to common size")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    A = load_img(args.a, "RGB")
    B = load_img(args.b, "RGB")

    # handle size mismatch
    if A.shape != B.shape:
        if not args.crop:
            raise ValueError(f"Image sizes differ: A={A.shape}, B={B.shape}. Use --crop to center-crop.")
        Ha, Wa = A.shape[:2]
        Hb, Wb = B.shape[:2]
        H = min(Ha, Hb)
        W = min(Wa, Wb)

        def center_crop(x, H, W):
            h, w = x.shape[:2]
            y0 = (h - H) // 2
            x0 = (w - W) // 2
            return x[y0:y0+H, x0:x0+W]

        A = center_crop(A, H, W)
        B = center_crop(B, H, W)

    # metrics
    diff = B - A  # signed
    absdiff = np.abs(diff)

    mae = float(np.mean(absdiff))
    mse = float(np.mean(diff ** 2))
    p = psnr(A, B)

    # save side-by-side
    pad = np.ones((A.shape[0], 10, 3), dtype=np.float32) * 0.1
    side = np.concatenate([A, pad, B], axis=1)
    save_img(side, os.path.join(args.outdir, "side_by_side.png"))

    # abs diff visualization (scaled for visibility)
    # scale absdiff by its 99th percentile so small differences show up
    # s = np.quantile(absdiff.reshape(-1, 3), 0.99, axis=0)
    # s = np.maximum(s, 1e-6)
    # abs_vis = np.clip(absdiff / s, 0.0, 1.0)
    abs_vis = np.clip(absdiff, 0.0, 1.0)
    save_img(abs_vis, os.path.join(args.outdir, "diff_abs_vis.png"))

    # signed diff visualization: map [-s,s] -> [0,1]
    s2 = np.quantile(np.abs(diff).reshape(-1, 3), 0.99, axis=0)
    s2 = np.maximum(s2, 1e-6)
    signed_vis = np.clip(diff / s2 * 0.5 + 0.5, 0.0, 1.0)
    save_img(signed_vis, os.path.join(args.outdir, "diff_signed_vis.png"))

    # grayscale abs diff heatmap + histogram
    abs_gray = to_gray(absdiff)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(abs_gray, cmap="gray")
    plt.title("abs diff (gray)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.hist(abs_gray.flatten(), bins=80)
    plt.title("abs diff histogram (gray)")
    plt.xlabel("abs diff")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "diff_gray_and_hist.png"), dpi=200)
    plt.close()

    # write metrics
    txt = os.path.join(args.outdir, "metrics.txt")
    with open(txt, "w") as f:
        f.write(f"A: {args.a}\n")
        f.write(f"B: {args.b}\n")
        f.write(f"shape: {A.shape}\n")
        f.write(f"MAE:  {mae:.8f}\n")
        f.write(f"MSE:  {mse:.8f}\n")
        f.write(f"PSNR: {p}\n")
        f.write("Note: images are L_0_vis visualizations (normalized/clipped), so metrics compare visuals.\n")

    print(f"[OK] Saved comparison to: {args.outdir}")
    print(f"     MAE={mae:.6f}  MSE={mse:.6f}  PSNR={p}")


if __name__ == "__main__":
    main()
