# laplacian_pyramid_min.py
# Minimal Gaussian/Laplacian pyramid + frequency/spatial energy analysis.
# Usage examples:
#   python laplacian_pyramid_min.py --image xxx.jpg --outdir out_no_blur --levels 4 --no_blur_up
#   python laplacian_pyramid_min.py --image xxx.jpg --outdir out_blur    --levels 4 --blur_up

import os
import argparse
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ====================== Pyramid ======================

class GaussianLaplacianPyramid:
    """
    Minimal Gaussian + Laplacian pyramid (differentiable).
    - pyr_down: Gaussian blur + stride-2 convolution
    - pyr_up  : bilinear upsample + (optional) Gaussian blur
    """

    def __init__(self, levels: int = 4, device: str = "cuda", blur_up: bool = True):
        assert levels >= 2, "levels should be >= 2"
        self.levels = levels
        self.device = torch.device(device)
        self.blur_up = blur_up
        self.kernel = self._create_gaussian_kernel_5x5().to(self.device)  # [1,1,5,5]

    @staticmethod
    def _create_gaussian_kernel_5x5() -> torch.Tensor:
        # classic [1, 4, 6, 4, 1] outer product, normalized
        k1 = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        k2 = torch.outer(k1, k1)
        k2 = k2 / k2.sum()
        return k2[None, None, :, :]  # [1,1,5,5]

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, 1, 5, 5)  # [C,1,5,5]
        x = F.pad(x, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(x, kernel, groups=C)

    def pyr_down(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,C,ceil(H/2),ceil(W/2)] (depends on padding)
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, 1, 5, 5)
        x = F.pad(x, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(x, kernel, stride=2, groups=C)

    def pyr_up(self, x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        # x: [B,C,h,w] -> [B,C,H,W]
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if self.blur_up:
            x = self._gaussian_blur(x)
        return x

    def build_gaussian_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        img: [B,C,H,W] in [0,1]
        returns: [G0, G1, ..., G_{L-1}]
        """
        g = [img]
        cur = img
        for _ in range(self.levels - 1):
            cur = self.pyr_down(cur)
            g.append(cur)
        return g

    def build_laplacian_pyramid_from_gaussian(self, g: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        returns: [L0, L1, ..., L_{L-2}, G_{L-1}]
        """
        l = []
        for i in range(self.levels - 1):
            cur = g[i]
            nxt = g[i + 1]
            up = self.pyr_up(nxt, size=cur.shape[-2:])
            l.append(cur - up)
        l.append(g[-1])  # top level residual is last Gaussian
        return l

    def build_laplacian_pyramid(self, img: torch.Tensor) -> List[torch.Tensor]:
        g = self.build_gaussian_pyramid(img)
        return self.build_laplacian_pyramid_from_gaussian(g)

    def reconstruct_from_laplacian(self, l: List[torch.Tensor]) -> torch.Tensor:
        """
        l: [L0, ..., L_{L-2}, G_{L-1}]
        returns reconstructed image [B,C,H,W]
        """
        cur = l[-1]
        for i in reversed(range(len(l) - 1)):
            cur = self.pyr_up(cur, size=l[i].shape[-2:]) + l[i]
        return cur


# ====================== IO helpers ======================

def load_image_as_tensor(path: str, device: torch.device, max_side: int | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if max_side is not None:
        w, h = img.size
        scale = max(w, h) / max_side
        if scale > 1.0:
            img = img.resize((int(w / scale), int(h / scale)), Image.BILINEAR)

    arr = np.asarray(img).astype(np.float32) / 255.0  # [H,W,3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return t


def save_tensor_image(t: torch.Tensor, path: str, clamp01: bool = True):
    # t: [B,C,H,W] or [C,H,W]
    if t.dim() == 4:
        t = t[0]
    if clamp01:
        t = t.clamp(0, 1)
    arr = (t.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_residual_visual(l: torch.Tensor, path: str, percentile: float = 99.0):
    """
    Visualize residual (can be negative): map to [0,1] by symmetric clipping.
    """
    if l.dim() == 4:
        l = l[0]
    x = l.detach().cpu()
    absx = x.abs().flatten()
    s = torch.quantile(absx, percentile / 100.0).item()
    s = max(s, 1e-6)
    vis = (x / s) * 0.5 + 0.5
    vis = vis.clamp(0, 1)
    save_tensor_image(vis, path, clamp01=True)


# ====================== Frequency analysis ======================

def fft_mag(img: torch.Tensor) -> torch.Tensor:
    """
    img: [B,C,H,W] or [C,H,W] in [0,1]
    return: magnitude spectrum (shifted), [H,W] float
    """
    if img.dim() == 4:
        x = img[0].mean(0)
    else:
        x = img.mean(0)

    X = torch.fft.fft2(x)
    X = torch.fft.fftshift(X)
    mag = (X.real**2 + X.imag**2).sqrt()
    return mag


def radial_profile(mag: torch.Tensor) -> torch.Tensor:
    """
    mag: [H,W] (shifted)
    returns: [R+1] radial average
    """
    H, W = mag.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, device=mag.device),
        torch.arange(W, device=mag.device),
        indexing="ij",
    )
    rr = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
    rr = rr.long()
    maxr = int(rr.max().item())
    prof = torch.zeros(maxr + 1, device=mag.device)
    cnt = torch.zeros(maxr + 1, device=mag.device)
    prof.scatter_add_(0, rr.flatten(), mag.flatten())
    cnt.scatter_add_(0, rr.flatten(), torch.ones_like(mag).flatten())
    return prof / (cnt + 1e-8)


def save_frequency_fig(img: torch.Tensor, out_png: str, title: str = ""):
    """
    Save 2-panel figure:
    - log magnitude spectrum image
    - radial profile curve (log)
    """
    mag = fft_mag(img).detach().cpu()
    logmag = torch.log1p(mag)

    prof = radial_profile(mag.to(img.device)).detach().cpu()
    prof = torch.log1p(prof)

    plt.figure(figsize=(10, 4))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.imshow(logmag.numpy(), cmap="gray")
    plt.title("log(1+|FFT|)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.plot(prof.numpy())
    plt.title("log radial profile")
    plt.xlabel("radius (frequency)")
    plt.ylabel("log energy")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def fft_power(img: torch.Tensor) -> torch.Tensor:
    """
    img: [B,C,H,W] or [C,H,W]
    return: power spectrum (shifted), [H,W] float, i.e. |FFT|^2
    """
    if img.dim() == 4:
        x = img[0].mean(0)
    else:
        x = img.mean(0)

    X = torch.fft.fft2(x)
    X = torch.fft.fftshift(X)
    power = (X.real ** 2 + X.imag ** 2)
    return power


def _radius_map_hw(H: int, W: int, device: torch.device) -> torch.Tensor:
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    rr = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
    return rr


@torch.no_grad()
def freq_metrics(img: torch.Tensor, r0: int = 8) -> Dict[str, Any]:
    """
    Quantitative frequency metrics based on power spectrum |FFT|^2.
    - total_power: sum of power spectrum
    - dc_ratio: energy within radius < r0 divided by total_power
    - peak_r: argmax of radial profile for r >= r0 (ignore DC area)
    - mean_r: spectral centroid (radius-weighted average)
    """
    device = img.device
    power = fft_power(img).to(device)  # [H,W]
    H, W = power.shape
    rr = _radius_map_hw(H, W, device=device)

    total_power = power.sum().clamp_min(1e-12)
    low_power = power[rr < float(r0)].sum()
    dc_ratio = (low_power / total_power).item()

    prof = radial_profile(power)  # radial average of power
    peak_r = int(torch.argmax(prof[r0:]).item() + r0) if prof.numel() > r0 else 0

    radii = torch.arange(prof.numel(), device=device, dtype=prof.dtype)
    mean_r = (radii * prof).sum() / (prof.sum().clamp_min(1e-12))

    return {
        "H": int(H),
        "W": int(W),
        "total_power": float(total_power.item()),
        "dc_ratio_r<r0": float(dc_ratio),
        "peak_r_ge_r0": int(peak_r),
        "mean_r": float(mean_r.item()),
    }


# ====================== Spatial energy ======================

@torch.no_grad()
def spatial_energy(x: torch.Tensor) -> Dict[str, float]:
    """
    Image-domain energy stats.
    x: [B,C,H,W] or [C,H,W]
    Returns:
      - l2_sum : sum of squares (resolution-dependent)
      - l2_mean: mean of squares (resolution-invariant; recommended for compare)
      - l1_mean: mean absolute value
      - mean, var: global mean/var
    """
    t = x if x.dim() == 4 else x.unsqueeze(0)
    l2_sum = (t * t).sum().item()
    l2_mean = (t * t).mean().item()
    l1_mean = t.abs().mean().item()
    mean = t.mean().item()
    var = t.var(unbiased=False).item()
    return {
        "l2_sum": float(l2_sum),
        "l2_mean": float(l2_mean),
        "l1_mean": float(l1_mean),
        "mean": float(mean),
        "var": float(var),
    }


@torch.no_grad()
def pyramid_energy_breakdown(
    pyr: GaussianLaplacianPyramid,
    g: List[torch.Tensor],
    l: List[torch.Tensor],
    r0: int = 8,
) -> List[Dict[str, Any]]:
    """
    For each level i (0..L-2), compute energies of:
      - G_i
      - Up(G_{i+1})
      - L_i = G_i - Up(G_{i+1})
    Also include TOP = G_{L-1}
    """
    rows: List[Dict[str, Any]] = []

    for i in range(len(g) - 1):
        Gi = g[i]
        Up = pyr.pyr_up(g[i + 1], size=Gi.shape[-2:])
        Li = l[i]

        eGi = spatial_energy(Gi)
        eUp = spatial_energy(Up)
        eLi = spatial_energy(Li)

        fGi = freq_metrics(Gi, r0=r0)
        fUp = freq_metrics(Up, r0=r0)
        fLi = freq_metrics(Li, r0=r0)

        rows.append({"name": f"LV{i}_G",  **{f"sp_{k}": v for k, v in eGi.items()}, **{f"fr_{k}": v for k, v in fGi.items()}})
        rows.append({"name": f"LV{i}_UP", **{f"sp_{k}": v for k, v in eUp.items()}, **{f"fr_{k}": v for k, v in fUp.items()}})
        rows.append({"name": f"LV{i}_L",  **{f"sp_{k}": v for k, v in eLi.items()}, **{f"fr_{k}": v for k, v in fLi.items()}})

    top = l[-1]
    et = spatial_energy(top)
    ft = freq_metrics(top, r0=r0)
    rows.append({"name": "TOP", **{f"sp_{k}": v for k, v in et.items()}, **{f"fr_{k}": v for k, v in ft.items()}})

    return rows


# ====================== Main ======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="path to an image")
    parser.add_argument("--outdir", type=str, default="pyr_out", help="output directory")
    parser.add_argument("--levels", type=int, default=4, help="pyramid levels (>=2)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--max_side", type=int, default=None, help="resize if max(H,W) > max_side")
    parser.add_argument("--blur_up", action="store_true", help="enable blur after upsample")
    parser.add_argument("--no_blur_up", action="store_true", help="disable blur after upsample")
    parser.add_argument("--r0", type=int, default=8, help="low-frequency radius for dc_ratio")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    blur_up = args.blur_up and (not args.no_blur_up)

    device = torch.device(args.device)
    img = load_image_as_tensor(args.image, device=device, max_side=args.max_side)

    pyr = GaussianLaplacianPyramid(levels=args.levels, device=args.device, blur_up=blur_up)

    # Build pyramids
    g = pyr.build_gaussian_pyramid(img)
    l = pyr.build_laplacian_pyramid_from_gaussian(g)
    recon = pyr.reconstruct_from_laplacian(l)

    # Save base images
    save_tensor_image(img, os.path.join(args.outdir, "input.png"))
    save_tensor_image(recon, os.path.join(args.outdir, "recon.png"))

    err = (recon - img).abs()
    save_tensor_image(err / (err.max().clamp_min(1e-8)), os.path.join(args.outdir, "abs_error_norm.png"))

    # Reconstruction error metrics
    max_err = (recon - img).abs().max().item()
    mean_err = (recon - img).abs().mean().item()

    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"blur_up = {blur_up}\n")
        f.write(f"levels  = {args.levels}\n")
        f.write(f"r0      = {args.r0}\n")
        f.write(f"max_abs_error  = {max_err:.8e}\n")
        f.write(f"mean_abs_error = {mean_err:.8e}\n")

    # -------- Energy breakdown (G / Up(Gnext) / L) --------
    rows = pyramid_energy_breakdown(pyr, g, l, r0=args.r0)

    print("\n================ Pyramid energy breakdown ================")
    print(f"blur_up={blur_up} | r0={args.r0}")
    print("{:<10s} {:>6s} {:>6s} {:>12s} {:>12s} {:>12s} {:>8s}".format(
        "name", "H", "W", "sp_l2_mean", "sp_l1_mean", "fr_dc_ratio", "peak_r"
    ))
    print("-" * 86)
    for r in rows:
        print("{:<10s} {:>6d} {:>6d} {:>12.6e} {:>12.6e} {:>12.6f} {:>8d}".format(
            r["name"],
            int(r["fr_H"]), int(r["fr_W"]),
            float(r["sp_l2_mean"]), float(r["sp_l1_mean"]),
            float(r["fr_dc_ratio_r<r0"]),
            int(r["fr_peak_r_ge_r0"]),
        ))
    print("==========================================================\n")

    # Save CSV
    import csv
    csv_path = os.path.join(args.outdir, "pyramid_energy_breakdown.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Save each level images + frequency plots
    for i, Gi in enumerate(g):
        save_tensor_image(Gi, os.path.join(args.outdir, f"G_{i}.png"))
        save_frequency_fig(Gi, os.path.join(args.outdir, f"freq_G_{i}.png"), title=f"G_{i}")

    for i, Li in enumerate(l[:-1]):  # last one is top Gaussian
        save_residual_visual(Li, os.path.join(args.outdir, f"L_{i}_vis.png"))
        save_frequency_fig(Li, os.path.join(args.outdir, f"freq_L_{i}.png"), title=f"L_{i} (residual)")

    save_frequency_fig(l[-1], os.path.join(args.outdir, "freq_top.png"), title="Top (G_{L-1})")

    print(f"[OK] Saved results to: {args.outdir}")
    print(f"     blur_up={blur_up} max_err={max_err:.3e} mean_err={mean_err:.3e}")


if __name__ == "__main__":
    main()
