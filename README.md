# Laplacian-GS: Laplacian Frequency Hierarchies for Efficient 3D Gaussian Splatting Training

[Project Page](https://SorenZhang574.github.io/Laplacian-GS/)

Laplacian-GS is a plug-and-play acceleration scheme for 3D Gaussian Splatting training.
It combines Laplacian image decomposition with coarse-to-fine, frequency-staged optimization.

![Laplacian-GS teaser](assets/teaser_Laplacian-GS.jpg)

## 1. About Laplacian-GS

A key bottleneck in 3DGS training is the continual growth of Gaussian primitives, which increases optimization cost and slows convergence, especially at high resolutions.
Laplacian-GS fits lower-frequency structure first, archives the corresponding Gaussian field, and then trains later fields on higher-frequency residuals without carrying the full primitive burden.

At inference time, the archived fields are rendered separately and composed in the image domain through a Laplacian-style reconstruction.
This design is orthogonal to existing 3DGS acceleration methods and can be combined with backbones such as FastGS and Taming-3DGS.

This release includes the Mip-NeRF 360 1K setting reproduction scripts for both FastGS-backed and Taming-3DGS-backed Laplacian-GS.

## 2. Quick Start

Create the conda environment. The local CUDA extensions are installed from `environment.yml`.

```bash
conda env create -f environment.yml
conda activate laplacian-gs
```

Prepare the Mip-NeRF 360 dataset under `dataset/360_v2`:

```text
Laplacian-GS/
  dataset/
    360_v2/
      bicycle/
      flowers/
      garden/
      stump/
      treehill/
      room/
      counter/
      kitchen/
      bonsai/
```

The scripts use `./dataset` by default. You can either place the dataset there or create a symlink:

```bash
ln -s /path/to/your/dataset ./dataset
```

The default image loading behavior follows the 3DGS convention: images wider than 1.6K pixels are resized down automatically unless you override `-r`.

## 3. Training & Evaluation

Run the FastGS-backed version:

```bash
bash run_mipnerf360_1k_fastgs.sh
```

Run the Taming-3DGS-backed version:

```bash
bash run_mipnerf360_1k_taming.sh
```

Both scripts train `train_lap.py` with:

- `--lap_n_levels 2`
- `--decomposition_func pn`
- `--blur_up`
- `--inherit_state only_xyz`
- `--iter_plan 10000_20000`

The FastGS script uses `--backbone fastgs`; the Taming-3DGS script uses `--backbone taming`.

## 4. Output Layout:

- FastGS training outputs: `output_mipnerf360_1k_fastgs/<scene>_big/`
- Taming-3DGS training outputs: `output_mipnerf360_1k_taming/<scene>/`
- reconstructed renders: `.../Reconstructed/`
- per-stage Laplacian outputs: `.../L1`, `.../L0`
- per-scene metrics: `laplacian_metrics_summary.csv`
- all-scene summary: `lap_results.csv`
