#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import os, random, time
from random import randint
from lpipsPyTorch import lpips
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim

import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.fast_lap_utils import compute_gaussian_score_fastgs, sampling_cameras
from laplacian.laplacian_torch import LaplacePyramid
import json
from torch import nn
from utils.schedule_utils import TrainingScheduler
from FastLanczos import lanczos_resample

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    raise ImportError("SparseGaussianAdam not available.")
    SPARSE_ADAM_AVAILABLE = False

from utils.prefetch_utils import DictPrefetcher, RandomCamSampler
from utils.runtime_profiler import RuntimeProfiler
from collections import deque


# ============================================
# Helper Functions
# ============================================

def clone_gaussian(gaussians: GaussianModel) -> GaussianModel:
    """
    Create a fully independent snapshot of a GaussianModel instance.
    
    The snapshot contains:
        - All Gaussian parameters (xyz, SH coefficients, scaling, rotation, opacity)
        - No optimizer / scheduler / gradients
        - Detached, cloned tensors (so training of the original model will NOT affect the snapshot)
    
    The returned GaussianModel can be used for evaluation, reporting, or
    combining results with the ongoing training model.
    """
    # Create a new GaussianModel with the same SH degree
    new = GaussianModel(gaussians.max_sh_degree, optimizer_type=gaussians.optimizer_type)

    # Copy simple scalar attributes
    new.active_sh_degree = gaussians.active_sh_degree
    new.spatial_lr_scale = gaussians.spatial_lr_scale

    device = gaussians._xyz.device  # Keep the same device (usually CUDA)

    # Deep-copy all Gaussian parameter tensors
    new._xyz = nn.Parameter(gaussians._xyz.detach().clone().to(device), requires_grad=False)
    new._features_dc = nn.Parameter(gaussians._features_dc.detach().clone().to(device), requires_grad=False)
    new._features_rest = nn.Parameter(gaussians._features_rest.detach().clone().to(device), requires_grad=False)
    new._scaling = nn.Parameter(gaussians._scaling.detach().clone().to(device), requires_grad=False)
    new._rotation = nn.Parameter(gaussians._rotation.detach().clone().to(device), requires_grad=False)
    new._opacity = nn.Parameter(gaussians._opacity.detach().clone().to(device), requires_grad=False)

    # Optional: copy radii buffer if present
    if gaussians.max_radii2D is not None and gaussians.max_radii2D.numel() > 0:
        new.max_radii2D = gaussians.max_radii2D.detach().clone().to(device)
    else:
        new.max_radii2D = torch.empty(0, device=device)

    # Snapshot should not contain optimizers or gradient accumulators
    new.optimizer = None
    new.shoptimizer = None
    new.xyz_gradient_accum = torch.empty(0, device=device)
    new.xyz_gradient_accum_abs = torch.empty(0, device=device)
    new.denom = torch.empty(0, device=device)

    # Optional: copy temporary radii if it exists
    if hasattr(gaussians, "tmp_radii"):
        if gaussians.tmp_radii is not None and gaussians.tmp_radii.numel() > 0:
            new.tmp_radii = gaussians.tmp_radii.detach().clone().to(device)
        else:
            new.tmp_radii = None

    return new



def save_gs(save_path, iteration, gs_field):
    point_cloud_path = os.path.join(save_path, "point_cloud/iteration_{}".format(iteration))
    gs_field.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    if hasattr(gs_field, "exposure_mapping"):
        exposure_dict = {
            image_name: gs_field.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in gs_field.exposure_mapping
        }

        with open(os.path.join(save_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)


def _equidistant_indices(N: int, k: int, offset_elems: int = 0, device=None):
    """
    Equidistant sampling with a fixed element offset.
    - N           : total number of points
    - k           : number of points to sample
    - offset_elems: integer offset in [0, stride), e.g. 0 for positive set,
                    stride//2 for negative set in the first split
    """
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    stride = max(1, N // k)
    start = offset_elems % stride
    indices = start + torch.arange(k, device=device) * stride
    indices = torch.clamp(indices, max=N - 1)
    indices = torch.unique(indices, sorted=True)

    return indices



# ============================================
# Version A: Param-only clone + fresh optimizer
# ============================================

def extract_sub_gaussians_reinit(prev_g, ratio: float, opt, offset_elems: int = 0):
    """
    Param-only sub-sampling:
    - Copy only the core per-point tensors and minimal meta before training_setup(opt).
    - Optimizer/schedulers/buffers are re-created by training_setup.
    """
    ratio = float(max(0.0, min(1.0, ratio)))
    N = prev_g.get_xyz.shape[0]
    if N == 0:
        raise ValueError("Previous Gaussian model has no points.")
    k = max(1, int(N * ratio))
    device = prev_g.get_xyz.device
    idx = _equidistant_indices(N, k, offset_elems, device)

    # New model (same class / SH degree / optimizer type)
    new_g = prev_g.__class__(prev_g.max_sh_degree, prev_g.optimizer_type)

    # Helper to make fresh leaf tensors
    def _param(x): 
        return nn.Parameter(x.detach().clone().requires_grad_(True))

    # ---- Core per-point parameters (must exist before training_setup) ----
    new_g._xyz           = _param(prev_g._xyz[idx])
    new_g._features_dc   = _param(prev_g._features_dc[idx])
    new_g._features_rest = _param(prev_g._features_rest[idx])
    new_g._scaling       = _param(prev_g._scaling[idx])
    new_g._rotation      = _param(prev_g._rotation[idx])
    new_g._opacity       = _param(prev_g._opacity[idx])

    # ---- Minimal meta needed by training_setup / runtime ----
    # SH degrees & optimizer type (keep behavior consistent)
    new_g.active_sh_degree = prev_g.active_sh_degree
    new_g.max_sh_degree    = prev_g.max_sh_degree
    new_g.optimizer_type   = prev_g.optimizer_type

    # VERY IMPORTANT: spatial_lr_scale used to build LR schedule
    new_g.spatial_lr_scale = getattr(prev_g, "spatial_lr_scale", 1.0)

    # Exposure mapping/tensor must exist before training_setup (exposure optimizer needs it)
    new_g.exposure_mapping     = getattr(prev_g, "exposure_mapping", {})
    new_g.pretrained_exposures = getattr(prev_g, "pretrained_exposures", None)
    if hasattr(prev_g, "_exposure") and prev_g._exposure is not None:
        new_g._exposure = _param(prev_g._exposure)
    else:
        # Fresh exposure if prev_g didn't have one (shape depends on #cameras)
        num_cams = len(new_g.exposure_mapping) if new_g.exposure_mapping else 1
        eye = torch.eye(3, 4, device=device)[None].repeat(num_cams, 1, 1)
        new_g._exposure = nn.Parameter(eye.requires_grad_(True))

    # Helpful for densify/prune (zeros ok)
    new_g.max_radii2D = torch.zeros((k,), device=device)

    # Let training_setup create: xyz_gradient_accum, denom, optimizers, schedulers, etc.
    new_g.training_setup(opt)
    return new_g

# ============================================
# Version B: Capture/restore + prune (keep optimizer state)
# ============================================

def extract_sub_gaussians_with_state(prev_g, ratio: float, opt, offset_elems: int = 0):
    """
    Create a sub-Gaussian model while preserving the optimizer state for retained points.
    1. Fully clone model via capture()/restore().
    2. Equidistant sample indices to keep.
    3. Use prune_points() to remove other points and prune optimizer state.
    4. Exposure optimizer state is NOT preserved (only the tensor values are copied).
    """
    ratio = float(max(0.0, min(1.0, ratio)))
    N = prev_g.get_xyz.shape[0]
    if N == 0:
        raise ValueError("Previous Gaussian model has no points.")

    k = max(1, int(N * ratio))
    device = prev_g.get_xyz.device
    keep_idx = _equidistant_indices(N, k, offset_elems, device)

    remove_mask = torch.ones((N,), dtype=torch.bool, device=device)
    remove_mask[keep_idx] = False

    new_g = prev_g.__class__(prev_g.max_sh_degree, prev_g.optimizer_type)
    captured = prev_g.capture()
    if hasattr(prev_g, "_exposure"):
        new_g._exposure = nn.Parameter(prev_g._exposure.detach().clone().requires_grad_(True))
        new_g.exposure_optimizer = torch.optim.Adam([new_g._exposure])
        new_g.pretrained_exposures = getattr(prev_g, "pretrained_exposures", None)
        new_g.exposure_mapping = getattr(prev_g, "exposure_mapping", None)
    new_g.restore(captured, opt)

    new_g.prune_points_inherit(remove_mask)
    return new_g



def extract_sub_gaussians_only_xyz(prev_g, ratio: float, opt, offset_elems: int = 0, use_max_sh_degree=-1):
    """
    Create a sub-Gaussian model while preserving the optimizer state for retained points.
    1. Fully clone model via capture()/restore().
    2. Equidistant sample indices to keep.
    3. Use prune_points() to remove other points and prune optimizer state.
    4. Exposure optimizer state is NOT preserved (only the tensor values are copied).
    """
    ratio = float(max(0.0, min(1.0, ratio)))
    N = prev_g.get_xyz.shape[0]
    if N == 0:
        raise ValueError("Previous Gaussian model has no points.")
    k = max(1, int(N * ratio))
    device = prev_g.get_xyz.device
    idx = _equidistant_indices(N, k, offset_elems, device)

    new_xyz = prev_g._xyz[idx].detach().clone()

    max_sh_degree = prev_g.max_sh_degree if use_max_sh_degree < 0 else use_max_sh_degree
    new_g = prev_g.__class__(max_sh_degree, prev_g.optimizer_type)

    exposure_mapping = prev_g.exposure_mapping if hasattr(prev_g, "exposure_mapping") else None
    new_g.create_from_xyz(new_xyz, exposure_mapping, prev_g.spatial_lr_scale)
    new_g.training_setup(opt)

    return new_g




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, runtime_breakdown=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    
    if pipe.backbone == "fastgs":
        from gaussian_renderer import render_fastgs as render_fn, network_gui_ws as network_gui_fn
    elif pipe.backbone == "taming":
        from gaussian_renderer_dash import render as render_fn, network_gui as network_gui_fn
    else:
        raise ValueError(f"Unknown backbone: {pipe.backbone}")
    
    if(args.websockets):
        network_gui_fn.init(args.ip, args.port)

    pyramid_func = LaplacePyramid(
        levels=pipe.lap_n_levels,
        device="cuda",
        decom_func_name=pipe.decomposition_func,
        blur_up=pipe.blur_up,
    )
    num_levels = pyramid_func.num_levels
    num_stages = pyramid_func.num_stages

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    raw_iter_plan = pipe.iter_plan
    runtime_profiler = RuntimeProfiler() if runtime_breakdown else None
    runtime_config = None
    if runtime_profiler is not None:
        if raw_iter_plan:
            planned_stage_iterations = [int(value) for value in raw_iter_plan.split("_")]
        else:
            remaining_iterations = opt.iterations
            planned_stage_iterations = []
            for _ in range(num_stages):
                remaining_iterations = int(remaining_iterations / num_stages)
                planned_stage_iterations.append(remaining_iterations)
        runtime_config = {
            "iter_plan": raw_iter_plan,
            "planned_stage_iterations": planned_stage_iterations,
            "total_planned_iterations": sum(planned_stage_iterations),
            "lap_n_levels": pipe.lap_n_levels,
            "decomposition_func": pipe.decomposition_func,
            "blur_up": pipe.blur_up,
            "inherit_state": pipe.inherit_state,
            "inherit_ratio": pipe.inherit_ratio,
            "optimizer_type": opt.optimizer_type,
            "densification_interval": opt.densification_interval,
            "pyramid_device": pipe.pyramid_device,
            "prefetch_lookahead": pipe.prefetch_lookahead,
        }

    def runtime_event_pair():
        if runtime_profiler is None:
            return None
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    runtime_component_events = {
        "data_fetch": runtime_event_pair(),
        "forward": runtime_event_pair(),
        "composition_loss": runtime_event_pair(),
        "backward": runtime_event_pair(),
        "densification_pruning": runtime_event_pair(),
        "optimizer": runtime_event_pair(),
    }
    runtime_trigger_events = {
        "densify": runtime_event_pair(),
        "opacity_reset": runtime_event_pair(),
        "final_prune": runtime_event_pair(),
    }
    # gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    # load gt
    gt_gaussian_pyr_dict = {}
    viewpoint_all = scene.getTrainCameras().copy()
    pyr_dev = pipe.pyramid_device   # cpu or cuda
    dataset_dev = dataset.data_device
    if runtime_profiler is not None:
        torch.cuda.synchronize()
        pyramid_precompute_start = time.perf_counter()
    for viewpoint_cam in viewpoint_all:
        if dataset_dev == "cpu":
            img_cpu = viewpoint_cam.original_image
            img_cpu_4d = img_cpu.unsqueeze(0)
            img_gpu = img_cpu_4d.to("cuda", non_blocking=True)
            pyr_gpu = pyramid_func.build_pyr_gaussian(img_gpu)
            pyr_cpu = [img_cpu_4d] + [lvl.detach().to(device=pyr_dev) for lvl in pyr_gpu[1:]]
            gt_gaussian_pyr_dict[viewpoint_cam.uid] = pyr_cpu
            del img_gpu, pyr_gpu
        else:
            pyr_gpu = pyramid_func.build_pyr_gaussian(viewpoint_cam.original_image.unsqueeze(0))
            gt_gaussian_pyr_dict[viewpoint_cam.uid] = [pyr_gpu[0]] + [lvl.detach().to(device=pyr_dev) for lvl in pyr_gpu[1:]]
            del pyr_gpu
    pyramid_func.pyr_size_list = [x.shape[-2:] for x in gt_gaussian_pyr_dict[viewpoint_all[0].uid]]
    if runtime_profiler is not None:
        torch.cuda.synchronize()
        pyramid_precompute_end = time.perf_counter()
        runtime_profiler.add_one_off_seconds(
            "pyramid_precompute",
            pyramid_precompute_end - pyramid_precompute_start,
        )
    pred_accumulated_dict = {}
    gt_image_dict = {}
    # iter plan
    if pipe.iter_plan != "":
        pipe.iter_plan = pipe.iter_plan.split("_")
        ori_iterations = opt.iterations
        ori_densify_until_iter = opt.densify_until_iter
        ori_position_lr_max_steps = opt.position_lr_max_steps

    ema_loss_for_log = 0.0
    
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    img_num = -1

    if pipe.do_training_report:
        log_it = 1
        gaussians_report_list = [None] * num_stages

        def make_render_adapter(render_fn, pipe, bg, mult, dataset, separate_sh):
            def adapter(viewpoint_cam, gaussians, *unused_args, render_size=None, pos_and_neg=False):
                if pipe.backbone == "fastgs":
                    return render_fn(
                        viewpoint_cam, gaussians, pipe, bg, mult, render_size=render_size, pos_and_neg=pos_and_neg,
                    )
                elif pipe.backbone == "taming":
                    return render_fn(
                        viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=separate_sh, render_size=render_size, pos_and_neg=pos_and_neg,
                    )
                else:
                    raise ValueError(f"Unknown backbone: {pipe.backbone}")
            return adapter
        render_adapter = make_render_adapter(
            render_fn=render_fn,
            pipe=pipe,
            bg=bg,
            mult=opt.mult,
            dataset=dataset,
            separate_sh=SPARSE_ADAM_AVAILABLE,
        )

    
    # ---- peak stats (per-stage) ----
    peak_alloc = 0
    peak_reserved = 0
    peak_gs = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    for i_stage in range(num_stages):
        # rescale the number of iterations according to the number of stages.
        if pipe.iter_plan != "":
            opt.iterations = int(pipe.iter_plan[i_stage])
            opt.save_iterations = [opt.iterations]
            opt.densify_until_iter = int(ori_densify_until_iter * (opt.iterations / ori_iterations))
            opt.position_lr_max_steps = opt.iterations
        else:
            opt.iterations = int(opt.iterations / num_stages)
            opt.save_iterations = [opt.iterations]
            opt.densify_until_iter = int(opt.densify_until_iter / num_stages)
            opt.position_lr_max_steps = int(opt.position_lr_max_steps / num_stages)
        
        # change the hyper-parameters for different stages.
        if i_stage >= 1:
            opt.dense = opt.dense * pipe.decay_dense
            opt.loss_thresh = opt.loss_thresh * pipe.decay_loss_thresh

        # calculate the gt.
        with torch.no_grad():
            if i_stage == 0:
                for key in gt_gaussian_pyr_dict:
                    gt_image_dict[key] = gt_gaussian_pyr_dict[key].pop()
                image_size_last_stage = next(iter(gt_image_dict.values())).shape

                save_sub_folder = f"L{num_levels-1}"
            else:
                if runtime_profiler is not None:
                    torch.cuda.synchronize()
                    stage_cache_build_start = time.perf_counter()
                bg = background
                viewpoint_all = scene.getTrainCameras().copy()
                for viewpoint_cam in viewpoint_all:
                    key = viewpoint_cam.uid
                    if i_stage == 1:
                        if pipe.backbone == "fastgs":
                            image_current = render_fn(viewpoint_cam, gaussians, pipe, bg, opt.mult, render_size=image_size_last_stage[-2:])["render"]
                        elif pipe.backbone == "taming":
                            image_current = render_fn(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=image_size_last_stage[-2:])["render"]
                        else:
                            raise ValueError(f"Unknown backbone: {pipe.backbone}")
                        image_current = image_current.unsqueeze(0)
                    else:
                        
                        if pipe.backbone == "fastgs":
                            render_pred = render_fn(viewpoint_cam, gaussians, pipe, bg, opt.mult, render_size=image_size_last_stage[-2:], pos_and_neg=True)["render"]
                        elif pipe.backbone == "taming":
                            render_pred = render_fn(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=image_size_last_stage[-2:],
                                                 pos_and_neg=True)["render"]
                        else:
                            raise ValueError(f"Unknown backbone: {pipe.backbone}")
                        render_pred = render_pred.unsqueeze(0)
                        # merge with previous prediction
                        pred_prev = pred_accumulated_dict[key]
                        if pyr_dev == "cpu":
                            pred_prev = pred_prev.to("cuda", non_blocking=True)
                        image_current = pred_prev + render_pred
                    # use the gaussian pyramid as the gt. 
                    gt_image_dict[key] = gt_gaussian_pyr_dict[key].pop()
                    pred_accumulated_dict[key] = pyramid_func.get_upsampled([gt_image_dict[key], image_current])
                    if pyr_dev == "cpu":
                        pred_accumulated_dict[key] = pred_accumulated_dict[key].to("cpu", non_blocking=True)
                image_size_last_stage = next(iter(gt_image_dict.values())).shape
                if runtime_profiler is not None:
                    torch.cuda.synchronize()
                    stage_cache_build_end = time.perf_counter()
                    runtime_profiler.add_one_off_seconds(
                        "stage_cache_build",
                        stage_cache_build_end - stage_cache_build_start,
                    )

                save_sub_folder = f"L{num_levels-1-i_stage}"
            os.makedirs(os.path.join(scene.model_path, save_sub_folder), exist_ok=True)

        # redefine the gaussians model for the next stage
        if runtime_profiler is not None and i_stage > 0:
            torch.cuda.synchronize()
            inheritance_start = time.perf_counter()
        if pipe.inherit_state == "keep_and_adc":
            if i_stage == 0:
                gaussians.training_setup(opt)
                # First stage — nothing to do
            else:
                size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                my_viewpoint_stack = scene.getTrainCameras().copy()
                # camlist = sampling_cameras(my_viewpoint_stack)
                camlist = my_viewpoint_stack

                # The multiview consistent densification of fastgs
                importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True, gt_image_dict=gt_image_dict, pred_accumulated_dict=pred_accumulated_dict, pos_and_neg=(i_stage>0),
                                                                                inherit_cal=True)                    
                # scores = importance_score.float()
                # threshold = scores.quantile(0.6)
                # mask = scores >= threshold
                importance_score_thres = len(camlist) * pipe.inherit_importance_ratio
                prune_mask = importance_score <= importance_score_thres
                gaussians.prune_points(prune_mask)

                # reset color as 0.0, fill 0.0
                gaussians._features_dc   = nn.Parameter(torch.zeros_like(gaussians._features_dc))
                gaussians._features_rest = nn.Parameter(torch.zeros_like(gaussians._features_rest))
                
                # gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                #                             min_opacity = 0.005, 
                #                             extent = scene.cameras_extent, 
                #                             radii=radii,
                #                             args = opt,
                #                             importance_score = importance_score,
                #                             pruning_score = pruning_score)
        else:
            ratio = getattr(pipe, "inherit_ratio")
            if i_stage == 0:
                gaussians.training_setup(opt)
                # First stage — nothing to do

            elif i_stage >= 1:
                g_prev = gaussians

                if pipe.inherit_state == "only_xyz":
                    g_new = extract_sub_gaussians_only_xyz(g_prev, ratio, opt, offset_elems=0, use_max_sh_degree=pipe.inherit_max_sh_degree)
                elif pipe.inherit_state == "with_state":  # ✅ preserve optimizer state
                    g_new = extract_sub_gaussians_with_state(g_prev, ratio, opt, offset_elems=0)
                else:  # ✅ only copy params, new optimizer
                    g_new = extract_sub_gaussians_reinit(g_prev, ratio, opt, offset_elems=0)

                gaussians = g_new
                scene.gaussians = gaussians
                # free g_prev
                del g_prev
                torch.cuda.empty_cache()

        if runtime_profiler is not None and i_stage > 0:
            torch.cuda.synchronize()
            inheritance_end = time.perf_counter()
            runtime_profiler.add_one_off_seconds(
                "inheritance",
                inheritance_end - inheritance_start,
            )

        if pipe.backbone == "taming":
            scheduler = TrainingScheduler(opt, pipe, gaussians, 
                                    [gt_image_dict[cam.uid] for cam in scene.getTrainCameras()])
            if i_stage >= 1:
                # the following stage should use different max_n_gaussian and momentum settings.
                scheduler.max_n_gaussian = scheduler.init_n_gaussian
                scheduler.momentum = 0
            render_scale = scheduler.get_res_scale(1)
        
        use_prefetch = (pipe.prefetch_lookahead >= 1 and (dataset_dev == "cpu" or pyr_dev == "cpu"))
        if use_prefetch:
            cam_queue = deque()
            sampler = RandomCamSampler(scene.getTrainCameras())
            prefetcher = DictPrefetcher(device="cuda", lookahead=pipe.prefetch_lookahead, pin_cpu=False)

            for _ in range(pipe.prefetch_lookahead):
                cam = sampler.next()
                cam_queue.append(cam)
                prefetcher.schedule(
                    uid=cam.uid,
                    gt_image_dict=gt_image_dict,
                    pred_accumulated_dict=pred_accumulated_dict,
                    need_pred=(i_stage >= 1),
                )
        else:
            sampler = None
            prefetcher = None
        
        progress_bar = tqdm(range(0, opt.iterations), desc="Training progress")
        first_iter = 1
        for iteration in range(first_iter, opt.iterations + 1):

            if websockets:
                if network_gui_fn.curr_id >= 0 and network_gui_fn.curr_id < len(scene.getTrainCameras()):
                    cam = scene.getTrainCameras()[network_gui_fn.curr_id]
                    net_image = render_fn(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                    network_gui_fn.latest_width = cam.image_width
                    network_gui_fn.latest_height = cam.image_height
                    network_gui_fn.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

            iter_start.record()
            
            lr_xyz = gaussians.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

            if runtime_profiler is not None:
                runtime_component_events["data_fetch"][0].record()

            # ---- Pick a camera + fetch data (prefetch-aware) ----
            if use_prefetch:
                # 1) Pop the camera for the current iteration (already prefetched)
                viewpoint_cam = cam_queue.popleft()
                uid = viewpoint_cam.uid
                # 2) Push one more camera to the tail and schedule prefetch
                #    to keep the queue "lookahead" steps ahead
                cam_next = sampler.next()
                cam_queue.append(cam_next)
                prefetcher.schedule(
                    uid=cam_next.uid,
                    gt_image_dict=gt_image_dict,
                    pred_accumulated_dict=pred_accumulated_dict,
                    need_pred=(i_stage >= 1),
                )
                # 3) Fetch GT / accumulated prediction on GPU (waits on the prefetch event)
                gt_4d, pred_4d = prefetcher.fetch(uid)
                gt_image = gt_4d.squeeze(0)  # (C,H,W) on CUDA

                if i_stage >= 1:
                    previous_accumulated = pred_4d.squeeze(0)  # (C,H,W) on CUDA
            else:
                # Original logic (prefetch disabled or data already on GPU)
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_indices = list(range(len(viewpoint_stack)))
                    if img_num == -1:
                        img_num = len(viewpoint_stack)
                rand_idx = randint(0, len(viewpoint_indices) - 1)
                viewpoint_cam = viewpoint_stack.pop(rand_idx)
                _ = viewpoint_indices.pop(rand_idx)
                # get gt_image
                gt_image = gt_image_dict[viewpoint_cam.uid].squeeze(0)
                # get previous accumulated image if need.
                if i_stage >=1:
                    previous_accumulated = pred_accumulated_dict[viewpoint_cam.uid].squeeze(0)

            if runtime_profiler is not None:
                runtime_component_events["data_fetch"][1].record()

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            if pipe.backbone == "fastgs":
                if runtime_profiler is not None:
                    runtime_component_events["forward"][0].record()
                render_pkg = render_fn(viewpoint_cam, gaussians, pipe, bg, opt.mult, render_size=gt_image.shape[-2:], pos_and_neg=(i_stage>0))
            elif pipe.backbone == "taming":
                if render_scale > 1:
                    gt_image = lanczos_resample(gt_image.permute(1, 2, 0), scale_factor=render_scale).permute(2, 0, 1)
                if runtime_profiler is not None:
                    runtime_component_events["forward"][0].record()
                render_pkg = render_fn(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_image.shape[-2:], pos_and_neg=(i_stage>0))
            else:
                raise ValueError(f"Unknown backbone: {pipe.backbone}")
            
            if runtime_profiler is not None:
                runtime_component_events["forward"][1].record()

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            if runtime_profiler is not None:
                runtime_component_events["composition_loss"][0].record()

            # get the previous accumulated image
            if i_stage >=1:
                image = image + previous_accumulated

            # Loss
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            if runtime_profiler is not None:
                runtime_component_events["composition_loss"][1].record()
                runtime_component_events["backward"][0].record()
            loss.backward()
            if runtime_profiler is not None:
                runtime_component_events["backward"][1].record()

            iter_end.record()

            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    # progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.set_postfix_str(f"Loss={ema_loss_for_log:.7f}, GS={gaussians.get_xyz.shape[0]}")
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                iter_time = iter_start.elapsed_time(iter_end)
                # Log and save
                if pipe.do_training_report:
                    gaussians_report_list[num_levels-1-i_stage] = gaussians
                    training_report(tb_writer, log_it, Ll1, loss, l1_loss, lr_xyz, iter_time, testing_iterations, scene, gaussians_report_list, pyramid_func, render_adapter)
                    log_it += 1

                optim_start.record()
                
                if runtime_profiler is not None:
                    runtime_active_triggers = []
                    runtime_component_events["densification_pruning"][0].record()

                # Densification
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        if runtime_profiler is not None:
                            runtime_trigger_events["densify"][0].record()
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        if pipe.backbone == "fastgs":
                            my_viewpoint_stack = scene.getTrainCameras().copy()
                            camlist = sampling_cameras(my_viewpoint_stack)

                            # The multiview consistent densification of fastgs
                            importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True, gt_image_dict=gt_image_dict, pred_accumulated_dict=pred_accumulated_dict, pos_and_neg=(i_stage>0))                    
                            gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold, 
                                                        min_opacity = 0.005, 
                                                        extent = scene.cameras_extent, 
                                                        radii=radii,
                                                        args = opt,
                                                        importance_score = importance_score,
                                                        pruning_score = pruning_score)
                        elif pipe.backbone == "taming":
                            # Apply DashGaussian primitive scheduler to control densification.
                            densify_rate = scheduler.get_densify_rate(iteration, gaussians.get_xyz.shape[0], render_scale)
                            momentum_add = gaussians.prune_and_densify_dash(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 
                                                                    size_threshold, radii, densify_rate=densify_rate)
                            # Update max_n_gaussian
                            scheduler.update_momentum(momentum_add)
                            # Update render scale based on the DashGaussian resolution scheduler. 
                            render_scale = scheduler.get_res_scale(iteration)
                        else:
                            raise ValueError(f"Unknown backbone: {pipe.backbone}")
                        
                        if runtime_profiler is not None:
                            runtime_trigger_events["densify"][1].record()
                            runtime_active_triggers.append("densify")

                        # ---- update peaks ----
                        peak_gs = max(peak_gs, gaussians.get_xyz.shape[0])
                        if torch.cuda.is_available():
                            peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
                            peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())

                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        if runtime_profiler is not None:
                            runtime_trigger_events["opacity_reset"][0].record()
                        gaussians.reset_opacity()
                        if runtime_profiler is not None:
                            runtime_trigger_events["opacity_reset"][1].record()
                            runtime_active_triggers.append("opacity_reset")

                # The multiview consistent pruning of fastgs. We do it every 3k iterations after 15k
                # In this stage, the model converge basically. So we can prune more aggressively without degrading rendering quality.
                # You can check the rendering results of 20K iterations in arxiv version (https://arxiv.org/abs/2511.04283), the rendering quality is already very good.
                if pipe.backbone == "fastgs" and not pipe.no_vcp:
                    # if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                    if iteration % int(opt.iterations*0.1) == 0 and iteration > int(opt.iterations*0.5) and iteration < opt.iterations:
                        if runtime_profiler is not None:
                            runtime_trigger_events["final_prune"][0].record()
                    # if iteration % int(opt.iterations*0.2) == 0 and iteration > int(opt.iterations*0.5) and iteration < opt.iterations:
                        my_viewpoint_stack = scene.getTrainCameras().copy()
                        camlist = sampling_cameras(my_viewpoint_stack)

                        _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, gt_image_dict=gt_image_dict, pred_accumulated_dict=pred_accumulated_dict, pos_and_neg=(i_stage>0))                    
                        gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)
                        if runtime_profiler is not None:
                            runtime_trigger_events["final_prune"][1].record()
                            runtime_active_triggers.append("final_prune")
            
                if runtime_profiler is not None:
                    runtime_component_events["densification_pruning"][1].record()
                    runtime_component_events["optimizer"][0].record()

                # Optimization step
                if iteration < opt.iterations:   # a small bug: the last iteration has no optimizer_step
                    if pipe.backbone == "fastgs":
                        if opt.optimizer_type == "default":
                            gaussians.optimizer_step(iteration)
                        elif opt.optimizer_type == "sparse_adam":
                            visible = radii > 0
                            gaussians.optimizer.step(visible, radii.shape[0])
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    elif pipe.backbone == "taming":
                        if opt.optimizer_type == "sparse_adam":
                            visible = radii > 0
                            gaussians.optimizer.step(visible, radii.shape[0])
                            gaussians.optimizer.zero_grad(set_to_none = True)
                        else:
                            gaussians.optimizer.step()
                            gaussians.optimizer.zero_grad(set_to_none = True)
                    else:
                        raise ValueError(f"Unknown backbone: {pipe.backbone}")

                if runtime_profiler is not None:
                    runtime_component_events["optimizer"][1].record()

                # record time
                optim_end.record()
                torch.cuda.synchronize()
                if runtime_profiler is not None:
                    for component_name, (component_start, component_end) in runtime_component_events.items():
                        runtime_profiler.add_component_seconds(
                            component_name,
                            component_start.elapsed_time(component_end) / 1e3,
                        )
                    for trigger_name in runtime_active_triggers:
                        trigger_start, trigger_end = runtime_trigger_events[trigger_name]
                        runtime_profiler.add_triggered_seconds(
                            trigger_start.elapsed_time(trigger_end) / 1e3,
                        )
                    runtime_profiler.add_iteration(save_sub_folder)
                optim_time = optim_start.elapsed_time(optim_end)
                total_time += (iter_time + optim_time) / 1e3
                                
                if (iteration in opt.save_iterations):
                    print("\n[ITER {}] [{}] Saving Gaussians".format(iteration, save_sub_folder))
                    # ---- update peaks ----
                    if torch.cuda.is_available():
                        peak_alloc = max(peak_alloc, torch.cuda.max_memory_allocated())
                        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
                    if runtime_profiler is not None:
                        torch.cuda.synchronize()
                        archive_start = time.perf_counter()
                    save_gs(os.path.join(scene.model_path, save_sub_folder), iteration, gaussians)
                    if runtime_profiler is not None:
                        torch.cuda.synchronize()
                        archive_end = time.perf_counter()
                        runtime_profiler.add_one_off_seconds(
                            "archive_io",
                            archive_end - archive_start,
                        )
                    if pipe.do_training_report:
                        gaussians_report_list[num_levels-1-i_stage] = clone_gaussian(gaussians)


        # scene.save(iteration)
        print(f"Gaussian number: {gaussians._xyz.shape[0]}")
        print(f"Training time: {total_time}")
        print(f"Peak GS number: {peak_gs}")
        print(f"Peak GPU allocated: {peak_alloc/1024**2:.2f} MB")
        print(f"Peak GPU reserved : {peak_reserved/1024**2:.2f} MB")
        with open(os.path.join(os.path.join(scene.model_path, save_sub_folder), "TRAIN_INFO"), "w+") as f:
            f.write("Training Time: {:.2f} seconds, {:.2f} minutes\n".format(total_time, total_time / 60.))
            f.write("GS Number: {}\n".format(gaussians._xyz.shape[0]))
            f.write(f"Peak GS number: {peak_gs}\n")
            f.write(f"Peak GPU allocated: {peak_alloc/1024**2:.2f} MB\n")
            f.write(f"Peak GPU reserved : {peak_reserved/1024**2:.2f} MB\n")


    if runtime_profiler is not None:
        runtime_profiler.set_optimization_time(total_time)
        gpu_index = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(gpu_index)
        runtime_metadata = {
            "scene": os.path.basename(os.path.normpath(dataset.source_path)),
            "source": dataset.source_path,
            "model": scene.model_path,
            "backbone": pipe.backbone,
            "config": runtime_config,
            "total_planned_stages": num_stages,
            "gpu": {
                "device_index": gpu_index,
                "name": gpu_properties.name,
                "total_memory_bytes": gpu_properties.total_memory,
                "cuda_version": torch.version.cuda,
            },
        }
        runtime_report = runtime_profiler.write_json(
            os.path.join(scene.model_path, "runtime_breakdown.json"),
            runtime_metadata,
        )
        component_summary = ", ".join(
            "{}={:.3f}".format(name, values["total_s"])
            for name, values in runtime_report["components"].items()
        )
        one_off_summary = ", ".join(
            "{}={:.3f}".format(name, values["total_s"])
            for name, values in runtime_report["one_off_components"].items()
        )
        triggered = runtime_report["triggered_densification"]
        print(
            "Runtime breakdown (s): optimization={:.3f}, inclusive={:.3f}".format(
                runtime_report["optimization_time_s"],
                runtime_report["inclusive_time_s"],
            )
        )
        print("  optimization: " + component_summary)
        print(
            "  one-offs: {}; triggered topology actions={} ({:.3f}s)".format(
                one_off_summary,
                triggered["count"],
                triggered["triggered_total_s"],
            )
        )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def render_from_list(gs_list, pyr_func, viewpoint, render_adapter):
    merged = render_adapter(viewpoint, gs_list[-1], render_size=pyr_func.pyr_size_list[-1], pos_and_neg=False)["render"].unsqueeze(0)
    for i in range(len(gs_list) - 2, -1, -1):
        target_size = pyr_func.pyr_size_list[i][-2:]
        if merged.shape[-2:] != target_size:
            merged = pyr_func._pyr_up(merged, size=target_size)
        else:
            merged = merged
        
        if gs_list[i] is not None:
            render_level = render_adapter(viewpoint, gs_list[i], render_size=pyr_func.pyr_size_list[i], pos_and_neg=True)["render"].unsqueeze(0)
            merged = merged + render_level
        else:
            merged = merged
    return merged.squeeze(0)

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, lr_xyz, elapsed, testing_iterations, scene : Scene, gs_list, pyr_func, render_adapter):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('GS_count', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('lr_xyz', lr_xyz, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(render_from_list(gs_list, pyr_func, viewpoint, render_adapter), 0.0, 1.0)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                    # lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                # lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    # tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000 * (i + 1) for i in range(30)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--runtime_breakdown", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args), 
        op.extract(args), 
        pp.extract(args), 
        args.test_iterations, 
        args.save_iterations, 
        args.checkpoint_iterations, 
        args.start_checkpoint, 
        args.debug_from, 
        args.websockets,
        args.runtime_breakdown
    )

    # All done
    print("\nTraining complete.")
