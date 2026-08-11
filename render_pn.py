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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_fastgs
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import GaussianModel
from laplacian.laplacian_torch import LaplacePyramid
import torchvision.utils as vutils
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
import time


def getSubFolder(model_path, num_levels):
    return [os.path.join(model_path, f"L{i}") for i in range(num_levels)]


def load_gaussians(model_para, pipe, load_iteration):
    subfolders = getSubFolder(model_para.model_path, pipe.lap_n_levels)
    gaussians_list = []
    for i_s, subfolder in enumerate(subfolders):
        if load_iteration < 0:
            find_iterations = []
            point_cloud_dir = os.path.join(subfolder, "point_cloud")
            for name in os.listdir(point_cloud_dir):
                if name.startswith("iteration_"):
                    try:
                        it = int(name.split("_")[1])
                        find_iterations.append(it)
                    except:
                        pass
            if len(find_iterations) <= 0:
                raise Exception(f"[WARN] No iteration_* folders found in {point_cloud_dir}")
            use_iteration = max(find_iterations)
        else:
            use_iteration = load_iteration
        if i_s < len(subfolders) - 1 and pipe.inherit_max_sh_degree >= 1:
            use_sh = pipe.inherit_max_sh_degree
        else:
            use_sh = model_para.sh_degree
        gm = GaussianModel(use_sh)
        # gm.load_ply(os.path.join(subfolder, "point_cloud", "iteration_" + str(use_iteration), "point_cloud.ply"), model_para.train_test_exp)
        # train_test_exp means exposure mapping is used. No exposure in FastGS.
        gm.load_ply(os.path.join(subfolder, "point_cloud", "iteration_" + str(use_iteration), "point_cloud.ply"))
        gaussians_list.append(gm)
    return gaussians_list


def render_set(model_path, name, iteration, views, gaussians_list, pipeline, opt, background, train_test_exp, separate_sh, save_tensor=False, save_depth=False, save_inverse_gamma=False):
    if pipeline.backbone == "fastgs":
        from gaussian_renderer import render_fastgs as render_fn
    elif pipeline.backbone == "taming":
        from gaussian_renderer_dash import render as render_fn
    else:
        raise ValueError(f"Unknown backbone: {pipeline.backbone}")
    
    subfolders = getSubFolder(model_path, pipeline.lap_n_levels)
    output_path_list = []
    output_type = ["renders", "gt"]
    if save_tensor:
        output_type += ["renders_pt", "gt_pt"]
    if save_inverse_gamma:
        output_type += ["renders_inv_gamma", "gt_inv_gamma"]
    if save_depth:
        output_type += ["depth"]
    for folder in subfolders:
        output_path_dict = {}
        for out_t in output_type:
            a_path = os.path.join(folder, name, "ours_{}".format(iteration), out_t)
            makedirs(a_path, exist_ok=True)
            output_path_dict[out_t] = a_path
        output_path_list.append(output_path_dict)
    rec_path_dict = {}
    for out_t in ["renders", "gt"]:
        a_path = os.path.join(os.path.join(model_path, "Reconstructed"), name, "ours_{}".format(iteration), out_t)
        makedirs(a_path, exist_ok=True)
        rec_path_dict[out_t] = a_path

    pyramid_func = LaplacePyramid(levels=pipeline.lap_n_levels, device="cuda", decom_func_name=pipeline.decomposition_func, blur_up=pipeline.blur_up)

    total_time = 0.0

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt_full = view.original_image[0:3, :, :]

        gt_gaussian_pyr = pyramid_func.build_pyr_gaussian(gt_full.to("cuda").unsqueeze(0))

        for i_stage in range(len(gaussians_list)):
            gaussians = gaussians_list[-i_stage-1]

            gt_gau = gt_gaussian_pyr[-i_stage-1].squeeze(0)

            start_time = time.time()
            if i_stage == 0:
                # render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, render_size=gt_gau.shape[-2:], 
                #                     pos_and_neg=False)
                # render_pkg = render_fastgs(view, gaussians, pipeline, background, opt.mult, render_size=gt_gau.shape[-2:], pos_and_neg=False)
                if pipeline.backbone == "fastgs":
                    render_pkg = render_fn(view, gaussians, pipeline, background, opt.mult, render_size=gt_gau.shape[-2:], pos_and_neg=False)
                elif pipeline.backbone == "taming":
                    render_pkg = render_fn(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_gau.shape[-2:], pos_and_neg=False)
                else:
                    raise ValueError(f"Unknown backbone: {pipeline.backbone}")
                rendering = render_pkg["render"]
                pred_accumulated = rendering
            else:
                # render_pkg = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, render_size=gt_gau.shape[-2:], 
                #                     pos_and_neg=True)
                # render_pkg = render_fastgs(view, gaussians, pipeline, background, opt.mult, render_size=gt_gau.shape[-2:], pos_and_neg=True)
                if pipeline.backbone == "fastgs":
                    render_pkg = render_fn(view, gaussians, pipeline, background, opt.mult, render_size=gt_gau.shape[-2:], pos_and_neg=True)
                elif pipeline.backbone == "taming":
                    render_pkg = render_fn(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, render_size=gt_gau.shape[-2:], pos_and_neg=True)
                else:
                    raise ValueError(f"Unknown backbone: {pipeline.backbone}")
                rendering = render_pkg["render"]
                upsampled = pyramid_func.get_upsampled([rendering.unsqueeze(0), pred_accumulated.unsqueeze(0)])
                upsampled = upsampled.squeeze(0)   # upsampled is the upsampled result of pred_accumulated. 
                pred_accumulated = rendering + upsampled
                # rescale the rendering and gt from [-1, 1] to [0,1], and clamp
                rendering = (rendering + 1.0) / 2.0
            end_time = time.time()
            total_time += (end_time - start_time)

            # get a gt for visualization
            if i_stage == 0:
                gt = gt_gau
            else:
                gt = gt_gau - upsampled  # residual
                gt = (gt + 1.0) / 2.0

            output_path_dict = output_path_list[-i_stage-1]
            if save_depth:
                depth = render_pkg["depth"]
                torch.save(depth, os.path.join(output_path_dict["depth"], f"{view.uid}.pt"))

            if args.train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2:]
                gt = gt[..., gt.shape[-1] // 2:]

            if save_tensor:
                torch.save(rendering, os.path.join(output_path_dict["renders_pt"], f"{idx:05d}.pt"))
                torch.save(gt, os.path.join(output_path_dict["gt_pt"], f"{idx:05d}.pt"))
            
            if save_inverse_gamma:
                torchvision.utils.save_image(pyramid_func.invert_single_level_gamma(rendering, gamma=pipeline.lap_gamma), os.path.join(output_path_dict["renders_inverse_gamma"], '{0:05d}'.format(idx) + ".png"))
                torchvision.utils.save_image(pyramid_func.invert_single_level_gamma(gt, gamma=pipeline.lap_gamma), os.path.join(output_path_dict["gt_inv_gamma"], '{0:05d}'.format(idx) + ".png"))
            
            torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(output_path_dict["renders"], '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt.clamp(0.0,1.0), os.path.join(output_path_dict["gt"], '{0:05d}'.format(idx) + ".png"))
        
        # the final results will be pred_accumulated and gt_full.
        torchvision.utils.save_image(pred_accumulated.clamp(0.0,1.0), os.path.join(rec_path_dict["renders"], '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_full.clamp(0.0,1.0), os.path.join(rec_path_dict["gt"], '{0:05d}'.format(idx) + ".png"))
    
    num_frames = len(views)
    avg_time = total_time / num_frames if num_frames > 0 else 0
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")
    with open(os.path.join(model_path, "FPS_INFO"), "w+") as f:
        f.write(f"[{name}] Rendered {num_frames} frames in {total_time:.2f} seconds. Average FPS: {fps:.2f}")


def render_sets(model_para : ModelParams, iteration : int, pipeline : PipelineParams, opt: OptimizationParams, skip_train : bool, skip_test : bool, separate_sh: bool, save_tensor : bool, save_depth : bool, save_inverse_gamma : bool):
    with torch.no_grad():
        gaussians_list = load_gaussians(model_para, pipeline, iteration)
        scene = Scene(model_para, None, shuffle=False)

        bg_color = [1,1,1] if model_para.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(model_para.model_path, "train", iteration, scene.getTrainCameras(), gaussians_list, pipeline, opt, background, model_para.train_test_exp, separate_sh, 
                        save_tensor=save_tensor, save_depth=save_depth, save_inverse_gamma=save_inverse_gamma)

        if not skip_test:
             render_set(model_para.model_path, "test", iteration, scene.getTestCameras(), gaussians_list, pipeline, opt, background, model_para.train_test_exp, separate_sh, 
                        save_tensor=save_tensor, save_depth=save_depth, save_inverse_gamma=save_inverse_gamma)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--save_tensor", action="store_true")
    parser.add_argument("--save_depth", action="store_true")
    parser.add_argument("--save_inverse_gamma", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), op.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.save_tensor, args.save_depth, args.save_inverse_gamma)
