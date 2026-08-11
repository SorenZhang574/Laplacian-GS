import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from functools import partial


class LaplacePyramid:
    def __init__(self, levels=4, device='cpu', decom_func_name=None, blur_up=False):
        self.levels = levels
        self.device = device
        self.blur_up = blur_up

        self.kernel = self._create_gaussian_kernel().to(device)

        self.decom_func_name = decom_func_name
        self._dispatch = {
            "range01_gamma": (self.build_laplacian_pyramid_range01_gamma,
                              self.reconstruct_from_range01_gamma),
            "signed": (self.build_pyr_signed,
                    self.reconstruct_from_signed),
            "signed_no_down_up": (
                partial(self.build_pyr_signed, no_down_up=True),
                partial(self.reconstruct_from_signed, no_down_up=True),
            ),
        }
        if "signed" in decom_func_name:
            self.stages_ids = []
            for i in range(levels):
                if i < levels - 1:
                    self.stages_ids.extend([i, i])
                else:
                    self.stages_ids.append(i)
            self.num_levels = len(self.stages_ids)
        else:
            self.stages_ids = list(range(levels))
            self.num_levels = levels
        self.num_stages = levels

    def _create_gaussian_kernel(self):
        """Create a 5x5 Gaussian kernel (separable form) for differentiable convolution"""
        kernel_1d = torch.tensor([1., 4., 6., 4., 1.])
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d /= kernel_2d.sum()
        kernel_2d = kernel_2d[None, None, :, :]  # shape: (1, 1, 5, 5)
        return kernel_2d

    def _gaussian_blur(self, x):
        """Apply differentiable Gaussian blur using group convolution"""
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, 1, 5, 5)
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        x = F.conv2d(x, kernel, groups=C)
        return x

    def _pyr_down(self, x):
        """Downsample using stride-2 convolution (differentiable)"""
        B, C, H, W = x.shape
        kernel = self.kernel.expand(C, 1, 5, 5)
        x = F.pad(x, (2, 2, 2, 2), mode='reflect')
        return F.conv2d(x, kernel, stride=2, groups=C)

    def _pyr_up(self, x, size):
        """Upsample using bilinear interpolation followed by Gaussian blur (differentiable)"""
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        if self.blur_up:
            x = self._gaussian_blur(x)
        return x

    def build_gaussian_pyramid(self, image):
        """Build the Gaussian pyramid from an input image"""
        current = image
        pyramid = [current]
        for _ in range(self.levels-1):
            current = self._pyr_down(current)
            pyramid.append(current)
        return pyramid

    def build_laplacian_from_gaussian_pyramid(self, gaussian_pyramid):
        """Build Laplacian pyramid from given Gaussian pyramid"""
        laplacian_pyramid = []
        for i in range(self.levels - 1):
            current = gaussian_pyramid[i]
            next_level = gaussian_pyramid[i + 1]
            upsampled = self._pyr_up(next_level, size=current.shape[2:])
            lap = current - upsampled
            laplacian_pyramid.append(lap)
        laplacian_pyramid.append(gaussian_pyramid[-1])  # last residual
        return laplacian_pyramid
    
    def build_one_lap_layer_from_two_gauss(self, gaussian_pyramid):
        if gaussian_pyramid[0].shape[2:] != gaussian_pyramid[1].shape[2:]:
            upsampled = self._pyr_up(gaussian_pyramid[1], size=gaussian_pyramid[0].shape[2:])
        else:
            upsampled = gaussian_pyramid[1]
        lap = gaussian_pyramid[0] - upsampled
        return lap, upsampled
    
    def get_upsampled(self, gaussian_pyramid):
        if gaussian_pyramid[0].shape[2:] != gaussian_pyramid[1].shape[2:]:
            upsampled = self._pyr_up(gaussian_pyramid[1], size=gaussian_pyramid[0].shape[2:])
        else:
            upsampled = gaussian_pyramid[1]
        return upsampled

    def build_laplacian_pyramid(self, image):
        """Convenience method to build Laplacian pyramid directly from image"""
        g_pyr = self.build_gaussian_pyramid(image)
        l_pyr = self.build_laplacian_from_gaussian_pyramid(g_pyr)
        return l_pyr

    def reconstruct(self, lap_pyramid):
        """Reconstruct the original image from the Laplacian pyramid"""
        image = lap_pyramid[-1]
        for i in reversed(range(len(lap_pyramid) - 1)):
            up = self._pyr_up(image, size=lap_pyramid[i].shape[2:])
            image = up + lap_pyramid[i]
        return image
    
    def build_pyr(self, image, **kwargs):
        build_func, _ = self._dispatch[self.decom_func_name]
        return build_func(image, **kwargs)

    def reconstruct_from_pyr(self, l_pyr, **kwargs):
        _, recon_func = self._dispatch[self.decom_func_name]
        return recon_func(l_pyr, **kwargs)
    
    def build_pyr_gaussian(self, image, **kwargs):
        g_pyr = self.build_gaussian_pyramid(image)

        if "no_down_up" in self.decom_func_name:
            init_size = g_pyr[0].shape[2:]  # (H, W)
            resized_g_pyr = []
            for i, x in enumerate(g_pyr):
                if i > 0:
                    x_resized = self._pyr_up(x, size=init_size)
                else:
                    x_resized = x
                resized_g_pyr.append(x_resized)
            return resized_g_pyr
        else:
            return g_pyr
    
    def build_laplacian_pyramid_range01_gamma(self, image, gamma=2.2, eps=1e-8):
        l_pyr = self.build_laplacian_pyramid(image)
        ginv = 1.0 / gamma
        # the last is already [0,1]
        # process all levels except the last
        for i, res in enumerate(l_pyr[:-1]):
            # split positive and negative parts
            pos = torch.relu(res)
            neg = torch.relu(-res)
            # apply gamma transform separately
            pos_gamma = torch.pow(pos + eps, ginv)
            neg_gamma = -torch.pow(neg + eps, ginv)
            # combine back
            res_gamma = pos_gamma + neg_gamma
            # map from [-1,1] → [0,1]
            res_mapped = (res_gamma + 1.0) * 0.5
            # overwrite level
            l_pyr[i] = res_mapped
        return l_pyr
    
    def reconstruct_from_range01_gamma(self, l_pyr, gamma=2.2, eps=1e-8):
        """
        Invert the per-level transform done in build_laplacian_pyramid_range01_gamma:
        - For all levels except the last: [0,1] -> [-1,1], then inverse gamma (sign-aware)
        - Keep the last level unchanged ([0,1] already)
        - Finally call self.reconstruct(new_pyr)
        """
        inv_residuals = []
        for res_mapped in l_pyr[:-1]:
            # [0,1] -> [-1,1]
            res_gamma = res_mapped * 2.0 - 1.0
            # Split sign in gamma space
            pos_g = torch.relu(res_gamma)      # >= 0
            neg_g = torch.relu(-res_gamma)     # magnitude of negative part
            # Inverse gamma on magnitudes (matching forward: (x + eps)^(1/gamma))
            pos = torch.pow(pos_g, gamma) - eps
            neg = torch.pow(neg_g, gamma) - eps
            # Safety clamp (avoid tiny negatives due to fp errors)
            pos = torch.clamp(pos, min=0.0)
            neg = torch.clamp(neg, min=0.0)
            # Restore sign
            inv_residuals.append(pos - neg)

        # Assemble a new pyramid with inverted residuals and the original last/base level
        new_pyr = inv_residuals + [l_pyr[-1]]

        # Reconstruct using your existing function
        img = self.reconstruct(new_pyr)
        return img
    
    def invert_single_level_gamma(self, res_mapped, gamma=2.2, eps=1e-8):
        """
        Invert gamma transform for one Laplacian level.
        Input:  res_mapped in [0,1] (gamma-transformed residual)
        Output: res_inverted in [0,1] (linear residual, no gamma)
        """
        # [0,1] -> [-1,1]
        res_gamma = res_mapped * 2.0 - 1.0

        # split sign
        pos_g = torch.relu(res_gamma)
        neg_g = torch.relu(-res_gamma)

        # inverse gamma
        pos = torch.pow(pos_g, gamma) - eps
        neg = torch.pow(neg_g, gamma) - eps
        pos = torch.clamp(pos, min=0.0)
        neg = torch.clamp(neg, min=0.0)

        res = pos - neg  # back to [-1,1]

        # map back to [0,1]
        res_out = (res + 1.0) * 0.5
        return res_out
    
    def build_pyr_signed(self, image, gamma: float = 2.2, eps: float = 1e-8, no_down_up: bool = False):
        """
        Output: new_pyr (list)
        Order: [L0_pos, L0_neg, L1_pos, L1_neg, ..., base]
        Each tensor is in [0,1].
        """
        if no_down_up:
            pyr = self.build_laplacian_pyramid_no_down_up(image)  # [r0, r1, ..., base]
        else:
            pyr = self.build_laplacian_pyramid(image)  # [r0, r1, ..., base]
        new_pyr = []

        for res in pyr[:-1]:
            pos = torch.relu(res)
            neg = torch.relu(-res)

            # Apply gamma compression
            if gamma == 1.0:
                pos_m, neg_m = pos, neg
            else:
                ginv = 1.0 / gamma
                pos_m = torch.pow(pos + eps, ginv)
                neg_m = torch.pow(neg + eps, ginv)

            # new_pyr.append(torch.clamp(pos_m, 0.0, 1.0))
            # new_pyr.append(torch.clamp(neg_m, 0.0, 1.0))
            new_pyr.append(pos_m)
            new_pyr.append(neg_m)

        # The last base layer (low frequency / Gaussian pyramid base)
        base = pyr[-1]
        new_pyr.append(base)
        return new_pyr

    def reconstruct_from_signed(self, new_pyr, gamma: float = 2.2, eps: float = 1e-8, no_down_up: bool = False):
        """
        Input: new_pyr list, ordered as [L0_pos, L0_neg, L1_pos, L1_neg, ..., base]
        Output: reconstructed image
        """
        assert len(new_pyr) >= 1 and (len(new_pyr) - 1) % 2 == 0, "new_pyr should be [pos, neg, ..., base]"
        inv_residuals = []

        for i in range(0, len(new_pyr) - 1, 2):
            # pos_m = torch.clamp(new_pyr[i],   0.0, 1.0)
            # neg_m = torch.clamp(new_pyr[i+1], 0.0, 1.0)
            pos_m = new_pyr[i]
            neg_m = new_pyr[i+1]

            if gamma == 1.0:
                pos, neg = pos_m, neg_m
            else:
                pos = torch.pow(pos_m, gamma) - eps
                neg = torch.pow(neg_m, gamma) - eps
                # pos = torch.clamp(pos, min=0.0)
                # neg = torch.clamp(neg, min=0.0)

            res = pos - neg
            inv_residuals.append(res)

        # base = torch.clamp(new_pyr[-1], 0.0, 1.0)
        base = new_pyr[-1]
        if no_down_up:
            img = self.reconstruct_no_down_up(inv_residuals + [base])
        else:
            img = self.reconstruct(inv_residuals + [base])
        # return torch.clamp(img, 0.0, 1.0)
        return img
    
    """"Begin: build and reconstruct without downsampling / upsampling"""

    def build_laplacian_pyramid_no_down_up(self, image):
        """Convenience method to build Laplacian pyramid directly from image"""
        g_pyr = self.build_gaussian_pyramid(image)

        init_size = g_pyr[0].shape[2:]  # (H, W)
        resized_g_pyr = []
        for i, x in enumerate(g_pyr):
            if i > 0:
                x_resized = self._pyr_up(x, size=init_size)
            else:
                x_resized = x
            resized_g_pyr.append(x_resized)

        laplacian_pyramid = []
        for i in range(self.levels - 1):
            current = resized_g_pyr[i]
            next_level = resized_g_pyr[i + 1]
            lap = current - next_level
            laplacian_pyramid.append(lap)
        laplacian_pyramid.append(resized_g_pyr[-1])  # last residual
        return laplacian_pyramid
    
    def reconstruct_no_down_up(self, lap_pyramid):
        """Reconstruct the original image from the Laplacian pyramid"""
        image = lap_pyramid[-1]
        for i in reversed(range(len(lap_pyramid) - 1)):
            image = image + lap_pyramid[i]
        return image

    """"End: build and reconstruct without downsampling / upsampling"""


def save_pyramid_images(g_pyr, l_pyr, save_folder="laplacian_torch_outputs"):
    os.makedirs(save_folder, exist_ok=True)

    # Save Gaussian pyramid
    for i, g in enumerate(g_pyr):
        g_cpu = g.detach().cpu().squeeze(0)  # (C,H,W)
        g_np = g_cpu.permute(1, 2, 0).numpy()  # (H,W,C)
        g_np = np.clip(g_np * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{save_folder}/g{i}.png", cv2.cvtColor(g_np, cv2.COLOR_RGB2BGR))

    # Save Laplacian pyramid (shift to visualizable range)
    for i, lap in enumerate(l_pyr[:-1]):  # skip the last residual image
        lap_cpu = lap.detach().cpu().squeeze(0)
        lap_np = lap_cpu.permute(1, 2, 0).numpy()
        lap_np = (lap_np + 1.0) / 2.0 * 255.0  # Shift range to [0,255]
        lap_np = np.clip(lap_np, 0, 255).astype(np.uint8)
        cv2.imwrite(f"{save_folder}/l{i}.png", cv2.cvtColor(lap_np, cv2.COLOR_RGB2BGR))


# Example usage (main function)
if __name__ == "__main__":
    # Load an example image and convert to PyTorch tensor
    img = Image.open("_DSC8679.JPG").convert("RGB")
    img_tensor = TF.to_tensor(img).unsqueeze(0).to("cuda")  # shape: (1, 3, H, W)

    # img_tensor.requires_grad_(True)
    
    # Create the Laplacian pyramid builder
    pyramid_func = LaplacePyramid(levels=4, device="cuda")

    # Build the Laplacian pyramid
    # l_pyr = pyramid.build_laplacian_pyramid(img_tensor)
    g_pyr = pyramid_func.build_gaussian_pyramid(img_tensor)
    l_pyr = pyramid_func.build_laplacian_from_gaussian_pyramid(g_pyr)

    save_pyramid_images(g_pyr, l_pyr, save_folder="laplacian_torch_outputs")

    # Reconstruct the image from the pyramid
    reconstructed = pyramid_func.reconstruct(l_pyr)

    # Compute reconstruction loss (L1 loss)
    loss = F.l1_loss(reconstructed, img_tensor)
    print("L1 reconstruction loss:", loss.item())

    # Backpropagate if used in training
    # loss.backward()

    pass
