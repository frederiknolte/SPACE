import imageio
import numpy as np
import torch
from utils import spatial_transform
from .utils import bbox_in_one
from attrdict import AttrDict
from torchvision.utils import make_grid
from torch.utils.data import Subset, DataLoader
import wandb


class SpaceVis:
    def __init__(self):
        pass
    
    
    @torch.no_grad()
    def train_vis(self, log, global_step, mode, num_batch=10):
        """
        """
        B = num_batch
        
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
                if isinstance(log[key], torch.Tensor) and log[key].ndim > 0:
                    log[key] = log[key][:num_batch]
        log = AttrDict(log)
        
        # (B, 3, H, W)
        fg_box = bbox_in_one(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon, fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.view(B*N, 3, H, W)
        
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        
        wandb.log({f'{mode}/#0-separations': wandb.Image(grid_image)}, commit=False)

        grid_image = make_grid(log.imgs, 5, normalize=False, pad_value=1)
        wandb.log({f'{mode}/1-image': wandb.Image(grid_image)}, commit=False)

        grid_image = make_grid(log.y, 5, normalize=False, pad_value=1)
        wandb.log({f'{mode}/2-reconstruction_overall': wandb.Image(grid_image)}, commit=False)

        grid_image = make_grid(log.bg, 5, normalize=False, pad_value=1)
        wandb.log({f'{mode}/3-background': wandb.Image(grid_image)}, commit=False)

        mse = (log.y - log.imgs) ** 2
        mse = mse.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        log_like, kl_z_what, kl_z_where, kl_z_pres, kl_z_depth, kl_bg = (
            log['log_like'].mean(), log['kl_z_what'].mean(), log['kl_z_where'].mean(),
            log['kl_z_pres'].mean(), log['kl_z_depth'].mean(), log['kl_bg'].mean()
        )
        loss_boundary = log.boundary_loss.mean()
        loss = log.loss.mean()
        
        count = log.z_pres.flatten(start_dim=1).sum(dim=1).mean(dim=0)
        wandb.log({f'{mode}/mse': mse.item(),
                   f'{mode}/loss': loss,
                   f'{mode}/count': count,
                   f'{mode}/log_like': log_like.item(),
                   f'{mode}/loss_boundary': loss_boundary.item(),
                   f'{mode}/What_KL': kl_z_what.item(),
                   f'{mode}/Where_KL': kl_z_where.item(),
                   f'{mode}/Pres_KL': kl_z_pres.item(),
                   f'{mode}/Depth_KL': kl_z_depth.item(),
                   f'{mode}/Bg_KL': kl_bg.item(),
                   'global_step': global_step}, commit=True)

    @torch.no_grad()
    def show_vis(self, model, dataset, indices, path, device):
        dataset = Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=len(indices), shuffle=False)
        data = next(iter(dataloader))
        data = data.to(device)
        loss, log = model(data, 100000000)
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                log[key] = value.detach().cpu()
        log = AttrDict(log)
        # (B, 3, H, W)
        fg_box = bbox_in_one(
            log.fg, log.z_pres, log.z_scale, log.z_shift
        )
        # (B, 1, 3, H, W)
        imgs = log.imgs[:, None]
        fg = log.fg[:, None]
        recon = log.y[:, None]
        fg_box = fg_box[:, None]
        bg = log.bg[:, None]
        # (B, K, 3, H, W)
        comps = log.comps
        # (B, K, 3, H, W)
        masks = log.masks.expand_as(comps)
        masked_comps = comps * masks
        alpha_map = log.alpha_map[:, None].expand_as(imgs)
        grid = torch.cat([imgs, recon,  fg, fg_box, bg, masked_comps, masks, comps, alpha_map], dim=1)
        nrow = grid.size(1)
        B, N, _, H, W = grid.size()
        grid = grid.view(B*N, 3, H, W)
        
        # (3, H, W)
        grid_image = make_grid(grid, nrow, normalize=False, pad_value=1)
        
        # (H, W, 3)
        image = torch.clamp(grid_image, 0.0, 1.0)
        image = image.permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        imageio.imwrite(path, image)
