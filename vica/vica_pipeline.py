# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix Pipeline and trainer"""
import cv2
import os
from nerfstudio.cameras import camera_utils
from PIL import Image
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from vica.data.vica_datamanager import (
    VICADataManagerConfig,
)
from vica.vica_utils import *
import open3d as o3d
from vica.ip2p import InstructPix2Pix
from tqdm import tqdm
import matplotlib.pyplot as plt

@dataclass
class VICAPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VICAPipeline)
    """target class to instantiate"""
    datamanager: VICADataManagerConfig = VICADataManagerConfig()
    """specifies the datamanager config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 7.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    diffusion_steps: int = 5
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.5
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = False
    """Whether to use full precision for InstructPix2Pix"""
    first_run: bool = True
    control: bool = False
    warm_up_iters: int = -1
    post_refine: bool = False
    vis_name: str = "default"
    seed: int = 0

class VICAPipeline(VanillaPipeline):
    """VICA pipeline"""

    config: VICAPipelineConfig

    def __init__(
        self,
        config: VICAPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)
        # select device for InstructPix2Pix
        self.ip2p_device = (
            torch.device(device)
            if self.config.ip2p_device is None
            else torch.device(self.config.ip2p_device)
        )

        self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        # load base text embedding using classifier free guidance
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )
        self.auto_thres = 0.5
        self.edit_thres = 0.9
        self.seg = False
        if self.config.warm_up_iters==-1 and test_mode != "inference":  # when using rendering script
            self.warm_up_iters = int(self.datamanager.image_batch['image'].shape[0]**(1/2))
        else:
            self.warm_up_iters = self.config.warm_up_iters


        # seed everything
        torch.manual_seed(self.config.seed)
        
    def seg_update(self, image1, image2, idx, p=""):
        if not self.seg:
            return image2.squeeze()
        if p!="blending":
            return image2.squeeze()
        else:
            image1 = self.old_img[idx].cuda()
        if image1.shape[0]==3:
            image1 = image1.unsqueeze(0)
        if image2.shape[0]==3:
            image2 = image2.unsqueeze(0)

        if (image2.size() != image1.size()):
            image2 = torch.nn.functional.interpolate(image2, size=image1.size()[2:], mode='bilinear')
        seg_mask = self.seg_mask[idx].cuda()
        if len(image1.shape)==3:
            seg_mask = seg_mask.unsqueeze(-1)
        final_image = image1*(1-seg_mask) + image2*seg_mask
        return final_image.squeeze()   

    def get_reprojection_error(self, forward_grid, backward_grid, curr_grid, thres=1):
        forward_grid = forward_grid.squeeze() # H x W x 2
        backward_grid = backward_grid.squeeze()
        H,W = forward_grid.shape[:2]
        # print(forward_grid)
        forward_grid[...,0] = forward_grid[...,0]/(W)*2-1
        forward_grid[...,1] = forward_grid[...,1]/(H)*2-1
        W_bound = 1-1/W
        H_bound = 1-1/H
        forward_mask = (forward_grid[...,0] > -W_bound) & (forward_grid[...,0] < W_bound) & (forward_grid[...,1] > -H_bound) & (forward_grid[...,1] < H_bound)
        
        backward_grid[...,0] = backward_grid[...,0]/(W)*2-1
        backward_grid[...,1] = backward_grid[...,1]/(H)*2-1
        
        backward_mask = (backward_grid[...,0] > -W_bound) & (backward_grid[...,0] < W_bound) & (backward_grid[...,1] > -H_bound) & (backward_grid[...,1] < H_bound)
        # print(backward_mask.shape)
        backward_grid[~backward_mask,:]= 10 
        
        curr_grid[...,0] = curr_grid[...,0]/(W)*2-1
        curr_grid[...,1] = curr_grid[...,1]/(H)*2-1
        
        re_proj_grid = F.grid_sample(backward_grid[None,...].permute(0,3,1,2),forward_grid[None,...],mode="nearest",padding_mode="zeros").squeeze().permute(1,2,0)
        
        error = torch.norm(re_proj_grid - curr_grid.cuda(), dim=-1)

        valid_mask = (error < thres)&forward_mask

        return valid_mask
        
    @torch.no_grad()
    def cal_warpped_img(self, ref_spot, cur_spot, ref_image, thres=0.01, mode="bilinear"):
        cur_mask = self.datamanager.mask[cur_spot].to(self.device)
        current_index = self.datamanager.original_image_batch["image_idx"][cur_spot]
        ref_index = self.datamanager.original_image_batch["image_idx"][ref_spot]

        cur_img = self.datamanager.image_batch["image"][cur_spot].to(self.device)
        cur_camera_transforms = self.model.camera_optimizer(current_index.unsqueeze(dim=0))
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=cur_camera_transforms)
        curr_coords = current_camera.get_image_coords(index=0).squeeze()[:,:,[1,0]]
        
        ref_camera_transforms = self.model.camera_optimizer(ref_index.unsqueeze(dim=0))
        ref_camera = self.datamanager.train_dataparser_outputs.cameras[ref_index].to(self.device)
        ref_ray_bundle = ref_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=ref_camera_transforms)
        
        point = current_ray_bundle.origins + current_ray_bundle.directions * self.depth[cur_spot][:,:,None,None].to(self.device)
        new_direction = point - ref_ray_bundle.origins
        
        coords_cur_ref = generate_coords_from_directions(ref_camera, torch.tensor(list(range(1))).unsqueeze(-1), new_direction,camera_opt_to_camera=ref_camera_transforms)

        point = ref_ray_bundle.origins + ref_ray_bundle.directions * self.depth[ref_spot][:,:,None,None].to(self.device)
        new_direction = point - current_ray_bundle.origins
        
        coords_ref_cur = generate_coords_from_directions(current_camera, torch.tensor(list(range(1))).unsqueeze(-1), new_direction,camera_opt_to_camera=cur_camera_transforms)
        
        valid_mask = self.get_reprojection_error(coords_cur_ref, coords_ref_cur, curr_coords, thres=thres)
        
        new_img = self.warp_image(ref_image, coords_cur_ref, valid_mask, mode = mode)

        mask = valid_mask*(1-cur_mask.squeeze()) #*rgb_mask
        blended_img = new_img*mask[...,None] + cur_img*(1-mask)[...,None]
        mask = (valid_mask+(cur_mask.squeeze())).clamp(0,1)
        
        return blended_img, mask

        
    def warp_image(self, image, src_grid, mask, mode="bilinear"):
        image = image.permute(2,0,1).unsqueeze(0)  # 1, 3 ,H, W
        
        H, W = image.shape[2:]
        src_grid = src_grid.squeeze().unsqueeze(0) # 1, H,W,2
        src_grid[:, 0] = src_grid[:, 0] / ((W) / 2) - 1 # scale to -1~1
        src_grid[:, 1] = src_grid[:, 1] / ((H) / 2) - 1  # scale to -1~1
        src_grid = src_grid.float() # 1, N, 1,2
        wraped_rgb = F.grid_sample(image,src_grid, mode=mode, padding_mode='zeros',align_corners=False).squeeze().permute(1,2,0) #N,3
        new_image = torch.zeros(H,W,3,device=image.device)

        new_image[mask[...,None].expand(*mask.shape,3)] = wraped_rgb[mask[...,None].expand(*mask.shape,3)]
        return new_image

    def get_eval_render_image(self,current_spot):
        current_index = self.datamanager.eval_image_batch["image_idx"][current_spot]
        current_camera = self.datamanager.eval_dataset._dataparser_outputs.cameras[current_index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1))

        camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        
        depth_image = camera_outputs["depth"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        
        return depth_image, rendered_image
    
    def get_render_image(self,current_spot):
        current_index = self.datamanager.original_image_batch["image_idx"][current_spot]

        camera_transforms = self.model.camera_optimizer(current_index.unsqueeze(dim=0))
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

        camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
       
        depth_image = camera_outputs["depth"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        
        
        return depth_image, rendered_image

    def get_inpaint_image(self,current_spot, image=None, diffusion_steps=None, lower_bound=None, upper_bound=None):
        original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device).unsqueeze(dim=0).permute(0, 3, 1, 2)
        if image is None:
            rendered_image = self.get_render_image(current_spot)[1]

        else:
            rendered_image = image.unsqueeze(dim=0).permute(0, 3, 1, 2)
        ratio=0
        lower_bound=0.5
        upper_bound=0.6
        # for i in range(itt):
        edited_image = self.ip2p.average_edit(
                                self.text_embedding,
                                rendered_image.to(self.device),
                                original_image.to(self.device),
                                guidance_scale=7.5,
                                image_guidance_scale=1.5,
                                diffusion_steps=3,
                                lower_bound=lower_bound,
                                upper_bound=upper_bound,
                                iters = 5
                            ).detach()
        edited_image = self.seg_update(original_image,edited_image,current_spot).unsqueeze(0)
        if image is not None:
            edited_image = self.ip2p.average_edit(
                                    self.text_embedding,
                                    edited_image.to(self.device),
                                    rendered_image.to(self.device),
                                    guidance_scale=7.5,
                                    image_guidance_scale=1.5,
                                    diffusion_steps=3,
                                    lower_bound=lower_bound,
                                    upper_bound=upper_bound,
                                    iters = 5
                                ).detach()
        edited_image = self.seg_update(original_image,edited_image,current_spot, p = "blending").unsqueeze(0)
        if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')
        rendered_image = edited_image*(1-ratio)+rendered_image.to(self.device)*(ratio)
        edited_image = rendered_image

        return edited_image.permute(0, 2, 3, 1).squeeze(0)
    
    def get_edit_image(self,current_spot,render=False,image=None, mask_edit=False,diffusion_steps=10, lower_bound=0.02, upper_bound=0.98, use_image=False, old=False):

        # get original image from dataset
        original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)

        # generate current index in datamanger
        current_index = self.datamanager.original_image_batch["image_idx"][current_spot]

        camera_transforms = self.model.camera_optimizer(current_index.unsqueeze(dim=0))
        current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)


        original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
        camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)

        if use_image:
            image = self.datamanager.image_batch["image"][current_spot].to(self.device)
        if image is not None:
            rendered_image=image.unsqueeze(dim=0).permute(0, 3, 1, 2).to(self.ip2p_device)
        
     
        original_image = self.old_img[current_spot].cuda() if old==True else original_image

        edited_image = self.ip2p.edit_image(
                    self.text_embedding.to(self.ip2p_device),
                    rendered_image.to(self.ip2p_device),
                    original_image.to(self.ip2p_device),
                    guidance_scale=self.config.guidance_scale,
                    image_guidance_scale=self.config.image_guidance_scale,
                    diffusion_steps=diffusion_steps,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    mask = None
                    
                )
        edited_image = self.seg_update(original_image,edited_image,current_spot).unsqueeze(0)

        if (edited_image.size() != rendered_image.size()):
            edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')

        return edited_image,rendered_image,current_index

    def get_key_frame(self):
        maxi = 0.3
        key_frames = self.chosed
        H,W = self.datamanager.mask.shape[1:3]
        noise_sum = self.datamanager.mask.sum(dim=1).sum(dim=1).squeeze()/(H*W)
        noise_k = noise_sum.clone()
        noise_k[noise_k<maxi]=2*maxi- noise_k[noise_k<maxi]
        # print(noise_sum.shape)
        noise_ord = torch.argsort(noise_k,descending=False).squeeze()
        if noise_sum.min()>self.edit_thres:
            return None,1
        for k in noise_ord:
            if k not in key_frames:
                return int(k),noise_sum.min()
        return None,1
        
        
    @torch.no_grad()
    def warm_up(self):
        rate = 0
        for _ in tqdm(range(self.warm_up_iters)):
            chosen_idx = np.random.randint(0,self.datamanager.mask.shape[0])
            edited_image,rendered_image, current_index = self.get_edit_image(chosen_idx,mask_edit=False,use_image=True,lower_bound=self.config.lower_bound,upper_bound=self.config.upper_bound,diffusion_steps=10)
            ref_image = edited_image.squeeze().permute(1,2,0).detach()
            
            for i in tqdm(range(0,self.datamanager.image_batch['image'].shape[0])): 
                image, mask = self.cal_warpped_img(chosen_idx,i,ref_image)
                image = image.detach().cpu() * (1-rate)+self.datamanager.image_batch["image"][i] * rate 

                self.datamanager.image_batch["image"][i] = self.seg_update(image.cuda(),image,i)
                

                
    def edit_one_frame(self,chosen_idx, step, masked=0):
        agree = "0"
        while agree!="1":
            vis_mask = (self.datamanager.mask[chosen_idx].cpu().numpy()*255).astype(np.uint8)
            edited_image,rendered_image, current_index = self.get_edit_image(chosen_idx,mask_edit=False,use_image=True,lower_bound=self.config.lower_bound,upper_bound=self.config.upper_bound,diffusion_steps=10)
            ref_image = edited_image.squeeze().permute(1,2,0).detach()
            if not self.config.control:
                break
            
            new_img, mask = self.cal_warpped_img(chosen_idx,chosen_idx,ref_image)
            
            if masked>self.auto_thres:
                agree = "1"
            else:
                # show the image and mask
                vis_image = (new_img.cpu().numpy()*255).astype(np.uint8)

                ax = plt.subplot(1, 2, 1)
                ax.imshow(vis_image)
                ax = plt.subplot(1, 2, 2)
                ax.imshow(vis_mask, cmap='gray')
                plt.show()
                agree = input(f"Disagree[0],  Agree[1],  Reset[2]: ").strip()
                if agree == "2":
                    self.datamanager.mask[:]=0
                    agree = "0"

        mask_percent = torch.zeros(self.datamanager.image_batch['image'].shape[0]).cuda()
        mask_percent[chosen_idx] = 1

        for i in tqdm(range(0,self.datamanager.image_batch['image'].shape[0])): 

            image, mask = self.cal_warpped_img(chosen_idx,i,ref_image)
            
            self.datamanager.image_batch["image"][i] = self.seg_update(self.datamanager.image_batch["image"][i].cuda(),image,i)
            mask_percent[i] = mask.sum()/mask.numel()
            self.datamanager.mask[i] = mask[:,:,None].detach()

        chosen_idx = torch.argsort(mask_percent,descending=True)[:10]

    
    def key_frame_editing(self, step):
        chosen_idx = int(np.random.choice(self.datamanager.image_batch['image'].shape[0],1))
        self.chosed=[]
        self.chosed.append(chosen_idx)
        masked = 0
        while chosen_idx is not None:
            self.edit_one_frame(chosen_idx,step,masked)
            chosen_idx,masked = self.get_key_frame()
            self.chosed.append(chosen_idx)
        self.new_images = []

        for k in tqdm(range(self.datamanager.image_batch['image'].shape[0])):
            image=self.get_inpaint_image(k,self.datamanager.image_batch["image"][k])
            
            self.datamanager.original_image_batch["image"][k]=self.datamanager.image_batch["image"][k]
            self.new_images.append(image.detach().cpu())
            self.datamanager.image_batch["image"][k] = self.old_img[k].cuda().squeeze().permute(1,2,0)
            self.datamanager.image_batch["image"][k] = image
            

   
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        ray_bundle, batch = self.datamanager.next_train(step)

        model_outputs = self.model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.first_run:   
            with torch.no_grad():
                self.config.first_run=False
                self.inverse_index = torch.zeros_like(self.datamanager.original_image_batch["image_idx"])    
                for i in range(self.datamanager.original_image_batch["image_idx"].shape[0]):
                    self.inverse_index[self.datamanager.original_image_batch["image_idx"][i]] = i  
                self.old_img = []
                self.depth = torch.zeros(self.datamanager.image_batch['image'].shape[0],  *self.datamanager.image_batch['image'].shape[1:3])
                self.train_indices_order = cycle(range(self.datamanager.original_image_batch["image_idx"].shape[0]))
                
                for i in tqdm(range(self.datamanager.image_batch['image'].shape[0])):
                    depth,render = self.get_render_image(i)
                    self.old_img.append(render.detach().cpu())
                    self.depth[i] = depth.detach().squeeze()
                self.inverse_index = self.inverse_index.cuda()
                self.warm_up()
                self.key_frame_editing(step)

 
        if self.config.post_refine and step==35000:
            for k in tqdm(range(self.datamanager.image_batch['image'].shape[0])):
                self.datamanager.image_batch["image"][k]=self.get_inpaint_image(k,None)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
