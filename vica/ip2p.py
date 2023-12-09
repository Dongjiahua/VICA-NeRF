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

"""InstructPix2Pix module"""

# Modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py

import sys
from dataclasses import dataclass
from typing import Union

import torch
from rich.console import Console
from torch import nn
from torchtyping import TensorType
import matplotlib.pyplot as plt
CONSOLE = Console(width=120)

try:
    from diffusers import (
        DDIMScheduler,
        StableDiffusionInstructPix2PixPipeline,
    )
    from transformers import logging

except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)

logging.set_verbosity_error()
IMG_DIM = 512
CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
SD_SOURCE = "runwayml/stable-diffusion-v1-5"
CLIP_SOURCE = "openai/clip-vit-large-patch14"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor

class InstructPix2Pix(nn.Module):
    """InstructPix2Pix implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str], num_train_timesteps: int = 1000, ip2p_use_full_precision=False) -> None:
        super().__init__()

        self.device = device
        self.num_train_timesteps = num_train_timesteps
        self.ip2p_use_full_precision = ip2p_use_full_precision

        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(IP2P_SOURCE, torch_dtype=torch.float16, safety_checker=None)
        pipe.scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler")
        pipe.scheduler.set_timesteps(100)
        assert pipe is not None
        pipe = pipe.to(self.device)

        self.pipe = pipe

        # improve memory performance
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()

        self.scheduler = pipe.scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # type: ignore

        pipe.unet.eval()
        pipe.vae.eval()

        # use for improved quality at cost of higher memory
        if self.ip2p_use_full_precision:
            print("Using full precision")
            pipe.unet.float()
            pipe.vae.float()

        self.unet = pipe.unet
        self.auto_encoder = pipe.vae

        CONSOLE.print("InstructPix2Pix loaded!")
    
    @torch.no_grad()
    def mix(self,img1,img2,ratio):
        latents_1 = self.imgs_to_latent(img1)
        latents_2 = self.imgs_to_latent(img2)
        latents = latents_1 * ratio + latents_2 * (1 - ratio)
        final_image =  self.latents_to_img(latents)
        return final_image
    
    @torch.no_grad()
    def average_edit_inner(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],  
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 10,
        lower_bound: float = 0.2,
        upper_bound: float = 0.25,
        iters = 3,
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents_o = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        final_image = None
        final_latents = torch.zeros_like(latents_o)
        latents_all = torch.zeros((iters, *latents_o.shape), device=self.device)
        for i in range(iters):
            latents_all[i] = self.scheduler.add_noise(latents_o, torch.randn_like(latents_o), self.scheduler.timesteps[0])  # type: ignore
        for i, t in enumerate(self.scheduler.timesteps):
            noise = 0
            for it in range(iters):
                with torch.no_grad():
                    latents = latents_all[it]
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                    # print(latent_model_input.dtype)
                    # print(t.dtype)
                    # print(text_embeddings.dtype)
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )     
                noise+=noise_pred
            noise /= iters
            for it in range(iters):
                latents_all[it] = self.scheduler.step(noise, t, latents_all[it]).prev_sample
            
        final_latents = latents_all[0]
        
            # decode latents to get edited image
        #     with torch.no_grad():
        #         decoded_img = self.latents_to_img(latents)
        #         if final_image is None:
        #             final_image = torch.zeros_like(decoded_img)
        #         final_image+=decoded_img
        # final_image/=iters
        # final_latents/=iters
        if final_image is None:
            final_image =  self.latents_to_img(final_latents)
        

        return final_image
    @torch.no_grad()
    def double_average_edit(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],  
        image_cond2: TensorType["BS", 3, "H", "W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 10,
        lower_bound: float = 0.2,
        upper_bound: float = 0.25,
        iters = 3,
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents_o = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)
            image_cond_latents2 = self.prepare_image_latents(image_cond2)

        # add noise
        final_image = None
        final_latents = torch.zeros_like(latents_o)
        for it in range(iters):
            latents = self.scheduler.add_noise(latents_o, torch.randn_like(latents_o), self.scheduler.timesteps[0])  # type: ignore
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                    # print(latent_model_input.dtype)
                    # print(t.dtype)
                    # print(text_embeddings.dtype)
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat([latent_model_input, image_cond_latents2], dim=1)
                    # print(latent_model_input.dtype)
                    # print(t.dtype)
                    # print(text_embeddings.dtype)
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform classifier-free guidance
                _, noise_pred_image2, noise_pred_uncond2 = noise_pred.chunk(3)
                if i>=5:
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale/2 * (noise_pred_image - noise_pred_uncond)
                        +  image_guidance_scale/2*(noise_pred_image2 - noise_pred_uncond2)
                    )
                else:
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )   

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            final_latents+=latents
        
            # decode latents to get edited image
        #     with torch.no_grad():
        #         decoded_img = self.latents_to_img(latents)
        #         if final_image is None:
        #             final_image = torch.zeros_like(decoded_img)
        #         final_image+=decoded_img
        # final_image/=iters
        final_latents/=iters
        if final_image is None:
            final_image =  self.latents_to_img(final_latents)
        

        return final_image
    @torch.no_grad()
    def average_edit(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],  
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 10,
        lower_bound: float = 0.2,
        upper_bound: float = 0.25,
        iters = 3,
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents_o = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        final_image = None
        final_latents = torch.zeros_like(latents_o)
        for it in range(iters):
            latents = self.scheduler.add_noise(latents_o, torch.randn_like(latents_o), self.scheduler.timesteps[0])  # type: ignore
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                    # print(latent_model_input.dtype)
                    # print(t.dtype)
                    # print(text_embeddings.dtype)
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            final_latents+=latents
        
            # decode latents to get edited image
        #     with torch.no_grad():
        #         decoded_img = self.latents_to_img(latents)
        #         if final_image is None:
        #             final_image = torch.zeros_like(decoded_img)
        #         final_image+=decoded_img
        # final_image/=iters
        final_latents/=iters
        if final_image is None:
            final_image =  self.latents_to_img(final_latents)
        

        return final_image
    
    @torch.no_grad()
    def edit_image(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],  
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.70,
        upper_bound: float = 0.98,
        mask=None,
        insert_step = 0
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)
            origin_latents = self.imgs_to_latent(image_cond)

        # add noise
        noise = torch.randn_like(latents)
        
        latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
        if mask!=None:
            # mask[...]=1
            base = 0
            mask = mask*(1-base)+base
        # plt.imshow(mask.squeeze().cpu().numpy(),cmap="gray")
        # plt.savefig(f"edited/temp.png")
        # plt.close()
        # mask = None
        # print(self.scheduler.timesteps)
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                if mask!=None and i==insert_step:
                # pred noise
                    # mask*=1
                    if i<len(self.scheduler.timesteps)-1:
                        prev_gt = self.scheduler.add_noise(origin_latents, torch.randn_like(latents), self.scheduler.timesteps[i]) 
                    else:
                        prev_gt = origin_latents
                    latents = latents*mask + prev_gt*(1-mask)
                    
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)
                # print(latent_model_input.dtype)
                # print(t.dtype)
                # print(text_embeddings.dtype)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample


        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return decoded_img
    
    def pred_image_dif(
        self,
        image1: TensorType["BS", 3, "H", "W"],
        image2: TensorType["BS", 3, "H", "W"]

    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """



        with torch.no_grad():
            # prepare image and image_cond latents
            latents1 = self.imgs_to_latent(image1)
            latents2 = self.imgs_to_latent(image2)

        noise = latents1-latents2
            # break

        # decode latents to get edited image


        return noise
    
    def pred_image_noise(
        self,
        text_embeddings: TensorType["N", "max_length", "embed_dim"],
        image: TensorType["BS", 3, "H", "W"],
        image_cond: TensorType["BS", 3, "H", "W"],  
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        lower_bound: float = 0.02,
        upper_bound: float = 0.02
    ) -> torch.Tensor:
        """Edit an image for Instruct-NeRF2NeRF using InstructPix2Pix
        Args:
            text_embeddings: Text embeddings
            image: rendered image to edit
            image_cond: corresponding training image to condition on
            guidance_scale: text-guidance scale
            image_guidance_scale: image-guidance scale
            diffusion_steps: number of diffusion steps
            lower_bound: lower bound for diffusion timesteps to use for image editing
            upper_bound: upper bound for diffusion timesteps to use for image editing
        Returns:
            edited image
        """

        min_step = int(self.num_train_timesteps * lower_bound)
        max_step = int(self.num_train_timesteps * upper_bound)

        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        
        self.scheduler.config.num_train_timesteps = T.item()
        self.scheduler.set_timesteps(diffusion_steps)

        with torch.no_grad():
            # prepare image and image_cond latents
            latents = self.imgs_to_latent(image)
            image_cond_latents = self.prepare_image_latents(image_cond)

        # add noise
        noise = torch.randn_like(latents)
        # latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])  # type: ignore
        # print(self.scheduler.timesteps)
        old_latent = latents.clone()
        # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
        for i, t in enumerate(self.scheduler.timesteps):
            t=0
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents] * 3)
                latent_model_input = torch.cat([latent_model_input, image_cond_latents], dim=1)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform classifier-free guidance
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + guidance_scale * (noise_pred_text - noise_pred_image)
                + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            
            # get previous sample, continue loop
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            break
        noise = latents-old_latent
            # break

        # decode latents to get edited image
        with torch.no_grad():
            decoded_img = self.latents_to_img(latents)

        return noise_pred,decoded_img

    def latents_to_img(self, latents: TensorType["BS", 4, "H", "W"]) -> TensorType["BS", 3, "H", "W"]:
        """Convert latents to images
        Args:
            latents: Latents to convert
        Returns:
            Images
        """

        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.auto_encoder.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def imgs_to_latent(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert images to latents
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        posterior = self.auto_encoder.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: TensorType["BS", 3, "H", "W"]) -> TensorType["BS", 4, "H", "W"]:
        """Convert conditioning image to latents used for classifier-free guidance
        Args:
            imgs: Images to convert
        Returns:
            Latents
        """
        imgs = 2 * imgs - 1

        image_latents = self.auto_encoder.encode(imgs).latent_dist.mode()

        uncond_image_latents = torch.zeros_like(image_latents)
        image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
