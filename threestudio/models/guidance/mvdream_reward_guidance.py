import sys
import random
import numpy as np
from typing import List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from mvdream.camera_utils import convert_opengl_to_blender, normalize_camera
from mvdream.model_zoo import build_model

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseModule
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.typing import *


@threestudio.register("multiview-reward-guidance")
class MultiviewDiffusionGuidance(BaseModule):
    @dataclass
    class Config(BaseModule.Config):
        model_name: str = "sd-v2.1-base-4view"  # check mvdream.model_zoo.PRETRAINED_MODELS
        ckpt_path: Optional[str] = None  # path to local checkpoint (None for loading from url)
        guidance_scale: float = 7.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        camera_condition_type: str = "rotation"
        view_dependent_prompting: bool = False

        n_view: int = 4
        image_size: int = 256
        recon_loss: bool = True
        recon_std_rescale: float = 0.5

        reward_model: str = "hpsv2-score"  # ['hpsv2-score', 'imagereward-score', 'brique-score', 'reward3d-score']
        beta_dpo: float = 0

        # AI Feedback
        use_ai_feedback: bool = False
        neg_prompt: bool = False
        ai_start_iter: int = 600
        ai_prob: float = 0.1

        # Parallel scoring (batched q_sample, reward + AI feedback for both candidates)
        parallel_scoring: bool = True

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Multiview Diffusion ...")
        self.global_step = 0
        self.model = build_model(self.cfg.model_name, ckpt_path=self.cfg.ckpt_path)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = 1000
        min_step_percent = C(self.cfg.min_step_percent, 0, 0)
        max_step_percent = C(self.cfg.max_step_percent, 0, 0)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
        self.grad_clip_val: Optional[float] = None
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        # reward model
        self.score_func = threestudio.find(self.cfg.reward_model)({})
        print(f"Initialize Reward Model from {self.cfg.reward_model}")

        # AI feedback model
        if self.cfg.use_ai_feedback:
            self.ai_score_func = threestudio.find("qwen-score")({})

        self.to(self.device)
        # self.model = self.model.to(self.weights_dtype)

        threestudio.info(f"Loaded Multiview Diffusion!")

    def get_camera_cond(
        self,
        camera: Float[Tensor, "B 4 4"],
        fovy=None,
    ):
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        if self.cfg.camera_condition_type == "rotation":  # normalized camera
            camera = normalize_camera(camera)
            camera = camera.flatten(start_dim=1)
        else:
            raise NotImplementedError(f"Unknown camera_condition_type={self.cfg.camera_condition_type}")
        return camera

    def encode_images(
        self, imgs: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    def decode_latents(
        self,
        latents
    ):
        input_dtype = latents.dtype
        x_sample = self.model.decode_first_stage(latents)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        return x_sample.to(input_dtype)

    def process_noise_pred(
        self,
        noise_pred,
        latents_noisy,
        t,
        prompt,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        noise_pred_text, noise_pred_null = noise_pred.chunk(2)
        noise_pred = noise_pred_null + self.cfg.guidance_scale * (noise_pred_text - noise_pred_null)
        pred_original_sample = self.model.predict_start_from_noise(latents_noisy, t, noise_pred)
        hat_x_t = self.decode_latents(pred_original_sample)
        score = self.score_func(
            hat_x_t, prompt,
            prompt_utils, elevation, azimuth, camera_distances,
        )
        return score, noise_pred, noise_pred_text, hat_x_t

    def process_and_score(
        self,
        noise_pred_1: Float[Tensor, "2B ..."],
        noise_pred_2: Float[Tensor, "2B ..."],
        latents_noisy_1: Float[Tensor, "B ..."],
        latents_noisy_2: Float[Tensor, "B ..."],
        t: Float[Tensor, ""],
        prompt: str,
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):
        """
        Batched variant of process_noise_pred for the two candidates.

        - Each noise_pred_* is [2B, ...] (text + null).
        - Each latents_noisy_* is [B, ...].
        - We:
          * split (text, null) per candidate,
          * apply CFG separately,
          * concatenate latents and guided noises,
          * decode and score in a single batched pass.
        """

        # Candidate 1: split text / null and apply CFG
        noise_pred_text_1, noise_pred_null_1 = noise_pred_1.chunk(2)
        guided_1 = noise_pred_null_1 + self.cfg.guidance_scale * (
            noise_pred_text_1 - noise_pred_null_1
        )

        # Candidate 2: split text / null and apply CFG
        noise_pred_text_2, noise_pred_null_2 = noise_pred_2.chunk(2)
        guided_2 = noise_pred_null_2 + self.cfg.guidance_scale * (
            noise_pred_text_2 - noise_pred_null_2
        )

        # Concatenate latents & guided noise for a single decode+score pass
        latents_noisy_all = torch.cat([latents_noisy_1, latents_noisy_2], dim=0)
        guided_all = torch.cat([guided_1, guided_2], dim=0)

        pred_original_all = self.model.predict_start_from_noise(
            latents_noisy_all, t, guided_all
        )
        hat_x_t_all = self.decode_latents(pred_original_all)

        # Batched reward scoring
        score_all = self.score_func(
            hat_x_t_all,
            prompt,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
        )

        # Split back to per-candidate tensors
        hat_x_t_1, hat_x_t_2 = hat_x_t_all.chunk(2, dim=0)
        score_1, score_2 = score_all.chunk(2, dim=0)

        return (
            score_1,
            guided_1,
            noise_pred_text_1,
            hat_x_t_1,
            score_2,
            guided_2,
            noise_pred_text_2,
            hat_x_t_2,
        )

    def forward(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        c2w: Float[Tensor, "B 4 4"],
        rgb_as_latents: bool = False,
        fovy=None,
        timestep=None,
        text_embeddings=None,
        input_is_latent=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]
        camera = c2w
        prompt = prompt_utils.prompt

        rgb = rgb.to(self.weights_dtype)
        rgb_BCHW = rgb.permute(0, 3, 1, 2)

        if text_embeddings is None:
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            pos_text_embeddings, null_text_embeddings = text_embeddings.chunk(2)
            text_embeddings = torch.cat(
                [pos_text_embeddings, null_text_embeddings, pos_text_embeddings, null_text_embeddings]
            )

        if input_is_latent:
            latents = rgb_BCHW
        else:
            latents: Float[Tensor, "B 4 64 64"]
            if rgb_as_latents:
                latents = F.interpolate(rgb_BCHW, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
            else:
                # interp to 512x512 to be fed into vae.
                pred_rgb = F.interpolate(
                    rgb_BCHW,
                    (self.cfg.image_size, self.cfg.image_size),
                    mode='bilinear',
                    align_corners=False,
                )
                # encode image into latents with vae, requires grad!
                latents = self.encode_images(pred_rgb)

        # sample timestep
        if timestep is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=latents.device)
        else:
            assert 0 <= timestep < self.num_train_timesteps
            t = torch.full([1], timestep, dtype=torch.long, device=latents.device)
        t_expand = t.repeat(text_embeddings.shape[0])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # --- q_sample for candidate 1 and 2 (batched or separate depending on parallel_scoring) ---
            if self.cfg.parallel_scoring:
                # replicate latents along batch dimension: [B, ...] -> [2B, ...]
                latents_rep = torch.cat([latents, latents], dim=0)
                # sample independent noises for the 2B latents
                noise_all = torch.randn_like(latents_rep)
                # single q_sample call for both candidates
                latents_noisy_all = self.model.q_sample(latents_rep, t, noise_all)
                # split back into candidate 1 / 2
                latents_noisy_1, latents_noisy_2 = latents_noisy_all.chunk(2, dim=0)
                noise_1, noise_2 = noise_all.chunk(2, dim=0)
            else:
                # original behavior: separate q_sample calls
                noise_1 = torch.randn_like(latents)
                latents_noisy_1 = self.model.q_sample(latents, t, noise_1)
                noise_2 = torch.randn_like(latents)
                latents_noisy_2 = self.model.q_sample(latents, t, noise_2)

            # pred noise
            latent_model_input = torch.cat([latents_noisy_1] * 2 + [latents_noisy_2] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera, fovy)
                camera = camera.repeat(4, 1).to(text_embeddings)
                context = {"context": text_embeddings, "camera": camera, "num_frames": self.cfg.n_view}
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

            noise_pred_1, noise_pred_2 = noise_pred.chunk(2)

            # --- Reward scoring (parallel vs sequential) ---
            if self.cfg.parallel_scoring:
                (
                    score_1,
                    noise_pred_1,
                    noise_pred_text_1,
                    hat_x_t_1,
                    score_2,
                    noise_pred_2,
                    noise_pred_text_2,
                    hat_x_t_2,
                ) = self.process_and_score(
                    noise_pred_1,
                    noise_pred_2,
                    latents_noisy_1,
                    latents_noisy_2,
                    t,
                    prompt,
                    prompt_utils,
                    elevation,
                    azimuth,
                    camera_distances,
                )
            else:
                score_1, noise_pred_1, noise_pred_text_1, hat_x_t_1 = self.process_noise_pred(
                    noise_pred_1, latents_noisy_1, t, prompt, prompt_utils, elevation, azimuth, camera_distances
                )
                score_2, noise_pred_2, noise_pred_text_2, hat_x_t_2 = self.process_noise_pred(
                    noise_pred_2, latents_noisy_2, t, prompt, prompt_utils, elevation, azimuth, camera_distances
                )

            win_mask = score_1 >= score_2

            random_value = random.random()
            if (
                self.cfg.use_ai_feedback
                and self.global_step > self.cfg.ai_start_iter
                and random_value < self.cfg.ai_prob
            ):
                # Batched AI feedback if requested
                if self.cfg.parallel_scoring:
                    hat_x_t_all = torch.cat([hat_x_t_1, hat_x_t_2], dim=0)
                    ai_score_all = self.ai_score_func(hat_x_t_all, prompt)
                    ai_score_1, ai_score_2 = ai_score_all.chunk(2, dim=0)
                else:
                    ai_score_1 = self.ai_score_func(hat_x_t_1, prompt)
                    ai_score_2 = self.ai_score_func(hat_x_t_2, prompt)

                if self.cfg.neg_prompt:
                    win_mask_ai = ai_score_1 < ai_score_2
                else:
                    win_mask_ai = ai_score_1 > ai_score_2
                win_mask = torch.logical_or(win_mask, win_mask_ai)

            win_mask_1, win_mask_2 = win_mask, torch.logical_not(win_mask)

            score_gap, score_lose = torch.zeros_like(score_1), torch.zeros_like(score_1)
            score_gap[win_mask_1] = (score_1[win_mask_1] - score_2[win_mask_1])
            score_gap[win_mask_2] = (score_2[win_mask_2] - score_1[win_mask_2])
            beta_dpo = self.cfg.beta_dpo
            grad = self.calc_grad_smooth(batch_size, noise_1, noise_2, noise_pred_1, noise_pred_2, noise_pred_text_1, noise_pred_text_2, win_mask_1, win_mask_2, score_gap, beta_dpo)

            w = (1 - self.model.alphas_cumprod[t])
            grad = w * grad

        # clip grad for stable training?
        if self.grad_clip_val is not None:
            grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum") / latents.shape[0]

        return {
            "loss_sds": loss,
            "grad_norm": grad.norm(),
        }

    def calc_grad(self, batch_size, noise_1, noise_2, noise_pred_1, noise_pred_2, noise_pred_text_1, noise_pred_text_2, win_mask_1, win_mask_2, score_gap, beta_dpo):
        gap_mask = (score_gap < beta_dpo).int().view(batch_size, 1, 1, 1)
        mask = gap_mask.to(self.device)

        reward = torch.zeros_like(noise_pred_1)
        reward[win_mask_1] = (noise_pred_1[win_mask_1] - noise_1[win_mask_1]) - (
                noise_pred_text_2[win_mask_1] - noise_2[win_mask_1]
            )
        reward[win_mask_2] = (noise_pred_2[win_mask_2] - noise_2[win_mask_2]) - (
                noise_pred_text_1[win_mask_2] - noise_1[win_mask_2]
            )
            # sds
        reward_sds = torch.zeros_like(noise_pred_1)
        reward_sds[win_mask_1] = noise_pred_1[win_mask_1]
        reward_sds[win_mask_2] = noise_pred_2[win_mask_2]
        grad = mask * reward_sds + (1 - mask) * reward
        return grad

    def calc_grad_smooth(self, batch_size, noise_1, noise_2, noise_pred_1, noise_pred_2, noise_pred_text_1, noise_pred_text_2, win_mask_1, win_mask_2, score_gap, beta_dpo):
        if beta_dpo > 0.0:
            # Linear ramp: small gaps -> more SDS, large gaps -> more DPO
            dpo_weight = torch.sigmoid(score_gap / beta_dpo) 
        else:
            # Backwards-compatible: beta_dpo <= 0 means "always use DPO-style reward"
            dpo_weight = torch.ones_like(score_gap)

        # [B] -> [B, 1, 1, 1] for broadcasting over latent shape
        dpo_weight = dpo_weight.view(batch_size, 1, 1, 1).to(self.device)
        sds_weight = 1.0 - dpo_weight

        # reward: "DPO-ish" gradient; reward_sds: pure SDS gradient
        reward = torch.zeros_like(noise_pred_1)
        reward[win_mask_1] = (noise_pred_1[win_mask_1] - noise_1[win_mask_1]) - (
            noise_pred_text_2[win_mask_1] - noise_2[win_mask_1]
        )
        reward[win_mask_2] = (noise_pred_2[win_mask_2] - noise_2[win_mask_2]) - (
            noise_pred_text_1[win_mask_2] - noise_1[win_mask_2]
        )

        reward_sds = torch.zeros_like(noise_pred_1)
        reward_sds[win_mask_1] = noise_pred_1[win_mask_1]
        reward_sds[win_mask_2] = noise_pred_2[win_mask_2]

        # Smooth mix: per-sample convex combination
        grad = sds_weight * reward_sds + dpo_weight * reward
        return grad

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.global_step = global_step
        min_step_percent = C(self.cfg.min_step_percent, epoch, global_step)
        max_step_percent = C(self.cfg.max_step_percent, epoch, global_step)
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)
