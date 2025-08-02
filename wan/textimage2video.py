# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


def clear_cache():
    gc.collect()
    if torch.backends.mps.is_built():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class WanTI2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        t5_quant=False,
        vae_tile_size=None,
        disk_offload=False,
        mps_ram="10GB",
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        if isinstance(device_id, torch.device):
            self.device = device_id
        elif device_id == "mps" or (isinstance(device_id, int) and device_id == -1):
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(f"cuda:{device_id}")

        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.t5_fsdp = t5_fsdp
        self.convert_model_dtype = convert_model_dtype

        self.t5_quant = t5_quant
        self.vae_tile_size = vae_tile_size
        self.disk_offload = disk_offload
        self.mps_ram = mps_ram

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size

        logging.info(f"Creating WanModel from {checkpoint_dir}")

        self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if self.convert_model_dtype:
            model.to(self.param_dtype)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        vae_model_z_dim = 48
        target_shape = (
            vae_model_z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        logging.info("Loading text encoder model.")
        if self.t5_quant:
            from .modules.t5_gguf import run_llama_embedding

            checkpoint_path = os.path.join(
                self.checkpoint_dir, self.config.t5_quant_checkpoint
            )
            context = [
                torch.from_numpy(run_llama_embedding(checkpoint_path, input_prompt)).to(
                    self.device
                )
            ]
            context_null = [
                torch.from_numpy(run_llama_embedding(checkpoint_path, n_prompt)).to(
                    self.device
                )
            ]
        else:
            from .modules.t5 import T5EncoderModel

            self.text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=os.path.join(
                    self.checkpoint_dir, self.config.t5_checkpoint
                ),
                tokenizer_path=os.path.join(
                    self.checkpoint_dir, self.config.t5_tokenizer
                ),
                shard_fn=None,
            )

            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
            else:
                context = self.text_encoder([input_prompt], torch.device("cpu"))
                context_null = self.text_encoder([n_prompt], torch.device("cpu"))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            if offload_model:
                del self.text_encoder
                logging.info("Remove text encoder model.")
                clear_cache()

        logging.info("Loading WanModel")
        if self.disk_offload:
            logging.info("Use disk offload.")
            self.model = WanModel.from_pretrained(
                self.checkpoint_dir,
                device_map="auto",
                max_memory={"mps": self.mps_ram, "cpu": "0.5GB"},
                offload_folder="disk_offload",
                offload_state_dict=True,
            )
        else:
            self.model = WanModel.from_pretrained(self.checkpoint_dir)
            self.model.to(self.device)
        self.model = self._configure_model(self.model)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
            amp.autocast(  # type: ignore
                device_type=str(self.device), dtype=self.param_dtype
            ),
            torch.no_grad(),
            no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if offload_model:
                clear_cache()
            logging.info("Start generation loop.")
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                del noise_pred, noise_pred_cond, noise_pred_uncond, temp_x0
                if offload_model:
                    clear_cache()
            logging.info("End generation loop.")
            x0 = latents
            if offload_model:
                del self.model
                logging.info("Remove WanModel.")
                clear_cache()
            if self.rank == 0:
                logging.info("Loading VAE model.")
                self.vae = Wan2_2_VAE(
                    vae_pth=os.path.join(
                        self.checkpoint_dir, self.config.vae_checkpoint
                    ),
                    device=self.device,
                    tile_size=self.vae_tile_size,
                )
                logging.info("Decoding video frames.")
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            del self.vae
            clear_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return videos[0] if self.rank == 0 else None

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        vae_model_z_dim = 48
        noise = torch.randn(
            vae_model_z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        logging.info("Loading text encoder model.")
        if self.t5_quant:
            from .modules.t5_gguf import run_llama_embedding

            checkpoint_path = os.path.join(
                self.checkpoint_dir, self.config.t5_quant_checkpoint
            )
            context = [
                torch.from_numpy(run_llama_embedding(checkpoint_path, input_prompt)).to(
                    self.device
                )
            ]
            context_null = [
                torch.from_numpy(run_llama_embedding(checkpoint_path, n_prompt)).to(
                    self.device
                )
            ]
        else:
            from .modules.t5 import T5EncoderModel

            self.text_encoder = T5EncoderModel(
                text_len=self.config.text_len,
                dtype=self.config.t5_dtype,
                device=torch.device("cpu"),
                checkpoint_path=os.path.join(
                    self.checkpoint_dir, self.config.t5_checkpoint
                ),
                tokenizer_path=os.path.join(
                    self.checkpoint_dir, self.config.t5_tokenizer
                ),
                shard_fn=None,
            )

            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
            else:
                context = self.text_encoder([input_prompt], torch.device("cpu"))
                context_null = self.text_encoder([n_prompt], torch.device("cpu"))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            if offload_model:
                del self.text_encoder
                logging.info("Remove text encoder model.")
                clear_cache()

        logging.info("Loading VAE model.")
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(self.checkpoint_dir, self.config.vae_checkpoint),
            device=self.device,
            tile_size=self.vae_tile_size,
        )
        logging.info("Encoding input image.")
        z = self.vae.encode([img])
        if offload_model:
            del img, self.vae
            clear_cache()

        @contextmanager
        def noop_no_sync():
            yield

        logging.info("Loading WanModel")
        if self.disk_offload:
            logging.info("Use disk offload.")
            self.model = WanModel.from_pretrained(
                self.checkpoint_dir,
                device_map="auto",
                max_memory={"mps": self.mps_ram, "cpu": "0.5GB"},
                offload_folder="disk_offload",
                offload_state_dict=True,
            )
        else:
            self.model = WanModel.from_pretrained(self.checkpoint_dir)
            self.model.to(self.device)
        self.model = self._configure_model(self.model)

        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        # evaluation mode
        with (
            amp.autocast(device_type=str(self.device), dtype=self.param_dtype),
            torch.no_grad(),
            no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model:
                clear_cache()

            logging.info("Start generation loop.")

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent]
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null
                )[0]
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del (
                    latent_model_input,
                    timestep,
                    noise_pred,
                    noise_pred_cond,
                    noise_pred_uncond,
                    temp_x0,
                )

            logging.info("End generation loop.")

            if offload_model:
                del self.model
                logging.info("Remove WanModel.")
                clear_cache()

            if self.rank == 0:
                logging.info("Loading VAE model.")
                self.vae = Wan2_2_VAE(
                    vae_pth=os.path.join(
                        self.checkpoint_dir, self.config.vae_checkpoint
                    ),
                    device=self.device,
                    tile_size=self.vae_tile_size,
                )
                logging.info("Decoding video frames.")
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            del self.vae
            clear_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        return videos[0] if self.rank == 0 else None
