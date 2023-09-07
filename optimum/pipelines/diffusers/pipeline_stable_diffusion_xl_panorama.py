from optimum.onnxruntime.modeling_diffusion import ORTStableDiffusionXLPipelineBase
from optimum.pipelines.diffusers.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipelineMixin
from optimum.pipelines.diffusers.pipeline_utils import preprocess, rescale_noise_cfg
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
import logging
from typing import Any, Optional, List, Union, Tuple, Callable, Dict
import torch
import numpy as np
import PIL
import inspect

logger = logging.getLogger(__name__)


DEFAULT_WINDOW = 64
DEFAULT_STRIDE = 16


class StableDiffusionXLPanoramaPipelineMixin(StableDiffusionXLImg2ImgPipelineMixin):
    def __init__(
            self,
            *args,
            window: int = DEFAULT_WINDOW,
            stride: int = DEFAULT_STRIDE,
            **kwargs,
    ):
        super().__init__(self, *args, **kwargs)

        self.window = window
        self.stride = stride


    def set_window_size(self, window: int, stride: int):
        self.window = window
        self.stride = stride


    def get_views(self, panorama_height, panorama_width, window_size, stride):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        panorama_height /= 8
        panorama_width /= 8

        num_blocks_height = abs((panorama_height - window_size) // stride) + 1
        num_blocks_width = abs((panorama_width - window_size) // stride) + 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        logger.debug(
            "panorama generated %s views, %s by %s blocks",
            total_num_blocks,
            num_blocks_height,
            num_blocks_width,
        )

        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride)
            h_end = h_start + window_size
            w_start = int((i % num_blocks_width) * stride)
            w_end = w_start + window_size
            views.append((h_start, h_end, w_start, w_end))

        return views


    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents_img2img(self, image, timestep, batch_size, num_images_per_prompt, dtype, generator=None):
        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image
        else:
            init_latents = self.vae_encoder(sample=image)[0] * self.vae_decoder.config.get("scaling_factor", 0.18215)

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = np.concatenate([init_latents] * additional_image_per_prompt, axis=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = np.concatenate([init_latents], axis=0)

        # add noise to latents using the timesteps
        noise = generator.randn(*init_latents.shape).astype(dtype)
        init_latents = self.scheduler.add_noise(
            torch.from_numpy(init_latents), torch.from_numpy(noise), torch.from_numpy(timestep)
        )
        return init_latents.numpy()


    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents_text2img(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = generator.randn(*shape).astype(dtype)
        elif latents.shape != shape:
            raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * np.float64(self.scheduler.init_noise_sigma)

        return latents


    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        extra_step_kwargs = {}

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        return extra_step_kwargs


    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.__call__
    def text2img(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        pooled_prompt_embeds: Optional[np.ndarray] = None,
        negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`Optional[Union[str, List[str]]]`, defaults to None):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`Optional[int]`, defaults to None):
                The height in pixels of the generated image.
            width (`Optional[int]`, defaults to None):
                The width in pixels of the generated image.
            num_inference_steps (`int`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to 5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`Optional[np.random.RandomState]`, defaults to `None`)::
                A np.random.RandomState to make generation deterministic.
            latents (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] instead of a
                plain tuple.
            callback (Optional[Callable], defaults to `None`):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            guidance_rescale (`float`, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # 0. Default height and width to unet
        height = height or self.unet.config["sample_size"] * self.vae_scale_factor
        width = width or self.unet.config["sample_size"] * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            1.0,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latents = self.prepare_latents_text2img(
            batch_size * num_images_per_prompt,
            self.unet.config.get("in_channels", 4),
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = (original_size + crops_coords_top_left + target_size,)
        add_time_ids = np.array(add_time_ids, dtype=prompt_embeds.dtype)

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate((negative_prompt_embeds, prompt_embeds), axis=0)
            add_text_embeds = np.concatenate((negative_pooled_prompt_embeds, add_text_embeds), axis=0)
            add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)
        add_time_ids = np.repeat(add_time_ids, batch_size * num_images_per_prompt, axis=0)

        # Adapted from diffusers to extend it for other runtimes than ORT
        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

        # 8. Panorama additions
        views = self.get_views(height, width, self.window, self.stride)
        count = np.zeros_like(latents)
        value = np.zeros_like(latents)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            count.fill(0)
            value.fill(0)

            for h_start, h_end, w_start, w_end in views:
                # get the latents corresponding to the current view coordinates
                latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents_for_view] * 2) if do_classifier_free_guidance else latents_for_view
                latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
                latent_model_input = latent_model_input.cpu().numpy()

                # predict the noise residual
                timestep = np.array([t], dtype=timestep_dtype)
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    text_embeds=add_text_embeds,
                    time_ids=add_time_ids,
                )
                noise_pred = noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(
                    torch.from_numpy(noise_pred), t, torch.from_numpy(latents_for_view), **extra_step_kwargs
                )
                latents_view_denoised = scheduler_output.prev_sample.numpy()

                value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count[:, :, h_start:h_end, w_start:w_end] += 1

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            latents = np.where(count > 0, value / count, value)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if output_type == "latent":
            image = latents
        else:
            latents = latents / self.vae_decoder.config.get("scaling_factor", 0.18215)
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )
            image = self.watermark.apply_watermark(image)

            # TODO: add image_processor
            image = np.clip(image / 2 + 0.5, 0, 1).transpose((0, 2, 3, 1))

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


    # Adapted from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.__call__
    def img2img(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        image: Union[np.ndarray, PIL.Image.Image] = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        pooled_prompt_embeds: Optional[np.ndarray] = None,
        negative_pooled_prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`Optional[Union[str, List[str]]]`, defaults to None):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            image (`Union[np.ndarray, PIL.Image.Image]`):
                `Image`, or tensor representing an image batch which will be upscaled.
            strength (`float`, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to 5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. Ignored when not using guidance (i.e., ignored if `guidance_scale`
                is less than `1`).
            num_images_per_prompt (`int`, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`Optional[np.random.RandomState]`, defaults to `None`)::
                A np.random.RandomState to make generation deterministic.
            latents (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] instead of a
                plain tuple.
            callback (Optional[Callable], defaults to `None`):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            guidance_rescale (`float`, defaults to 0.7):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Check inputs. Raise error if not correct
        self.check_inputs(prompt, strength, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 1. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if generator is None:
            generator = np.random

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode input prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

        # 3. Preprocess image
        image = preprocess(image)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
        latent_timestep = np.repeat(timesteps[:1], batch_size * num_images_per_prompt, axis=0)
        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

        latents_dtype = prompt_embeds.dtype
        image = image.astype(latents_dtype)

        # 5. Prepare latent variables
        latents = self.prepare_latents_img2img(
            image, latent_timestep, batch_size, num_images_per_prompt, latents_dtype, generator
        )

        # 6. Prepare extra step kwargs
        extra_step_kwargs = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            dtype=prompt_embeds.dtype,
        )

        if do_classifier_free_guidance:
            prompt_embeds = np.concatenate((negative_prompt_embeds, prompt_embeds), axis=0)
            add_text_embeds = np.concatenate((negative_pooled_prompt_embeds, add_text_embeds), axis=0)
            add_time_ids = np.concatenate((add_time_ids, add_time_ids), axis=0)
        add_time_ids = np.repeat(add_time_ids, batch_size * num_images_per_prompt, axis=0)

        # 8. Panorama additions
        views = self.get_views(height, width, self.window, self.stride)
        count = np.zeros_like(latents)
        value = np.zeros_like(latents)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            count.fill(0)
            value.fill(0)

            for h_start, h_end, w_start, w_end in views:
                # get the latents corresponding to the current view coordinates
                latents_for_view = latents[:, :, h_start:h_end, w_start:w_end]

                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents_for_view] * 2) if do_classifier_free_guidance else latents_for_view
                latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
                latent_model_input = latent_model_input.cpu().numpy()

                # predict the noise residual
                timestep = np.array([t], dtype=timestep_dtype)
                noise_pred = self.unet(
                    sample=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    text_embeds=add_text_embeds,
                    time_ids=add_time_ids,
                )
                noise_pred = noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(
                    torch.from_numpy(noise_pred), t, torch.from_numpy(latents_for_view), **extra_step_kwargs
                )
                latents_view_denoised = scheduler_output.prev_sample.numpy()

                value[:, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count[:, :, h_start:h_end, w_start:w_end] += 1

            # take the MultiDiffusion step. Eq. 5 in MultiDiffusion paper: https://arxiv.org/abs/2302.08113
            latents = np.where(count > 0, value / count, value)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if output_type == "latent":
            image = latents
        else:
            latents = latents / self.vae_decoder.config.get("scaling_factor", 0.18215)
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )
            image = self.watermark.apply_watermark(image)

            # TODO: add image_processor
            image = np.clip(image / 2 + 0.5, 0, 1).transpose((0, 2, 3, 1))

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


    def __call__(
        self,
        *args,
        **kwargs,
    ):
        if "image" in kwargs or (
            len(args) > 1
            and (
                isinstance(args[1], np.ndarray) or isinstance(args[1], PIL.Image.Image)
            )
        ):
            logger.debug("running img2img panorama XL pipeline")
            return self.img2img(*args, **kwargs)
        else:
            logger.debug("running txt2img panorama XL pipeline")
            return self.text2img(*args, **kwargs)
