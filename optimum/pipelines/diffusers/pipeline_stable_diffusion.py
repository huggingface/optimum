#  Copyright 2023 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import inspect
import logging
from typing import Callable, List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput

from .pipeline_utils import DiffusionPipelineMixin, rescale_noise_cfg


logger = logging.getLogger(__name__)


class StableDiffusionPipelineMixin(DiffusionPipelineMixin):
    # Copied from https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L114
    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int,
        do_classifier_free_guidance: bool,
        negative_prompt: Optional[Union[str, list]],
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`Union[str, List[str]]`):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`Optional[Union[str, list]]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`Optional[np.ndarray]`, defaults to `None`):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]

        prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )
            negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]

        if do_classifier_free_guidance:
            negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    # Copied from https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L217
    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int],
        width: Optional[int],
        callback_steps: int,
        negative_prompt: Optional[str] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, generator, latents=None):
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

    # Adapted from https://github.com/huggingface/diffusers/blob/v0.17.1/src/diffusers/pipelines/stable_diffusion/pipeline_onnx_stable_diffusion.py#L264
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        guidance_rescale: float = 0.0,
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
            guidance_scale (`float`, defaults to 7.5):
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
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (Optional[Callable], defaults to `None`):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            guidance_rescale (`float`, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        height = height or self.unet.config.get("sample_size", 64) * self.vae_scale_factor
        width = width or self.unet.config.get("sample_size", 64) * self.vae_scale_factor

        # check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # define call parameters
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

        prompt_embeds = self._encode_prompt(
            prompt,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.config.get("in_channels", 4),
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Adapted from diffusers to extend it for other runtimes than ORT
        timestep_dtype = self.unet.input_dtype.get("timestep", np.float32)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
            latent_model_input = latent_model_input.cpu().numpy()

            # predict the noise residual
            timestep = np.array([t], dtype=timestep_dtype)
            noise_pred = self.unet(sample=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds)
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
                torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs
            )
            latents = scheduler_output.prev_sample.numpy()

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        else:
            latents /= self.vae_decoder.config.get("scaling_factor", 0.18215)
            # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
            image = np.concatenate(
                [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
            )
            image, has_nsfw_concept = self.run_safety_checker(image)

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def run_safety_checker(self, image: np.ndarray):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            feature_extractor_input = self.image_processor.numpy_to_pil(image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors="np"
            ).pixel_values.astype(image.dtype)
            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)

        return image, has_nsfw_concept
