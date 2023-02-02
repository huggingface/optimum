import torch
import torch.nn as nn

from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from schedulers.scheduling_pndm import ScriptablePNDMScheduler


def get_scheduler_args(batch_size, num_channels_latents, width, height, num_images_per_prompt, vae_scale_factor):
    ets_buffer = torch.zeros(
        4,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height // vae_scale_factor,
        width // vae_scale_factor,
    )
    set_ets = torch.tensor(0, dtype=torch.int64)
    counter = torch.tensor(0, dtype=torch.int64)
    cur_sample_buffer = torch.zeros(1, 4, 64, 64, dtype=torch.float32)

    return (ets_buffer, set_ets, counter, cur_sample_buffer)

class ScriptableStableDiffusionPipeline(nn.Module):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`ScriptablePNDMScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: ScriptablePNDMScheduler,
    ):
        super().__init__()

        if scheduler.steps_offset != 1:
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if scheduler.clip_sample is True:
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.scheduler = scheduler

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        assert all(device == self.vae.device for device in [self.text_encoder.device, self.unet.device])
        self.device = self.vae.device

        self.sample_size = torch.tensor(self.unet.config.sample_size, dtype=torch.int64)
        self.unet_in_channels = torch.tensor(self.unet.in_channels, dtype=torch.int64)

        self.height = -1
        self.width = -1
        self.guidance_scale = 7.5
        self.num_images_per_prompt = 1

        # TODO: remove
        torch.manual_seed(42)
        num_channels_latents = self.unet_in_channels
        height = self.sample_size * self.vae_scale_factor
        width = self.sample_size * self.vae_scale_factor
        # batch size = 1 fixed!
        shape = (1, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        #self.deterministic_latents = torch.randn(shape, device="cuda", dtype=torch.float16)
        self.deterministic_latents = torch.randn(shape, device="cpu", dtype=torch.float32)
        

    def _encode_prompt(
        self,
        text_input_ids: torch.Tensor,
        uncond_text_input_ids: torch.Tensor,
        num_images_per_prompt: int,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
        """
        if text_input_ids is None:
            raise ValueError("text_input_ids should be set")

        batch_size = text_input_ids.shape[0]

        prompt_embeds = self.text_encoder(text_input_ids)

        # TODO: is it really ok to remove dtype cast?
        prompt_embeds = prompt_embeds[0]  # .to(dtype=self.text_encoder.dtype)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        negative_prompt_embeds = self.text_encoder(uncond_text_input_ids)
        negative_prompt_embeds = negative_prompt_embeds[0]

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        # TODO: is it really ok to remove dtype cast?
        negative_prompt_embeds = negative_prompt_embeds  # .to(dtype=self.text_encoder.dtype)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents: torch.Tensor):
        latents = 1 / 0.18215 * latents

        image = self.vae_decoder(latents)
        #image = self.vae.decode(latents, return_dict=False)
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            image = self.vae_decoder(latents)
        else:
            image = self.vae.decode(latents, return_dict=False)
        """

        image = image[0]

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

        image = image.permute(0, 2, 3, 1)
        return image

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        latents = torch.randn(shape, device=device, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def forward(
        self,
        text_input_ids: torch.Tensor,
        uncond_text_input_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        r"""
        Performs reversed stable diffusion.

        Args:
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        # NOTE: we don't support passing generator :(
        # NOTE: we don't support DDIMScheduler because eta is ignored
        # TODO: avoid hardcoding height, width, guidance_scale, num_images_per_prompt

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        # extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if text_input_ids is None:
            raise ValueError("text_input_ids should be passed")

        # 0. Default height and width to unet
        height = self.sample_size * self.vae_scale_factor
        width = self.sample_size * self.vae_scale_factor

        # height = height or self.unet.config.sample_size * self.vae_scale_factor
        # width = width or self.unet.config.sample_size * self.vae_scale_factor

        batch_size = text_input_ids.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # assert self.guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            text_input_ids,
            uncond_text_input_ids,
            self.num_images_per_prompt,
        )

        device = prompt_embeds.device

        """
        # 5. Prepare latent variables
        num_channels_latents = self.unet_in_channels
        latents = self.prepare_latents(
            batch_size * self.num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
        )
        """
        # TODO: remove this horror
        num_channels_latents = self.unet_in_channels
        latents = self.deterministic_latents

        scheduler_args = get_scheduler_args(batch_size, num_channels_latents, width, height, self.num_images_per_prompt, self.vae_scale_factor)

        # TODO: put this out of the forward, as this is scheduler specific
        # 4D buffer rolled at each step
        """
        ets_buffer = torch.zeros(
            4,
            batch_size * self.num_images_per_prompt,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        ).to(device, dtype=prompt_embeds.dtype)
        set_ets = torch.tensor(0, dtype=torch.int64).to(device)
        counter = torch.tensor(0, dtype=torch.int64).to(device)
        cur_sample_buffer = torch.zeros(1, 4, 64, 64, dtype=torch.float32)
        # self.scheduler.cur_sample = torch.rand(1, 4, 64, 64, dtype=torch.float32)
        """

        # 7. Denoising loop
        # TODO: what is self.scheduler.order?
        for _, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents, latents])

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            # shape: 2, 4, 64, 64
            noise_pred = self.unet(latent_model_input, t, prompt_embeds)[0]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            #latents = noise_pred
            
            # compute the previous noisy sample x_t -> x_t-1
            # t is used as an index here, and should be on CPU
            latents, scheduler_args = self.scheduler.step(
                noise_pred, t.to("cpu"), latents, *scheduler_args
            )  # a tuple is returned by step

        # 8. Post-processing
        image = self.decode_latents(latents)

        return (image,) 
