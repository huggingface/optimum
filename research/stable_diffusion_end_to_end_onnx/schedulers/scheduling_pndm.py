from diffusers import DiffusionPipeline
import torch
import torch.nn as nn
from diffusers.schedulers import SchedulerMixin


pipeline = DiffusionPipeline.from_pretrained(
    "hf-internal-testing/tiny-stable-diffusion-torch", low_cpu_mem_usage=False
)

pipeline = pipeline.to("cpu")

##
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


@torch.jit.export
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class ScriptablePNDMScheduler(nn.Module):
    order = 1

    prk_timesteps: torch.Tensor  # Optional is non subscriptable
    plms_timesteps: torch.Tensor
    timesteps: torch.Tensor
    ets: List[torch.Tensor]
    cur_sample: torch.Tensor
    # ets_buffer: torch.Tensor

    def __init__(
        self,
        num_train_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        skip_prk_steps: bool,
        set_alpha_to_one: bool,
        prediction_type: str,
        steps_offset: int,
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        **kwargs
    ):
        super().__init__()

        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)

        # else:
        #     raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # For now we only support F-PNDM, i.e. the runge-kutta method
        # For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        # mainly at formula (9), (12), (13) and the Algorithm 2.
        self.pndm_order = 4

        # running values
        self.cur_model_output = torch.tensor(0)
        self.counter = torch.tensor(0, dtype=torch.int64)  # shouldn't it be reset?
        self.cur_sample = torch.rand(1, 4, 64, 64, dtype=torch.float32)

        # setable values
        self.num_inference_steps = None
        self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        self.prk_timesteps = None
        self.plms_timesteps = None
        self.timesteps = None

        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps
        self.steps_offset = steps_offset
        self.skip_prk_steps = skip_prk_steps

        self.clip_sample = kwargs["clip_sample"]

        # self.ets_buffer = torch.empty(4)  # this buffer should be initialized in the pipeline's forward call
        self.set_ets = torch.tensor(
            0, dtype=torch.int64
        )  # this attribute should be reset to 0 in the pipeline's forward call

    @torch.jit.export
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.Tensor,
        sample: torch.FloatTensor,
        ets_buffer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        This function calls `step_prk()` or `step_plms()` depending on the internal variable `counter`.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.counter < len(self.prk_timesteps) and not self.skip_prk_steps:
            return (self.step_prk(model_output=model_output, timestep=timestep, sample=sample), ets_buffer)
        else:
            return self.step_plms(model_output=model_output, timestep=timestep, sample=sample, ets_buffer=ets_buffer)

    @torch.jit.export
    def step_prk(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.Tensor,
        sample: torch.FloatTensor,
    ) -> torch.Tensor:

        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        diff_to_prev = 0 if self.counter % 2 else self.num_train_timesteps // self.num_inference_steps // 2
        prev_timestep = timestep - diff_to_prev
        timestep = self.prk_timesteps[self.counter // 4 * 4]

        if self.counter % 4 == 0:
            self.cur_model_output += 1 / 6 * model_output
            self.ets_buffer = torch.roll(self.ets_buffer, shifts=-1, dims=0)
            self.ets_buffer[3] = model_output
            self.cur_sample = sample
        elif (self.counter - 1) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 2) % 4 == 0:
            self.cur_model_output += 1 / 3 * model_output
        elif (self.counter - 3) % 4 == 0:
            model_output = self.cur_model_output + 1 / 6 * model_output
            self.cur_model_output = torch.tensor(0)

        # cur_sample should not be `None`
        # if self.cur_sample_initialized
        cur_sample = self.cur_sample if self.cur_sample != -1 else sample

        prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
        self.counter = self.counter + 1

        return prev_sample

    @torch.jit.export
    def step_plms(
        self,
        model_output: torch.FloatTensor,
        timestep: torch.Tensor,
        sample: torch.FloatTensor,
        ets_buffer: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step function propagating the sample with the linear multi-step method. This has one forward pass with multiple
        times to approximate the solution.

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.

        Returns:
            [`~scheduling_utils.SchedulerOutput`] or `tuple`: [`~scheduling_utils.SchedulerOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.

        """

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if not self.skip_prk_steps and len(self.ets) < 3:
            raise ValueError(
                f"This can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )
        """
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        # print("sample dtype:", sample.dtype)
        # print("self.cur_sample dtype:", self.cur_sample.dtype)
        # print("model_output dtype:", model_output.dtype)

        # print("prev_timestep dtype", prev_timestep.dtype)
        # print("timestep dtype", timestep.dtype)

        if self.counter != 1:
            ets_buffer = torch.roll(ets_buffer, shifts=-1, dims=0)
            ets_buffer[3] = model_output
            self.set_ets = self.set_ets + 1
        else:
            prev_timestep = timestep
            timestep = timestep + self.num_train_timesteps // self.num_inference_steps

        # print("prev_timestep dtype", prev_timestep.dtype)
        # print("timestep dtype", timestep.dtype)

        # print("model_output.shape before", model_output.shape)
        # print("ets_buffer[-1] shape", ets_buffer[-1].shape)
        if self.set_ets == 1 and self.counter == 0:
            model_output = model_output
            self.cur_sample = sample
        elif self.set_ets == 1 and self.counter == 1:
            model_output = (model_output + ets_buffer[-1]) / 2
            sample = self.cur_sample
            # TODO: to fix?
            # self.cur_sample = torch.tensor(-5., dtype=torch.float32)
        elif self.set_ets == 2:
            model_output = (3 * ets_buffer[-1] - ets_buffer[-2]) / 2
        elif self.set_ets == 3:
            model_output = (23 * ets_buffer[-1] - 16 * ets_buffer[-2] + 5 * ets_buffer[-3]) / 12
        else:
            model_output = (1 / 24) * (
                55 * ets_buffer[-1] - 59 * ets_buffer[-2] + 37 * ets_buffer[-3] - 9 * ets_buffer[-4]
            )

        # print("model_output.shape after", model_output.shape)

        #TODO: put back?
        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
        #prev_sample = sample
        self.counter = self.counter + 1

        return (prev_sample, ets_buffer)

    @torch.jit.export
    def _get_prev_sample(
        self, sample: torch.Tensor, timestep: torch.Tensor, prev_timestep: torch.Tensor, model_output: torch.Tensor
    ):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        # print("alpha_prod_t shape", alpha_prod_t.shape)
        # print("alpha_prod_t_prev shape", alpha_prod_t_prev.shape)
        # print("self.final_alpha_cumprod shape", self.final_alpha_cumprod.shape)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        """
        if self.prediction_type == "v_prediction":
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        elif self.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.prediction_type} must be one of `epsilon` or `v_prediction`"
            )
        """

        # corresponds to (α_(t−δ) - α_t) divided by
        # denominator of x_t in formula (9) and plus 1
        # Note: (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t))
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # corresponds to denominator of e_θ(x_t, t) in formula (9)
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # full formula (9)
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )

        return prev_sample

    @torch.jit.ignore
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
        self._timesteps += self.steps_offset

        if self.skip_prk_steps:
            # for some models like stable diffusion the prk steps can/should be skipped to
            # produce better results. When using PNDM with `self.config.skip_prk_steps` the implementation
            # is based on crowsonkb's PLMS sampler implementation: https://github.com/CompVis/latent-diffusion/pull/51
            self.prk_timesteps = np.array([])
            self.plms_timesteps = np.concatenate([self._timesteps[:-1], self._timesteps[-2:-1], self._timesteps[-1:]])[
                ::-1
            ].copy()
        else:
            prk_timesteps = np.array(self._timesteps[-self.pndm_order :]).repeat(2) + np.tile(
                np.array([0, self.num_train_timesteps // num_inference_steps // 2]), self.pndm_order
            )
            self.prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1].copy()
            self.plms_timesteps = self._timesteps[:-3][
                ::-1
            ].copy()  # we copy to avoid having negative strides which are not supported by torch.from_numpy

        timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]).astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)

        self.plms_timesteps = torch.from_numpy(self.plms_timesteps)
        self.prk_timesteps = torch.from_numpy(self.prk_timesteps)

        self.counter = torch.tensor(0)

    @torch.jit.export
    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample


if __name__ == "__main__":
    my_scheduler = ScriptablePNDMScheduler(**pipeline.scheduler.config)
    my_scheduler.set_timesteps(50, device="cpu")

    scripted_scheduler = torch.jit.script(my_scheduler)
    print(scripted_scheduler.step_plms.code)
