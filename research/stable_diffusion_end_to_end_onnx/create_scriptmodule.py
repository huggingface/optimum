from optimum.exporters.onnx.utils import get_stable_diffusion_models_for_export
from optimum.exporters import TasksManager

from diffusers import DiffusionPipeline
import torch

from scriptable_pipeline_stable_diffusion import ScriptableStableDiffusionPipeline

from utils import export_models, get_traced_submodules, StableDiffusionPreprocessor

from schedulers.scheduling_pndm import ScriptablePNDMScheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu",
    action="store_true",
    help="use to trace and script on GPU.",
)
args = parser.parse_args()

#model_name = "CompVis/stable-diffusion-v1-4"
model_name = "hf-internal-testing/tiny-stable-diffusion-torch"
pipeline = DiffusionPipeline.from_pretrained(model_name, low_cpu_mem_usage=False)

num_inference_steps = 50

if args.gpu:
    device= "cuda"
else:
    device = "cpu"

scriptable_scheduler = ScriptablePNDMScheduler(**pipeline.scheduler.config)
scriptable_scheduler.set_timesteps(num_inference_steps, device=device)

pipeline = ScriptableStableDiffusionPipeline(
    vae=pipeline.vae,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer,
    unet=pipeline.unet,
    scheduler=scriptable_scheduler,
)

if args.gpu:
    pipeline = pipeline.to(device)

models_and_onnx_configs = get_stable_diffusion_models_for_export(pipeline)

in_dummy_out_per_model = export_models(
    models_and_onnx_configs=models_and_onnx_configs,
)

print("Starting tracing...")
text_encoder_traced, unet_traced, vae_decoder_traced = get_traced_submodules(
    in_dummy_out_per_model, models_and_onnx_configs
)
print("Tracing done.")

preprocessor = StableDiffusionPreprocessor(
    pipeline.tokenizer,
    pipeline.text_encoder.config,
    do_classifier_free_guidance=True,
    scheduler=pipeline.scheduler,
)

script = True

if script:
    pipeline.vae_decoder = vae_decoder_traced
    pipeline.vae = None  # the VAE is not scriptable
else:
    pipeline.vae.decode = lambda latents, return_dict: vae_decoder_traced(latents)
pipeline.unet = unet_traced
pipeline.text_encoder = text_encoder_traced

# torch.jit.script at the top level to capture the loops and controlflows in the scheduler
if script:
    scripted_pipeline = torch.jit.script(pipeline)

    print(scripted_pipeline.code)

    print("decode_latent:")
    print(scripted_pipeline.decode_latents.code)

    print("unet:")
    print(scripted_pipeline.unet.code)

    """
    print("scheduler:")
    print(scripted_pipeline.scheduler.step.code)

    print("scheduler plms:")
    print(scripted_pipeline.scheduler.step_plms.code)
    """

torch.jit.save(scripted_pipeline, f"scripted_sd_{device}.pt")