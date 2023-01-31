from optimum.exporters.onnx.utils import get_stable_diffusion_models_for_export
from optimum.exporters import TasksManager

import onnx
from diffusers import DiffusionPipeline
import torch
import os

from optimum.onnx.utils import check_model_uses_external_data, _get_onnx_external_data_tensors

from scriptable_pipeline_stable_diffusion import ScriptableStableDiffusionPipeline

from utils import export_models, get_traced_submodules, numpy_to_pil, StableDiffusionPreprocessor

from schedulers.scheduling_pndm import ScriptablePNDMScheduler

model_name = "CompVis/stable-diffusion-v1-4"
#model_name = "hf-internal-testing/tiny-stable-diffusion-torch"
pipeline = DiffusionPipeline.from_pretrained(model_name, low_cpu_mem_usage=False)

pipeline = pipeline.to("cpu")

num_inference_steps = 50

scriptable_scheduler = ScriptablePNDMScheduler(**pipeline.scheduler.config)
scriptable_scheduler.set_timesteps(num_inference_steps, device="cpu")

pipeline = ScriptableStableDiffusionPipeline(
    vae=pipeline.vae,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer,
    unet=pipeline.unet,
    scheduler=scriptable_scheduler,
)

text_encoder_config_constructor = TasksManager.get_exporter_config_constructor(
        model=pipeline.text_encoder, exporter="onnx", task="default"
    )
text_encoder_onnx_config = text_encoder_config_constructor(pipeline.text_encoder.config)


models_and_onnx_configs = get_stable_diffusion_models_for_export(pipeline)

in_dummy_out_per_model = export_models(
    models_and_onnx_configs=models_and_onnx_configs,
)

print("Starting tracing...")
text_encoder_traced, unet_traced, vae_decoder_traced = get_traced_submodules(in_dummy_out_per_model, models_and_onnx_configs)
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

# optionally make sure inference runs fine
#if debug:

"""
preprocessed_input = preprocessor.preprocess("A cat sleeping on the beach")

text_input_ids = preprocessed_input["text_input_ids"]
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"]
timesteps = preprocessed_input["timesteps"]

with torch.inference_mode():
    torch_image = pipeline(
        text_input_ids=text_input_ids,
        uncond_text_input_ids=uncond_text_input_ids,
        timesteps=timesteps,
    )[0][0] # first item in "image"

np_image = torch_image.cpu().float().numpy()

image = numpy_to_pil(np_image)
image[0].save("out.png")
"""

# torch.jit.script at the top level to capture the loops and controlflows in the scheduler
if script:
    scripted_pipeline = torch.jit.script(pipeline)

    print(scripted_pipeline.code)

    print("decode_latent:")
    print(scripted_pipeline.decode_latents.code)

    print("unet:")
    print(scripted_pipeline.unet.code)

    print("scheduler:")
    print(scripted_pipeline.scheduler.step.code)

    print("scheduler plms:")
    print(scripted_pipeline.scheduler.step_plms.code)


preprocessed_input = preprocessor.preprocess("A cat sleeping on the beach", num_inference_steps=num_inference_steps)

text_input_ids = preprocessed_input["text_input_ids"]
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"]
timesteps = preprocessed_input["timesteps"]


onnx_file = "stable_diffusion_pipeline.onnx"
torch.onnx.export(
    scripted_pipeline,
    args=(text_input_ids, uncond_text_input_ids, timesteps),
    f=onnx_file,
    input_names=["text_input_ids", "uncond_text_input_ids", "timesteps"],
    #output_names=output_names,
    #dynamic_axes={name: axes for name, axes in chain(inputs.items(), config.outputs.items())},
    #do_constant_folding=True,
    #opset_version=16,
)

# check if external data was exported
onnx_model = onnx.load(onnx_file, load_external_data=False)
model_uses_external_data = check_model_uses_external_data(onnx_model)

if model_uses_external_data:
    print("Saving external data to one file...")
    tensors_paths = _get_onnx_external_data_tensors(onnx_model)

    # try free model memory
    del onnx_model

    onnx_model = onnx.load(onnx_file, load_external_data=True)
    onnx.save(
        onnx_model,
        onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=onnx_file + "_data",
        size_threshold=1024,
    )

    # delete previous external data
    for tensor in tensors_paths:
        os.remove(tensor)


"""
torch.jit.save(scripted_pipeline, "scripted_sd.pt")

preprocessed_input = preprocessor.preprocess("A cat sleeping on the beach", num_inference_steps=num_inference_steps)

text_input_ids = preprocessed_input["text_input_ids"]
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"]
timesteps = preprocessed_input["timesteps"]

print("Running inference...")
print("text_input_ids:", text_input_ids)
with torch.inference_mode():
    if script:
        torch_image = scripted_pipeline(
            text_input_ids=text_input_ids,
            uncond_text_input_ids=uncond_text_input_ids,
            timesteps=timesteps,
        )[0][0] # first item in "image"
    else:
        torch_image = pipeline(
            text_input_ids=text_input_ids,
            uncond_text_input_ids=uncond_text_input_ids,
            timesteps=timesteps,
        )[0][0] # first item in "image"

np_image = torch_image.cpu().float().numpy()

image = numpy_to_pil(np_image)
image[0].save("scripted_out.png")
"""