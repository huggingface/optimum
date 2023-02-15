import torch
import onnx
import os
from utils import StableDiffusionPreprocessor
import argparse

from transformers import AutoTokenizer, CLIPTextConfig

from diffusers.schedulers import PNDMScheduler
from optimum.onnx.utils import check_model_uses_external_data, _get_onnx_external_data_tensors

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu",
    action="store_true",
    help="use to trace and script on GPU.",
)
args = parser.parse_args()

if args.gpu:
    device = "cuda"
else:
    device = "cpu"

scripted_pipeline = torch.load(f"scripted_sd_{device}.pt")

model_path = "CompVis/stable-diffusion-v1-4"
#model_path = "hf-internal-testing/tiny-stable-diffusion-torch"
text_encoder_config = CLIPTextConfig.from_pretrained(model_path, subfolder="text_encoder")
tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
scheduler = PNDMScheduler.from_pretrained(model_path, subfolder="scheduler")

preprocessor = StableDiffusionPreprocessor(
    tokenizer,
    text_encoder_config,
    do_classifier_free_guidance=True,
    scheduler=scheduler,
)

num_inference_steps = 50
preprocessed_input = preprocessor.preprocess("A cat sleeping on the beach", num_inference_steps=num_inference_steps)

text_input_ids = preprocessed_input["text_input_ids"]
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"]
timesteps = preprocessed_input["timesteps"]


text_input_ids = preprocessed_input["text_input_ids"].to(device)
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"].to(device)
timesteps = preprocessed_input["timesteps"].to(device)

if device == "cuda":
    scripted_pipeline = scripted_pipeline.to("cuda")

print(text_input_ids)
print(uncond_text_input_ids)
print(timesteps)

onnx_file = "stable_diffusion_pipeline.onnx"
with torch.inference_mode():
    torch.onnx.export(
        scripted_pipeline,
        args=(text_input_ids, uncond_text_input_ids, timesteps),
        f=onnx_file,
        input_names=["text_input_ids", "uncond_text_input_ids", "timesteps"],
        # output_names=output_names,
        # dynamic_axes={name: axes for name, axes in chain(inputs.items(), config.outputs.items())},
        # do_constant_folding=True,
        # opset_version=16,
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
