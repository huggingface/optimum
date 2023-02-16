# NOTE: create_scriptmodule.py needs to be run without the flag --gpu for this to work
import argparse

import torch

from utils import numpy_to_pil, StableDiffusionPreprocessor

from transformers import AutoTokenizer, CLIPTextConfig

from diffusers.schedulers import PNDMScheduler
import time
import argparse
from constants import MODEL_NAME

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

scripted_pipeline = torch.jit.load(f"scripted_sd_{device}.pt")

# NOTE: Beware that model_path should match with the .pt model!
model_path = MODEL_NAME
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

text_input_ids = preprocessed_input["text_input_ids"].to(device)
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"].to(device)
timesteps = preprocessed_input["timesteps"].to(device)

if device == "cuda":
    scripted_pipeline = scripted_pipeline.to("cuda")

print(text_input_ids)
print(uncond_text_input_ids)
print(timesteps)

# 0. Defaults
height = scripted_pipeline.sample_size * scripted_pipeline.vae_scale_factor
width = scripted_pipeline.sample_size * scripted_pipeline.vae_scale_factor
num_images_per_prompt = 1
guidance_scale = 7.5

print("Running inference...")
with torch.inference_mode():
    # warmup
    print("FORWARD")
    torch_images = scripted_pipeline(
        text_input_ids=text_input_ids,
        uncond_text_input_ids=uncond_text_input_ids,
        timesteps=timesteps,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
    )[0]

    for i in range(5):
        print("FORWARD")
        start = time.time()
        torch_images = scripted_pipeline(
            text_input_ids=text_input_ids,
            uncond_text_input_ids=uncond_text_input_ids,
            timesteps=timesteps,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        )[0]
        print(f"Took {time.time() - start} s")

np_images = torch_images[0].cpu().float().numpy()

images = numpy_to_pil(np_images)
for i, im in enumerate(images):
    im.save(f"scriptmodule_out{i}.png")
