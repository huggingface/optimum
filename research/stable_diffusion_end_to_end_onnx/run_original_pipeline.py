from diffusers import DiffusionPipeline
import argparse
import torch
import time
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

# NOTE: Beware that model_path should match with the .pt model!
model_path = MODEL_NAME

pipeline = DiffusionPipeline.from_pretrained(model_path)

num_inference_steps = 50
num_images_per_prompt = 2
width = 512
height = 512
guidance_scale = 7.5

pipeline = pipeline.to(device)

with torch.inference_mode():
    print("FORWARD")
    res = pipeline(
        "A cat sleeping on the beach",
        num_inference_steps=num_inference_steps,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    )

    for i in range(5):
        print("FORWARD")
        start = time.time()
        torch_images = pipeline(
            "A cat sleeping on the beach",
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
        )[0]
        print(f"Took {time.time() - start} s")
