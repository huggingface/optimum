from diffusers import DiffusionPipeline
from scriptable_pipeline_stable_diffusion import ScriptableStableDiffusionPipeline
import torch

import time
from utils import StableDiffusionPreprocessor, numpy_to_pil
from schedulers.scheduling_pndm import ScriptablePNDMScheduler

device = "cuda"
dtype = torch.float32

model_name = "CompVis/stable-diffusion-v1-4"
#model_name = "hf-internal-testing/tiny-stable-diffusion-torch"
pipeline = DiffusionPipeline.from_pretrained(model_name, low_cpu_mem_usage=False, torch_dtype=dtype)

num_inference_steps = 50
scriptable_scheduler = ScriptablePNDMScheduler(**pipeline.scheduler.config)
scriptable_scheduler.set_timesteps(num_inference_steps, device=device)

pipeline = pipeline.to(device)
pipeline = ScriptableStableDiffusionPipeline(
    vae=pipeline.vae,
    text_encoder=pipeline.text_encoder,
    tokenizer=pipeline.tokenizer,
    unet=pipeline.unet,
    scheduler=scriptable_scheduler,
)

preprocessor = StableDiffusionPreprocessor(
    pipeline.tokenizer,
    pipeline.text_encoder.config,
    do_classifier_free_guidance=True,
    scheduler=scriptable_scheduler,
)

# uncond_text_input_ids, uncond_attention_mask
preprocessed_input = preprocessor.preprocess("A cat sleeping on the beach", num_inference_steps=num_inference_steps)

text_input_ids = preprocessed_input["text_input_ids"].to(device)
uncond_text_input_ids = preprocessed_input["uncond_text_input_ids"].to(device)
timesteps = preprocessed_input["timesteps"].to(device)

# breakpoint()
# np_image = pipeline(text_input_ids=text_input_ids, attention_mask=attention_mask).images[0]
print("len timesteps", timesteps.shape)

with torch.inference_mode():
    # warmup
    print("FORWARD")
    torch_image = pipeline(
        text_input_ids=text_input_ids,
        uncond_text_input_ids=uncond_text_input_ids,
        timesteps=timesteps,
    )[0][0]  # first item in "image" output, indexed at 0

    for i in range(5):
        print("FORWARD")
        start = time.time()
        torch_image = pipeline(
            text_input_ids=text_input_ids,
            uncond_text_input_ids=uncond_text_input_ids,
            timesteps=timesteps,
        )[0][0] # first item in "image"
        print(f"Took {time.time() - start} s")

np_image = torch_image.cpu().float().numpy()

image = numpy_to_pil(np_image)
image[0].save("out.png")
