import onnxruntime as ort
import numpy as np
import time
from utils import numpy_to_pil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpu",
    action="store_true",
    help="use to do inference on GPU.",
)
args = parser.parse_args()

if args.gpu:
    providers = ["CUDAExecutionProvider"]
    #providers = ["TensorrtExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]

session_options = ort.SessionOptions()
#session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
#session_options.log_severity_level = 0

onnx_file = "stable_diffusion_pipeline.onnx"
session = ort.InferenceSession(onnx_file, providers=providers, sess_options=session_options)

# 1.4
"""
text_input_ids = np.array([[49406,   320,  2368,  6982,   525,   518,  2117, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407]])
uncond_text_input_ids = np.array([[49406, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407]])
timesteps = np.array([981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
        441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
        161, 141, 121, 101,  81,  61,  41,  21,   1])
"""

# tiny
text_input_ids = np.array([[  0, 197,  68, 434,  84, 536, 545, 383, 390, 449, 195,  67,  70, 885,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1]])
uncond_text_input_ids = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1]])
timesteps = np.array([981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
        441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
        161, 141, 121, 101,  81,  61,  41,  21,   1])

# WARNING: some height / width shapes do not work, not sure why. They don't work either for the ScriptModule inference.
inp = {
    "text_input_ids": text_input_ids.astype(np.int64),
    "uncond_text_input_ids": uncond_text_input_ids.astype(np.int64),
    "timesteps": timesteps.astype(np.int64),
    "num_images_per_prompt": np.array(1).astype(np.int64),
    "height": np.array(128).astype(np.int64),
    "width": np.array(60).astype(np.int64),
    "guidance_scale": np.array(7.5).astype(np.float64),
}

# warmup
print("FORWARD")
np_images = session.run(None, inp)[0]

"""
for i in range(5):
    print("FORWARD")
    start = time.time()
    np_images = session.run(None, inp)[0]
    print(f"Took {time.time() - start} s")
"""

images = numpy_to_pil(np_images)
for i, im in enumerate(images):
    im.save(f"ort_out{i}.png")
