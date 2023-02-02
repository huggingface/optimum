import onnxruntime as ort
import numpy as np

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
else:
    providers = ["CPUExecutionProvider"]

session_options = ort.SessionOptions()
session_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
)

session = ort.InferenceSession("stable_diffusion_pipeline.onnx", providers=providers, sess_options=session_options)

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
inp = {
    "text_input_ids": text_input_ids.astype(np.int64),
    "uncond_text_input_ids": uncond_text_input_ids.astype(np.int64),
    "timesteps": timesteps.astype(np.int64),
}

np_image = session.run(None, inp)[0]

image = numpy_to_pil(np_image)
image[0].save("ort_out.png")
