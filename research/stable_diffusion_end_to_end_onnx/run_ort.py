import onnxruntime as ort
import numpy as np
import torch

from utils import numpy_to_pil

session = ort.InferenceSession("stable_diffusion_pipeline.onnx", providers=["CUDAExecutionProvider"])


text_input_ids = torch.tensor([[  0, 197,  68, 434,  84, 536, 545, 383, 390, 449, 195,  67,  70, 885,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,
           1,   1,   1,   1,   1,   1,   1]])
uncond_text_input_ids = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1]])
timesteps = torch.tensor([981, 961, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741,
        721, 701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461,
        441, 421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181,
        161, 141, 121, 101,  81,  61,  41,  21,   1])

# prompt = ""

inp = {
    "text_input_ids": text_input_ids.numpy().astype(np.int64),
    "uncond_text_input_ids": uncond_text_input_ids.numpy().astype(np.int64),
    "timesteps": timesteps.numpy().astype(np.int64),
}

np_image = session.run(None, inp)[0]

image = numpy_to_pil(np_image)
image[0].save("ort_out.png")
