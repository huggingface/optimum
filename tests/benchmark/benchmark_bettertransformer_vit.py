import argparse

import requests
import torch
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModel

from optimum.bettertransformer import BetterTransformer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/vit-base-patch16-224",
        help="",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
    )
    parser.add_argument(
        "--use-half",
        action="store_true",
    )
    return parser


def get_batch(batch_size, model_name):
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    input_features = feature_extractor([image for _ in range(batch_size)], return_tensors="pt")
    return input_features


def timing_cuda(model, num_batches, input_features):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(num_batches):
        _ = model(input_features)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / num_batches


def benchmark(model_name, num_batches, batch_size, is_cuda, is_half):
    print("Loading model {}".format(model_name))
    if is_cuda:
        hf_model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float16 if is_half else None, device_map="auto"
        ).eval()
    else:
        hf_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if is_half else None).eval()

    bt_model = BetterTransformer.transform(hf_model, keep_original_model=True)

    inputs = get_batch(batch_size, model_name)
    input_features = inputs["pixel_values"]

    if is_cuda:
        input_features = input_features.to(0)

    # Warmup
    _ = hf_model(input_features)
    torch.cuda.synchronize()
    _ = bt_model(input_features)
    torch.cuda.synchronize()

    total_hf_time = timing_cuda(hf_model, num_batches, input_features)
    total_bt_time = timing_cuda(bt_model, num_batches, input_features)

    return total_bt_time, total_hf_time


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    BATCH_SIZES = [2, 4, 8, 16, 32]

    output_file = open("log_{}.csv".format(args.model_name.replace("/", "-")), "w")
    output_file.write("num_batches,batch_size,is_cuda,is_half,HF_time,BT_time,Speedup\n")
    for bs in BATCH_SIZES:
        total_bt_time, total_hf_time = benchmark(
            args.model_name,
            args.num_batches,
            bs,
            args.use_cuda,
            args.use_half,
        )

        speedup = total_hf_time / total_bt_time

        output_file.write(
            "{},{},{},{},{},{},{}\n".format(
                args.num_batches,
                bs,
                args.use_cuda,
                args.use_half,
                total_hf_time,
                total_bt_time,
                speedup,
            )
        )
    output_file.close()
