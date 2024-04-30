# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Defines the command line for the export with ONNX."""

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

from ...exporters import TasksManager
from ...utils import DEFAULT_DUMMY_SHAPES
from ..base import BaseOptimumCLICommand


if TYPE_CHECKING:
    from argparse import ArgumentParser


def parse_args_onnx(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store the generated ONNX model."
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
            f" {str(TasksManager.get_all_tasks())}. For decoder models, use `xxx-with-past` to export the model using past key values in the decoder."
        ),
    )
    optional_group.add_argument(
        "--opset",
        type=int,
        default=None,
        help="If specified, ONNX opset version to export the model with. Otherwise, the default opset for the given model architecture will be used.",
    )
    optional_group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='The device to use to do the export. Defaults to "cpu".',
    )
    optional_group.add_argument(
        "--fp16",
        action="store_true",
        help="Use half precision during the export. PyTorch-only, requires `--device cuda`.",
    )
    optional_group.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="The floating point precision to use for the export. Supported options: fp32 (float32), fp16 (float16), bf16 (bfloat16).",
    )
    optional_group.add_argument(
        "--optimize",
        type=str,
        default=None,
        choices=["O1", "O2", "O3", "O4"],
        help=(
            "Allows to run ONNX Runtime optimizations directly during the export. Some of these optimizations are specific to ONNX Runtime, and the resulting ONNX will not be usable with other runtime as OpenVINO or TensorRT. Possible options:\n"
            "    - O1: Basic general optimizations\n"
            "    - O2: Basic and extended general optimizations, transformers-specific fusions\n"
            "    - O3: Same as O2 with GELU approximation\n"
            "    - O4: Same as O3 with mixed precision (fp16, GPU-only, requires `--device cuda`)"
        ),
    )
    optional_group.add_argument(
        "--monolith",
        action="store_true",
        help=(
            "Forces to export the model as a single ONNX file. By default, the ONNX exporter may break the model in several"
            " ONNX files, for example for encoder-decoder models where the encoder should be run only once while the"
            " decoder is looped over."
        ),
    )
    optional_group.add_argument(
        "--no-post-process",
        action="store_true",
        help=(
            "Allows to disable any post-processing done by default on the exported ONNX models. For example, the merging of decoder"
            " and decoder-with-past models into a single ONNX model file to reduce memory usage."
        ),
    )
    optional_group.add_argument(
        "--variant",
        type=str,
        default="default",
        help=("Select a variant of the model to export."),
    )
    optional_group.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        default=None,
        help=(
            "The framework to use for the ONNX export."
            " If not provided, will attempt to use the local checkpoint's original framework"
            " or what is available in the environment."
        ),
    )
    optional_group.add_argument(
        "--atol",
        type=float,
        default=None,
        help="If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.",
    )
    optional_group.add_argument(
        "--cache_dir", type=str, default=HUGGINGFACE_HUB_CACHE, help="Path indicating where to store cache."
    )
    optional_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the model repository.",
    )
    optional_group.add_argument(
        "--pad_token_id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess"
            " it."
        ),
    )
    optional_group.add_argument(
        "--library-name",
        type=str,
        choices=["transformers", "diffusers", "timm", "sentence_transformers"],
        default=None,
        help=("The library on the model." " If not provided, will attempt to infer the local checkpoint's library"),
    )
    optional_group.add_argument(
        "--model-kwargs",
        type=json.loads,
        help=("Any kwargs passed to the model forward, or used to customize the export for a given model."),
    )
    optional_group.add_argument(
        "--legacy",
        action="store_true",
        help=(
            "Export decoder only models in three files (without + with past and the resulting merged model)."
            "Also disable the use of position_ids for text-generation models that require it for batched generation. This argument is introduced for backward compatibility and will be removed in a future release of Optimum."
        ),
    )
    optional_group.add_argument(
        "--no-dynamic-axes", action="store_true", help="Disable dynamic axes during ONNX export"
    )
    optional_group.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="PyTorch-only argument. Disables PyTorch ONNX export constant folding.",
    )

    input_group = parser.add_argument_group(
        "Input shapes (if necessary, this allows to override the shapes of the input given to the ONNX exporter, that requires an example input)."
    )
    doc_input = "to use in the example input given to the ONNX export."
    input_group.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["batch_size"],
        help=f"Text tasks only. Batch size {doc_input}",
    )
    input_group.add_argument(
        "--sequence_length",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["sequence_length"],
        help=f"Text tasks only. Sequence length {doc_input}",
    )
    input_group.add_argument(
        "--num_choices",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["num_choices"],
        help=f"Text tasks only. Num choices {doc_input}",
    )
    input_group.add_argument(
        "--width",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["width"],
        help=f"Image tasks only. Width {doc_input}",
    )
    input_group.add_argument(
        "--height",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["height"],
        help=f"Image tasks only. Height {doc_input}",
    )
    input_group.add_argument(
        "--num_channels",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["num_channels"],
        help=f"Image tasks only. Number of channels {doc_input}",
    )
    input_group.add_argument(
        "--feature_size",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["feature_size"],
        help=f"Audio tasks only. Feature size {doc_input}",
    )
    input_group.add_argument(
        "--nb_max_frames",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["nb_max_frames"],
        help=f"Audio tasks only. Maximum number of frames {doc_input}",
    )
    input_group.add_argument(
        "--audio_sequence_length",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["audio_sequence_length"],
        help=f"Audio tasks only. Audio sequence length {doc_input}",
    )
    input_group.add_argument(
        "--point_batch_size",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["point_batch_size"],
        help=(
            "For Segment Anything. It corresponds to how many segmentation masks we want the model to predict per "
            "input point."
        ),
    )
    input_group.add_argument(
        "--nb_points_per_image",
        type=int,
        default=DEFAULT_DUMMY_SHAPES["nb_points_per_image"],
        help="For Segment Anything. It corresponds to the number of points per segmentation masks.",
    )

    # deprecated argument
    parser.add_argument("--for-ort", action="store_true", help=argparse.SUPPRESS)


class ONNXExportCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_onnx(parser)

    def run(self):
        from ...exporters.onnx import main_export

        # Get the shapes to be used to generate dummy inputs
        input_shapes = {}
        for input_name in DEFAULT_DUMMY_SHAPES.keys():
            if hasattr(self.args, input_name):
                input_shapes[input_name] = getattr(self.args, input_name)

        main_export(
            model_name_or_path=self.args.model,
            output=self.args.output,
            task=self.args.task,
            opset=self.args.opset,
            device=self.args.device,
            fp16=self.args.fp16,
            dtype=self.args.dtype,
            optimize=self.args.optimize,
            monolith=self.args.monolith,
            no_post_process=self.args.no_post_process,
            framework=self.args.framework,
            atol=self.args.atol,
            cache_dir=self.args.cache_dir,
            trust_remote_code=self.args.trust_remote_code,
            pad_token_id=self.args.pad_token_id,
            for_ort=self.args.for_ort,
            use_subprocess=True,
            _variant=self.args.variant,
            library_name=self.args.library_name,
            legacy=self.args.legacy,
            no_dynamic_axes=self.args.no_dynamic_axes,
            model_kwargs=self.args.model_kwargs,
            do_constant_folding=not self.args.no_constant_folding,
            **input_shapes,
        )
