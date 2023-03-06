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
import subprocess
from pathlib import Path

from ...exporters import TasksManager
from ...utils import DEFAULT_DUMMY_SHAPES


def parse_args_onnx(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store generated ONNX model."
    )

    optional_group = parser.add_argument_group("Optional arguments")
    optional_group.add_argument(
        "--task",
        default="auto",
        help=(
            "The task to export the model for. If not specified, the task will be auto-inferred based on the model. Available tasks depend on the model, but are among:"
            f" {str(list(TasksManager._TASKS_TO_AUTOMODELS.keys()))}. For decoder models, use `xxx-with-past` to export the model using past key values in the decoder."
        ),
    )
    optional_group.add_argument(
        "--monolith",
        action="store_true",
        help=(
            "Force to export the model as a single ONNX file. By default, the ONNX exporter may break the model in several"
            " ONNX files, for example for encoder-decoder models where the encoder should be run only once while the"
            " decoder is looped over."
        ),
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
        help="Experimental option: use half precision during the export. PyTorch-only, requires `--device cuda`.",
    )
    optional_group.add_argument(
        "--opset",
        type=int,
        default=None,
        help="If specified, ONNX opset version to export the model with. Otherwise, the default opset will be used.",
    )
    optional_group.add_argument(
        "--atol",
        type=float,
        default=None,
        help="If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.",
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
        "--pad_token_id",
        type=int,
        default=None,
        help=(
            "This is needed by some models, for some tasks. If not provided, will attempt to use the tokenizer to guess"
            " it."
        ),
    )
    optional_group.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    optional_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the model repository.",
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

    input_group = parser.add_argument_group(
        "Input shapes (if necessary, this allows to override the shapes of the input given to the ONNX exporter, that requires an example input.)"
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

    # deprecated argument
    parser.add_argument("--for-ort", action="store_true", help=argparse.SUPPRESS)


class ONNXExportCommand:
    def __init__(self, args_string):
        self.args_string = args_string

    def run(self):
        full_command = f"python3 -m optimum.exporters.onnx {self.args_string}"
        subprocess.run(full_command, shell=True, check=True)
