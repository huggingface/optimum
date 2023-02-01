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
"""Defines the command line for the export with TensorFlow Lite."""

import subprocess
from pathlib import Path

from ...exporters import TasksManager


def parse_args_tflite(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "output", type=Path, help="Path indicating the directory where to store generated TFLite model."
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
        "--atol",
        type=float,
        default=None,
        help="If specified, the absolute difference tolerance when validating the model. Otherwise, the default atol for the model will be used.",
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
        help="Allow to use custom code for the modeling hosted in the model repository. This option should only be set for repositories you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the model repository.",
    )

    input_group = parser.add_argument_group("Input shapes")
    doc_input = "that the TFLite exported model will be able to take as input."
    input_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help=f"Batch size {doc_input}",
    )
    input_group.add_argument(
        "--sequence_length",
        type=int,
        default=None,
        help=f"Sequence length {doc_input}",
    )
    input_group.add_argument(
        "--num_choices",
        type=int,
        default=None,
        help=f"Only for the multiple-choice task. Num choices {doc_input}",
    )
    input_group.add_argument(
        "--width",
        type=int,
        default=None,
        help=f"Vision tasks only. Image width {doc_input}",
    )
    input_group.add_argument(
        "--height",
        type=int,
        default=None,
        help=f"Vision tasks only. Image height {doc_input}",
    )
    input_group.add_argument(
        "--num_channels",
        type=int,
        default=None,
        help=f"Vision tasks only. Number of channels used to represent the image {doc_input} (GREY = 1, RGB = 3, ARGB = 4)",
    )
    input_group.add_argument(
        "--feature_size",
        type=int,
        default=None,
        help=f"Audio tasks only. Feature dimension of the extracted features by the feature extractor {doc_input}",
    )
    input_group.add_argument(
        "--nb_max_frames",
        type=int,
        default=None,
        help=f"Audio tasks only. Maximum number of frames {doc_input}",
    )
    input_group.add_argument(
        "--audio_sequence_length",
        type=int,
        default=None,
        help=f"Audio tasks only. Audio sequence length {doc_input}",
    )


class TFLiteExportCommand:
    def __init__(self, args_string):
        self.args_string = args_string

    def run(self):
        full_command = f"python3 -m optimum.exporters.tflite {self.args_string}"
        subprocess.run(full_command, shell=True, check=True)
