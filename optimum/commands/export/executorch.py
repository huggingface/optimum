# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Defines the command line for the export with ExecuTorch."""

from pathlib import Path
from typing import TYPE_CHECKING

from ...exporters import TasksManager
from ..base import BaseOptimumCLICommand


if TYPE_CHECKING:
    from argparse import ArgumentParser


def parse_args_executorch(parser):
    required_group = parser.add_argument_group("Required arguments")
    required_group.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    required_group.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        help="Path indicating the directory where to store the generated ExecuTorch model.",
    )
    required_group.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help=(
            "The task to export the model for. Available tasks depend on the model, but are among:"
            f" {str(TasksManager.get_all_tasks())}."
        ),
    )
    required_group.add_argument(
        "--recipe",
        type=str,
        default="xnnpack",
        help='Pre-defined recipes for export to ExecuTorch. Defaults to "xnnpack".',
    )


class ExecuTorchExportCommand(BaseOptimumCLICommand):
    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        return parse_args_executorch(parser)

    def run(self):
        from ...exporters.executorch import main_export

        main_export(
            model_name_or_path=self.args.model,
            task=self.args.task,
            recipe=self.args.recipe,
            output_dir=self.args.output_dir,
        )
