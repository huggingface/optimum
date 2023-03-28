#!/usr/bin/env python
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

from argparse import ArgumentParser
from typing import Type, TypeVar

from optimum.commands.base import BaseOptimumCLICommand

from .env import EnvironmentCommand
from .export import ExportCommand
from .onnxruntime import ONNXRuntimeCommand

OPTIMUM_CLI_SUBCOMMANDS = [
    ExportCommand,
    EnvironmentCommand,
    ONNXRuntimeCommand,
    
]


def register_optimum_cli_subcommand(subcommand_cls: Type[BaseOptimumCLICommand]):
    OPTIMUM_CLI_SUBCOMMANDS.append(subcommand_cls)


def main():
    parser = ArgumentParser("Optimum CLI tool", usage="optimum-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="optimum-cli command helpers")

    # Register commands
    for subcommand_cls in OPTIMUM_CLI_SUBCOMMANDS:
        subcommand_cls(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
