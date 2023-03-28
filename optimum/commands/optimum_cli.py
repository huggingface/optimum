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
from typing import Type, Optional, Union

from optimum.commands.base import BaseOptimumCLICommand, CommandInfo, RootOptimumCLICommand

from .env import EnvironmentCommand
from .export import ExportCommand
from .onnxruntime import ONNXRuntimeCommand


OPTIMUM_CLI_SUBCOMMANDS = [
    ExportCommand,
    EnvironmentCommand,
    ONNXRuntimeCommand,
]

ROOT = RootOptimumCLICommand("Optimum CLI tool", usage="optimum-cli <command> [<args>]")



def register_optimum_cli_subcommand(command_or_command_info: Union[Type[BaseOptimumCLICommand], CommandInfo], parent_command_cls: Optional[Type[BaseOptimumCLICommand]] = None):
    if parent_command_cls is None:
        parent_command_cls = RootOptimumCLICommand
    if not isinstance(command_or_command_info, CommandInfo):
        command_info = CommandInfo(command_or_command_info.COMMAND.name, help=command_or_command_info.COMMAND.help, subcommand_class=command_or_command_info)
    else:
        command_info = command_or_command_info
    command_info.is_subcommand_info_or_raise()

    parent_command = ROOT

    was_registered = False
    to_visit = [parent_command]
    while to_visit:
        subcommand = to_visit.pop(0)
        if isinstance(subcommand, parent_command_cls):
            subcommand.register_subcommand(command_info)
            was_registered = True
            break
        to_visit += parent_command.registered_subcommands
    if not was_registered:
        raise RuntimeError(
            f"Could not register the subcommand called {command_info.name} to the parent command {parent_command} "
            f"because this parent command is not in the Optimum CLI subcommands."
        )


def main():
    # parser = ArgumentParser("Optimum CLI tool", usage="optimum-cli <command> [<args>]")
    # commands_parser = parser.add_subparsers()

    # Register commands
    for subcommand_cls in OPTIMUM_CLI_SUBCOMMANDS:
        # subcommand_cls(commands_parser)
        register_optimum_cli_subcommand(subcommand_cls)

    parser = ROOT.parser
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
