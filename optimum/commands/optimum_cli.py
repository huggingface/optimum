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

import importlib
from pathlib import Path
from typing import List, Optional, Tuple, Type, Union

from ..utils import logging
from .base import BaseOptimumCLICommand, CommandInfo, RootOptimumCLICommand
from .env import EnvironmentCommand
from .export import ExportCommand
from .onnxruntime import ONNXRuntimeCommand


logger = logging.get_logger()

OPTIMUM_CLI_SUBCOMMANDS = [
    ExportCommand,
    EnvironmentCommand,
    ONNXRuntimeCommand,
]
ROOT = RootOptimumCLICommand("Optimum CLI tool", usage="optimum-cli <command> [<args>]")


def dynamic_load_commands_in_register() -> (
    List[Tuple[Union[Type[BaseOptimumCLICommand], CommandInfo], Optional[Type[BaseOptimumCLICommand]]]]
):
    commands_to_register = []
    register_dir_path = Path(__file__).parent / "register"
    for filename in register_dir_path.iterdir():
        if filename.is_dir() or filename.suffix != ".py":
            if filename.name != "__pycache__":
                logger.warning(
                    f"Skipping {filename} because only python files are allowed when registering commands dynamically."
                )
            continue
        module_name = f".register.{filename.stem}"
        module = importlib.import_module(module_name, package="optimum.commands")
        commands_to_register_in_file = getattr(module, "REGISTER_COMMANDS", [])
        for command_idx, command in enumerate(commands_to_register_in_file):
            if isinstance(command, tuple):
                command_or_command_info, parent_command_cls = command
            else:
                command_or_command_info = command
                parent_command_cls = None
            if not isinstance(command_or_command_info, CommandInfo) and not issubclass(
                command_or_command_info, BaseOptimumCLICommand
            ):
                raise ValueError(
                    f"The command at index {command_idx} in {filename} is not of the right type: {type(command_or_command_info)}."
                )
            commands_to_register.append((command_or_command_info, parent_command_cls))
    return commands_to_register


def register_optimum_cli_subcommand(
    command_or_command_info: Union[Type[BaseOptimumCLICommand], CommandInfo],
    parent_command_cls: Optional[Type[BaseOptimumCLICommand]] = None,
):
    if parent_command_cls is None:
        parent_command_cls = RootOptimumCLICommand
    if not isinstance(command_or_command_info, CommandInfo):
        command_info = CommandInfo(
            command_or_command_info.COMMAND.name,
            help=command_or_command_info.COMMAND.help,
            subcommand_class=command_or_command_info,
        )
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
    # Register commands
    for subcommand_cls in OPTIMUM_CLI_SUBCOMMANDS:
        register_optimum_cli_subcommand(subcommand_cls)

    commands_in_register = dynamic_load_commands_in_register()
    for command_or_command_info, parent_command_cls in commands_in_register:
        register_optimum_cli_subcommand(command_or_command_info, parent_command_cls=parent_command_cls)

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
