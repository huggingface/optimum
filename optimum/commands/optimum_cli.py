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
from typing import Dict, List, Optional, Tuple, Type, Union

from ..subpackages import load_subpackages
from ..utils import logging
from .base import BaseOptimumCLICommand, CommandInfo, RootOptimumCLICommand
from .env import EnvironmentCommand
from .export import ExportCommand


logger = logging.get_logger()

# The table below contains the optimum-cli root subcommands provided by the optimum package
OPTIMUM_CLI_ROOT_SUBCOMMANDS = [ExportCommand, EnvironmentCommand]

# The table below is dynamically populated when loading subpackages
_OPTIMUM_CLI_SUBCOMMANDS = []


def optimum_cli_subcommand(parent_command: Optional[Type[BaseOptimumCLICommand]] = None):
    """
    A decorator to declare optimum-cli subcommands.

    The declaration of an optimum-cli subcommand looks like this:

    ```
    @optimum_cli_subcommand()
    class MySubcommand(BaseOptimumCLICommand):
        <implementation>
    ```

    or

    ```
    @optimum_cli_subcommand(ExportCommand)
    class MySubcommand(BaseOptimumCLICommand):
        <implementation>
    ```

    Args:
        parent_command: (`Optional[Type[BaseOptimumCLICommand]]`):
            The class of the parent command or None if this is a top-level command. Defaults to None.

    """

    if parent_command is not None and not issubclass(parent_command, BaseOptimumCLICommand):
        raise ValueError(f"The parent command {parent_command} must be a subclass of BaseOptimumCLICommand")

    def wrapper(subcommand):
        if not issubclass(subcommand, BaseOptimumCLICommand):
            raise ValueError(f"The subcommand {subcommand} must be a subclass of BaseOptimumCLICommand")
        _OPTIMUM_CLI_SUBCOMMANDS.append((subcommand, parent_command))

    return wrapper


def resolve_command_to_command_instance(
    root: RootOptimumCLICommand, commands: List[Type[BaseOptimumCLICommand]]
) -> Dict[Type[BaseOptimumCLICommand], BaseOptimumCLICommand]:
    """
    Retrieves command instances in the root CLI command from a list of command classes.

    Args:
        root (`RootOptimumCLICommand`):
            The root CLI command instance.
        commands (`List[Type[BaseOptimumCLICommand]]`):
            The list of command classes to retrieve the instances of in root.

    Returns:
        `Dict[Type[BaseOptimumCLICommand], BaseOptimumCLICommand]`: A dictionary mapping a command class to a command
        instance in the root CLI.
    """
    to_visit = [root]
    remaining_commands = set(commands)
    command2command_instance = {}
    while to_visit:
        current_command_instance = to_visit.pop(0)
        if current_command_instance.__class__ in remaining_commands:
            remaining_commands.remove(current_command_instance.__class__)
            command2command_instance[current_command_instance.__class__] = current_command_instance
        if not remaining_commands:
            break
        to_visit += current_command_instance.registered_subcommands
    if remaining_commands:
        class_names = (command.__name__ for command in remaining_commands)
        raise RuntimeError(
            f"Could not find an instance of the following commands in the CLI {root}: {', '.join(class_names)}."
        )
    return command2command_instance


def dynamic_load_commands_in_register() -> (
    List[Tuple[Union[Type[BaseOptimumCLICommand], CommandInfo], Optional[Type[BaseOptimumCLICommand]]]]
):
    """
    Loads a list of command classes to register to the CLI from the `optimum/commands/register/` directory.
    It will look in any python file if there is a `REGISTER_COMMANDS` list, and load the commands to register
    accordingly.
    At this point, nothing is actually registered in the root CLI.

    Returns:
        `List[Tuple[Union[Type[BaseOptimumCLICommand], CommandInfo], Optional[Type[BaseOptimumCLICommand]]]]`: A list
        of tuples with two elements. The first element corresponds to the command to register, it can either be a
        subclass of `BaseOptimumCLICommand` or a `CommandInfo`. The second element corresponds to the parent command,
        where `None` means that the parent command is the root CLI command.
    """
    commands_to_register = []
    register_dir_path = Path(__file__).parent / "register"
    for filename in register_dir_path.iterdir():
        if filename.is_dir() or filename.suffix != ".py":
            if filename.name not in ["__pycache__", "README.md"]:
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
    parent_command: BaseOptimumCLICommand,
):
    """
    Registers a command as being a subcommand of `parent_command`.

    Args:
        command_or_command_info (`Union[Type[BaseOptimumCLICommand], CommandInfo]`):
            The command to register.
        parent_command (`BaseOptimumCLICommand`):
            The instance of the parent command.
    """
    if not isinstance(command_or_command_info, CommandInfo):
        command_info = CommandInfo(
            command_or_command_info.COMMAND.name,
            help=command_or_command_info.COMMAND.help,
            subcommand_class=command_or_command_info,
        )
    else:
        command_info = command_or_command_info
    command_info.is_subcommand_info_or_raise()
    parent_command.register_subcommand(command_info)


def main():
    root = RootOptimumCLICommand("Optimum CLI tool", usage="optimum-cli")
    parser = root.parser

    for subcommand_cls in OPTIMUM_CLI_ROOT_SUBCOMMANDS:
        register_optimum_cli_subcommand(subcommand_cls, parent_command=root)

    # Load subpackages to give them a chance to declare their own subcommands
    load_subpackages()

    # Register subcommands declared by the subpackages or found in the register files under commands/register
    commands_to_register = _OPTIMUM_CLI_SUBCOMMANDS + dynamic_load_commands_in_register()
    command2command_instance = resolve_command_to_command_instance(
        root, [parent_command_cls for _, parent_command_cls in commands_to_register if parent_command_cls is not None]
    )

    for command_or_command_info, parent_command in commands_to_register:
        if parent_command is None:
            parent_command_instance = root
        else:
            parent_command_instance = command2command_instance[parent_command]
        register_optimum_cli_subcommand(command_or_command_info, parent_command=parent_command_instance)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
