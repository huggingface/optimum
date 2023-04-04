# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Optimum command-line interface base classes."""

from abc import ABC
from argparse import ArgumentParser, RawTextHelpFormatter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Type


if TYPE_CHECKING:
    from argparse import Namespace, _SubParsersAction


@dataclass(frozen=True)
class CommandInfo:
    name: str
    help: str
    subcommand_class: Optional[Type["BaseOptimumCLICommand"]] = None
    formatter_class: Type = RawTextHelpFormatter

    @property
    def is_subcommand_info(self):
        return self.subcommand_class is not None

    def is_subcommand_info_or_raise(self):
        if not self.is_subcommand_info:
            raise ValueError(f"The command info must define a subcommand_class attribute, but got: {self}.")


class BaseOptimumCLICommand(ABC):
    COMMAND: CommandInfo
    SUBCOMMANDS: Tuple[CommandInfo, ...] = ()

    def __init__(
        self,
        subparsers: Optional["_SubParsersAction"],
        args: Optional["Namespace"] = None,
        command: Optional[CommandInfo] = None,
        from_defaults_factory: bool = False,
        parser: Optional["ArgumentParser"] = None,
    ):
        """
        Initializes the instance.

        Args:
            subparsers (`Optional[_SubParsersAction]`):
                The parent subparsers this command will create its parser on.
            args (`Optional[Namespace]`, defaults to `None`):
                The parsed arguments that are going to be use in self.run().
            command (`Optional[CommandInfo]`, defaults to `None`):
                The command info for this instance. This can be used to set the class attribute `COMMAND`.
            from_defaults_factory (`bool`, defaults to `False`):
                When setting the parser defaults, we create a second instance of self. By setting
                `from_defaults_factory=True`, we do not do unnecessary actions for setting the defaults, such as
                creating a parser.
        """
        # For leaf commands, it is redundant to define the `COMMAND` attribute since it was already defined in the
        # parent command, by doing this we pass this information between parent and child command.
        if command is not None:
            self.COMMAND = command

        if from_defaults_factory:
            if parser is None:
                raise ValueError(
                    "The instance of the original parser must be passed when creating a defaults factory, command: "
                    f"{self}."
                )
            self.parser = parser
            self.subparsers = subparsers
        else:
            if subparsers is None:
                raise ValueError(f"A subparsers instance is needed when from_defaults_factory=False, command: {self}.")
            self.parser = subparsers.add_parser(self.COMMAND.name, help=self.COMMAND.help)
            self.parse_args(self.parser)

            def defaults_factory(args):
                return self.__class__(
                    self.subparsers, args, command=self.COMMAND, from_defaults_factory=True, parser=self.parser
                )

            self.parser.set_defaults(func=defaults_factory)

            for subcommand in self.SUBCOMMANDS:
                if not isinstance(subcommand, CommandInfo):
                    raise ValueError(f"Subcommands must be instances of CommandInfo, but got {type(subcommand)} here.")
                self.register_subcommand(subcommand)

        self.args = args

    @property
    def subparsers(self):
        """
        This property handles how subparsers are created, which are only needed when registering a subcommand.
        If `self` does not have any subcommand, no subparsers should be created or it will mess with the command.
        This property ensures that we create subparsers only if needed.
        """
        subparsers = getattr(self, "_subparsers", None)
        if subparsers is None:
            if self.SUBCOMMANDS:
                self._subparsers = self.parser.add_subparsers()
            else:
                self._subparsers = None
        return self._subparsers

    @subparsers.setter
    def subparsers(self, subparsers: Optional["_SubParsersAction"]):
        self._subparsers = subparsers

    @property
    def registered_subcommands(self):
        if not hasattr(self, "_registered_subcommands"):
            self._registered_subcommands = []
        return self._registered_subcommands

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        pass

    def register_subcommand(self, command_info: CommandInfo):
        command_info.is_subcommand_info_or_raise()
        self.SUBCOMMANDS = self.SUBCOMMANDS + (command_info,)
        self.registered_subcommands.append(command_info.subcommand_class(self.subparsers, command=command_info))

    def run(self):
        self.parser.print_help()


class RootOptimumCLICommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="root", help="optimum-cli root command")

    def __init__(self, cli_name: str, usage: Optional[str] = None, args: Optional["Namespace"] = None):
        self.parser = ArgumentParser(cli_name, usage=usage)
        self.subparsers = self.parser.add_subparsers()
        self.args = None
