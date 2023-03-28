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

from abc import ABC, abstractmethod
from argparse import RawTextHelpFormatter
from dataclasses import dataclass
from typing import Tuple, Type, Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace


@dataclass(frozen=True)
class CommandInfo:
    name: str
    help: str
    subcommand_class: Optional[Type] = None
    formatter_class: Type = RawTextHelpFormatter

    @property
    def is_subcommand_info(self):
        return self.subcommand_class is not None
    
    def is_subcommand_info_or_raise(self):
        if not self.is_subcommand_info:
            raise ValueError(f"The command info must define a subcommand_class attribute, but got: {self}.")


class InvalidCLICommand(Exception):
    pass


class BaseOptimumCLICommand(ABC):
    COMMAND: CommandInfo
    SUBCOMMANDS: Tuple[CommandInfo, ...] = tuple()

    def __init__(self, parser: "ArgumentParser", args: Optional["Namespace"] = None, command: Optional[CommandInfo] = None):
        if command is not None:
            self.COMMAND = command

        self.parser = parser.add_parser(self.COMMAND.name, help=self.COMMAND.help)
        self.subparsers = self.parser.add_subparsers()
        self.parse_args(self.parser)

        def defaults_factory(args):
            return self.__class__(self.subparsers, args, command=command)

        self.parser.set_defaults(func=defaults_factory)

        for subcommand in self.SUBCOMMANDS:
            if not isinstance(subcommand, CommandInfo):
                raise ValueError(f"Subcommands must be instances of CommandInfo, but got {type(subcommand)} here.")
            self.register_subcommand(subcommand)

        self.args = args

    @classmethod
    def add_subcommand(cls, command_info: CommandInfo):
        command_info.is_subcommand_info_or_raise()
        cls.SUBCOMMANDS = cls.SUBCOMMANDS + (command_info,)

    @staticmethod
    def parse_args(parser: "ArgumentParser"):
        pass


    def register_subcommand(self, command_info: CommandInfo):
        command_info.is_subcommand_info_or_raise()
        command_info.subcommand_class(self.subparsers, command=command_info)

    def run(self):
        raise NotImplementedError()

