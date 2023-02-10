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

import platform
from argparse import ArgumentParser

import huggingface_hub
from transformers import __version__ as transformers_version
from transformers.utils import is_tf_available, is_torch_available

from ..version import __version__ as version
from . import BaseOptimumCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseOptimumCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env", help="Get information about the environment used.")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        pt_version = "not installed"
        pt_cuda_available = "NA"
        if is_torch_available():
            import torch

            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()

        tf_version = "not installed"
        tf_cuda_available = "NA"
        if is_tf_available():
            import tensorflow as tf

            tf_version = tf.__version__
            try:
                # deprecated in v2.1
                tf_cuda_available = tf.test.is_gpu_available()
            except AttributeError:
                # returns list of devices, convert to bool
                tf_cuda_available = bool(tf.config.list_physical_devices("GPU"))

        info = {
            "`optimum` version": version,
            "`transformers` version": transformers_version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "Huggingface_hub version": huggingface_hub.__version__,
            "PyTorch version (GPU?)": f"{pt_version} (cuda availabe: {pt_cuda_available})",
            "Tensorflow version (GPU?)": f"{tf_version} (cuda availabe: {tf_cuda_available})",
        }

        print("\nCopy-and-paste the text below in your GitHub issue:\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"
