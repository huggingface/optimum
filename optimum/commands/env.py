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


from .base import BaseOptimumCLICommand, CommandInfo


class EnvironmentCommand(BaseOptimumCLICommand):
    COMMAND = CommandInfo(name="env", help="Get information about the environment used.")

    @staticmethod
    def format_dict(d):
        return "\n".join([f"- {prop}: {val}" for prop, val in d.items()]) + "\n"

    def run(self):
        import platform

        import huggingface_hub
        from transformers import __version__ as transformers_version
        from transformers.utils import is_torch_available

        from ..version import __version__ as version

        pt_version = "not installed"
        pt_cuda_available = "NA"
        if is_torch_available():
            import torch

            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()

        info = {
            "`optimum` version": version,
            "`transformers` version": transformers_version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "Huggingface_hub version": huggingface_hub.__version__,
            "PyTorch version (GPU?)": f"{pt_version} (cuda available: {pt_cuda_available})",
        }

        print("\nCopy-and-paste the text below in your GitHub issue:\n")
        print(self.format_dict(info))

        return info
