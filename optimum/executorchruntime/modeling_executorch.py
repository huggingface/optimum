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

"""ExecuTorchModelForXXX classes, allowing to run ExecuTorch Models with ExecuTorch Runtime using the same API as Transformers."""

import logging
import os
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Union

import torch
from executorch.extension.pybindings.portable_lib import (
    ExecuTorchModule,
    _load_for_executorch,
)
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import (
    AutoModelForCausalLM,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from ..exporters.executorch import main_export
from ..modeling_base import OptimizedModel


logger = logging.getLogger(__name__)


class ExecuTorchModelForCausalLM(OptimizedModel):
    """
    ExecuTorch model with a causal language modeling head for inference using the ExecuTorch Runtime.

    This class provides an interface for loading, running, and generating outputs from a causal language model
    optimized for ExecuTorch Runtime. It includes utilities for exporting and loading pre-trained models
    compatible with ExecuTorch runtime.

    Attributes:
        auto_model_class (`Type`):
            Associated Transformers class, `AutoModelForCausalLM`.
        et_model (`ExecuTorchModule`):
            The loaded ExecuTorch model.
        use_kv_cache (`bool`):
            Whether key-value caching is enabled. For performance reasons, the exported model is
            optimized to use a static cache.
        max_cache_size (`int`):
            Maximum sequence length supported by the cache.
        max_batch_size (`int`):
            Maximum supported batch size.
        dtype (`str`):
            Data type of the model parameters.
        bos_token_id (`int`):
            Beginning-of-sequence token ID.
        eos_token_id (`int`):
            End-of-sequence token ID.
        vocab_size (`int`):
            Size of the model vocabulary.
    """

    auto_model_class = AutoModelForCausalLM

    def __init__(
        self,
        model: "ExecuTorchModule",
        config: "PretrainedConfig",
    ):
        super().__init__(model, config)
        self.et_model = model
        metadata = self.et_model.method_names()
        logging.info(f"Load all static methods: {metadata}")
        if "use_kv_cache" in metadata:
            self.use_kv_cache = self.et_model.run_method("use_kv_cache")[0]
        if "get_max_seq_len" in metadata:
            self.max_cache_size = self.et_model.run_method("get_max_seq_len")[0]
        if "get_max_batch_size" in metadata:
            self.max_batch_size = self.et_model.run_method("get_max_batch_size")[0]
        if "get_dtype" in metadata:
            self.dtype = self.et_model.run_method("get_dtype")[0]
        if "get_bos_id" in metadata:
            self.bos_token_id = self.et_model.run_method("get_bos_id")[0]
        if "get_eos_id" in metadata:
            self.eos_token_id = self.et_model.run_method("get_eos_id")[0]
        if "get_vocab_size" in metadata:
            self.vocab_size = self.et_model.run_method("get_vocab_size")[0]

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model, which is compatible with the ExecuTorch runtime for LLM.

        Args:
            input_ids (`torch.Tensor`): Tensor representing current input token id to the model.
            cache_position (`torch.Tensor`): Tensor representing current input position in the cache.

        Returns:
            torch.Tensor: Logits output from the model.
        """
        return self.et_model.forward((input_ids, cache_position))[0]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        export: bool = True,
        task: str = "",
        recipe: str = "",
        config: "PretrainedConfig" = None,
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        force_download: bool = False,
        local_files_only: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        **kwargs,
    ) -> "ExecuTorchModelForCausalLM":
        """
        Load a pre-trained ExecuTorch model.

        Args:
            model_name_or_path (`Union[str, Path]`):
                Model ID on huggingface.co or path on disk to the model repository to export. Example: `model_name_or_path="meta-llama/Llama-3.2-1B"` or `mode_name_or_path="/path/to/model_folder`.
            export (`bool`, *optional*, defaults to `True`):
                If `True`, the model will be exported from eager to ExecuTorch after fetched from huggingface.co. `model_name_or_path` must be a valid model ID on huggingface.co.
                If `False`, the previously exported ExecuTorch model will be loaded from a local path. `model_name_or_path` must be a valid local directory where a `model.pte` is stored.
            task (`str`, defaults to `""`):
                The task to export the model for, e.g. "text-generation". It is required to specify a task when `export` is `True`.
            recipe (`str`, defaults to `""`):
                The recipe to use to do the export, e.g. "xnnpack". It is required to specify a task when `export` is `True`.
            config (`PretrainedConfig`, *optional*):
                Configuration of the pre-trained model.
            subfolder (`str`, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
                specify the folder name here.
            revision (`str`, defaults to `"main"`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, defaults to `None`):
                Path indicating where to store cache. The default Hugging Face cache path will be used by default.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            local_files_only (`Optional[bool]`, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
                Deprecated. Please use the `token` argument instead.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
            **kwargs:
                Additional configuration options to tasks and recipes.

        Returns:
            `ExecuTorchModelForCausalLM`: An instance of the ExecuTorch model for text generation task.
        """
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if export:
            # Fetch the model from huggingface.co and export it to ExecuTorch
            if task == "":
                raise ValueError("Please specify a task to export the model for.")
            if recipe == "":
                raise ValueError("Please specify a recipe to export the model for.")
            return cls._export(
                model_id=model_name_or_path,
                task=task,
                recipe=recipe,
                config=config,
                **kwargs,
            )
        else:
            # Load the ExecuTorch model from a local path
            return cls._from_pretrained(
                model_dir_path=model_name_or_path,
                config=config,
            )

    @classmethod
    def _from_pretrained(
        cls,
        model_dir_path: Union[str, Path],
        config: PretrainedConfig,
        subfolder: str = "",
        revision: Optional[str] = None,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        force_download: bool = False,
        local_files_only: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
    ) -> "ExecuTorchModelForCausalLM":
        """
        Load a pre-trained ExecuTorch model from a local directory.

        Args:
            model_dir_path (`Union[str, Path]`):
                Path to the directory containing the ExecuTorch model file (`model.pte`).
            config (`PretrainedConfig`, *optional*):
                Configuration of the pre-trained model.
            subfolder (`str`, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
                specify the folder name here.
            revision (`str`, defaults to `"main"`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            cache_dir (`Optional[str]`, defaults to `None`):
                Path indicating where to store cache. The default Hugging Face cache path will be used by default.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            local_files_only (`Optional[bool]`, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
                Deprecated. Please use the `token` argument instead.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).

        Returns:
            `ExecuTorchModelForCausalLM`: The initialized ExecuTorch model.

        """
        full_path = os.path.join(f"{model_dir_path}", "model.pte")
        model = _load_for_executorch(full_path)
        logging.info(f"Loaded model from {full_path}")
        logging.debug(f"{model.method_meta('forward')}")
        return cls(
            model=model,
            config=config,
        )

    def _save_pretrained(self, save_directory):
        """
        Saves a model weights into a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.
        """
        raise NotImplementedError

    @classmethod
    def _export(
        cls,
        model_id: str,
        task: str,
        recipe: str,
        config: PretrainedConfig,
        cache_dir: str = HUGGINGFACE_HUB_CACHE,
        trust_remote_code: bool = False,
        subfolder: str = "",
        revision: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        token: Optional[Union[bool, str]] = None,
        **kwargs,
    ):
        """
        Fetch a model from the Hugging Face Hub and export it to ExecuTorch format.

        Args:
            model_id (`str`):
                Model ID on huggingface.co, for example: `model_name_or_path="meta-llama/Llama-3.2-1B"`.
            task (`str`):
                The task to export the model for, e.g. "text-generation".
            recipe (`str`):
                The recipe to use to do the export, e.g. "xnnpack".
            config (`PretrainedConfig`, *optional*):
                Configuration of the pre-trained model.
            cache_dir (`Optional[str]`, defaults to `None`):
                Path indicating where to store cache. The default Hugging Face cache path will be used by default.
            trust_remote_code (`bool`, defaults to `False`):
                Allows to use custom code for the modeling hosted in the model repository. This option should only be set for repositories
                you trust and in which you have read the code, as it will execute on your local machine arbitrary code present in the
                model repository.
            subfolder (`str`, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
                specify the folder name here.
            revision (`str`, defaults to `"main"`):
                Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.
            force_download (`bool`, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            local_files_only (`Optional[bool]`, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            use_auth_token (`Optional[Union[bool,str]]`, defaults to `None`):
                Deprecated. Please use the `token` argument instead.
            token (`Optional[Union[bool,str]]`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
            **kwargs:
                Additional configuration options to tasks and recipes.

        Returns:
            `ExecuTorchModelForCausalLM`: The loaded and exported ExecuTorch model.

        """
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        # Export to ExecuTorch and save the pte file to the temporary directory
        main_export(
            model_name_or_path=model_id,
            output_dir=save_dir_path,
            task=task,
            recipe=recipe,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        return cls._from_pretrained(
            model_dir_path=save_dir_path,
            config=config,
            use_auth_token=use_auth_token,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
        )

    def generate(
        self,
        prompt_tokens: List[int],
        echo: bool = False,
        pos_base: int = 0,
        max_seq_len: Optional[int] = None,
    ) -> List[int]:
        """
        Generate tokens from a prompt using the ExecuTorch model.

        Args:
            prompt_tokens (List[int]):
                List of token IDs representing the prompt.
            echo (`bool`, *optional*):
                Whether to include prompt tokens in the generated output. Defaults to `False`.
            pos_base (`int`, *optional*):
                Base position for the prompt tokens. Defaults to 0.
            max_seq_len (`int`, *optional*):
                Maximum sequence length for the generated output.
                Defaults to None and uses the model's `max_cache_size` attribute.
                Will be truncated to maximal cache size if larger than `max_cache_size`.

        Returns:
            List[int]: List of generated token IDs.

        Note:
            Temporarily implemented this method in Python due to limited access to ExecuTorch's c++ LLM runner via pybind.
            Expect improvements to the pybind interface in ExecuTorch version 0.4.1.
        """
        self.device = torch.device("cpu")
        if max_seq_len is None:
            # Default to max_cache_size if max_seq_len is not specified
            max_seq_len = self.max_cache_size
        elif max_seq_len > self.max_cache_size:
            logging.warning(
                f"max_seq_len={max_seq_len} is larger than max_cache_size={self.max_cache_size}. Generating tokens will be truncated to max_cache_size."
            )
            max_seq_len = self.max_cache_size
        generated_tokens = []

        # prefill
        for i, prompt_token in enumerate(prompt_tokens):
            logits = self.forward(
                input_ids=torch.tensor([prompt_token], dtype=torch.long, device=self.device).unsqueeze(0),
                cache_position=torch.tensor([i], dtype=torch.long, device=self.device),
            )

        next_token = torch.argmax(logits, dim=-1).item()
        generated_tokens = prompt_tokens + [next_token]

        while len(generated_tokens) < max_seq_len:
            logits = self.forward(
                input_ids=torch.tensor([next_token], dtype=torch.long, device=self.device).unsqueeze(0),
                cache_position=torch.tensor(
                    [pos_base + len(generated_tokens) - 1],
                    dtype=torch.long,
                    device=self.device,
                ),
            )
            next_token = torch.argmax(logits, dim=-1).item()
            generated_tokens.append(next_token)
            if next_token == self.eos_token_id:
                break

        return generated_tokens if echo else generated_tokens[len(prompt_tokens) :]

    def text_generation(
        self,
        tokenizer: "PreTrainedTokenizer",
        prompt: str,
        echo: bool = True,
        max_seq_len: Optional[int] = None,
    ):
        """
        Perform text generation task for a given prompt using the ExecuTorch model.

        Args:
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer used to encode and decode the prompt and output.
            prompt (`str`):
                The text prompt to complete.
            echo (`bool`, *optional*):
                Whether to include prompt tokens in the generated output. Defaults to `True`.
            max_seq_len (`int`, *optional*):
                Maximum sequence length for the generated output.
                Defaults to None and uses the model's `max_cache_size` attribute.
                Will be truncated to maximal cache size if larger than `max_cache_size`.
        """
        self.tokenizer = tokenizer

        # Sanity check
        if self.tokenizer.bos_token_id is not None and self.tokenizer.bos_token_id != self.bos_token_id:
            raise ValueError(
                f"The tokenizer's bos_token_id={self.tokenizer.bos_token_id} must be the same as the model's bos_token_id={self.bos_token_id}."
            )
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id != self.eos_token_id:
            raise ValueError(
                f"The tokenizer's eos_token_id={self.tokenizer.eos_token_id} must be the same as the model's eos_token_id={self.eos_token_id}."
            )

        prompt_tokens = self.tokenizer.encode(prompt)
        generated_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            echo=echo,
            max_seq_len=max_seq_len,
        )
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
