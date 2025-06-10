# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import fnmatch
import glob
import hashlib
import importlib
import json
import os
import re
import tempfile
from collections import defaultdict
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import filelock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Graph, Node
from tqdm.auto import tqdm
from transformers.utils import http_user_agent

from .core import HashableSlice, ParameterMeta, ParameterSlice


def ensure_divisibility(numerator: int, denominator: int) -> None:
    if numerator % denominator != 0:
        raise RuntimeError(
            f"{numerator} is not divisible by {denominator}, check if the parallel dimension of weight parameters is divisible "
            "by parallelism level(world size of tensor parallel group)"
        )


def is_activation(node: Node) -> bool:
    # only consider leaf Module activations
    if node.op != "call_module":
        return False
    mod = node.graph.owning_module
    return getattr(mod.get_submodule(node.target), "__module__", "").startswith("torch.nn.modules.activation")


def is_linear(node: Node) -> bool:
    if node.op != "call_module":
        return False
    mod = node.graph.owning_module
    return isinstance(mod.get_submodule(node.target), nn.Linear)


def is_embedding(node: Node) -> bool:
    if node.op != "call_module":
        return False
    mod = node.graph.owning_module
    return isinstance(mod.get_submodule(node.target), nn.Embedding)


def is_shape_consumer(node: Node) -> bool:
    if node.op == "call_method":
        return node.target in {"view", "reshape", "expand", "resize", "resize_"}
    elif node.op == "call_function":
        return node.target in {torch.reshape}
    return False


def is_output(node: Node) -> bool:
    return node.op == "output"


def is_shape_generator(node: Node) -> bool:
    return node.op == "call_method" and node.target == "size"


def is_cross_entropy(node: Node) -> bool:
    if node.op == "call_function":
        return node.target is F.cross_entropy
    elif node.op == "call_module":
        mod = node.graph.owning_module
        return isinstance(mod.get_submodule(node.target), nn.CrossEntropyLoss)
    return False


def is_cross_entropy_parallel_compatible(node: Node) -> bool:
    """
    For now `VocabParallelCrossEntropyLoss` does not support weighted mode, index ignoring and label smoothing.
    """
    if node.op == "call_function":
        weight = node.kwargs.get("weight", None)
        ignore_index = node.kwargs.get("ignore_index", -100)
        label_smoothing = node.kwargs.get("label_smoothing", 0.0)
        if len(node.args) > 2 and weight is None:
            weight = node.args[2]
        if len(node.args) > 4 and ignore_index == -100:
            ignore_index = node.args[4]
        if len(node.args) > 7 and label_smoothing == 0.0:
            label_smoothing = node.args[7]

        return weight is None and ignore_index == -100 and label_smoothing == 0.0

    elif node.op == "call_module":
        mod: nn.CrossEntropyLoss = node.graph.owning_module.get_submodule(node.target)
        weight, label_smoothing, ignore_index = mod.weight, mod.label_smoothing, mod.ignore_index
        return weight is None and ignore_index == -100 and label_smoothing == 0.0

    return False


def stable_topological_sort(graph: Graph):
    def _args(n: torch.fx.Node) -> List[torch.fx.node.Argument]:
        args: List[torch.fx.node.Argument] = []
        torch.fx.map_arg((n.args, n.kwargs), args.append)
        return args

    # Nodes are in exactly one of these three collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.
    ready = set()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()
        waiting_for = [x for x in _args(node) if x not in ready]
        if waiting_for:
            # We have unprocessed input nodes. Might as well wait for the last
            # arg so an already sorted list will only recheck this node once.
            waiting[waiting_for[-1]].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    assert not waiting and len(ready) == len(graph.nodes)


def meta_init(init_fn):
    @wraps(init_fn)
    def wrapper(*args, **kwargs):
        kwargs["device"] = kwargs.pop("device", torch.device("meta"))
        return init_fn(*args, **kwargs)

    return wrapper


@wraps(nn.Linear.forward)
def meta_aware_linear_forward(*args, **kwargs):
    self = args[0]
    input = args[1]

    if self.weight.device != torch.device("meta"):
        return F.linear(input, self.weight, self.bias)

    orig_device = input.device
    input = input.to("meta")
    meta_output = F.linear(input, self.weight, self.bias)
    return torch.empty_like(meta_output, device=orig_device)


@wraps(nn.Embedding.forward)
def meta_aware_embedding_forward(*args, **kwargs):
    self = args[0]
    input = args[1]

    if self.weight.device != torch.device("meta"):
        return F.embedding(
            input=input,
            weight=self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    orig_device = input.device
    input = input.to("meta")
    meta_output = F.embedding(
        input=input,
        weight=self.weight,
        padding_idx=self.padding_idx,
        max_norm=self.max_norm,
        norm_type=self.norm_type,
        scale_grad_by_freq=self.scale_grad_by_freq,
        sparse=self.sparse,
    )
    return torch.empty_like(meta_output, device=orig_device)


class MetaAwareMethodsPatcher:
    """
    A patcher class which patches `__init__` and `forward` methods on modules which will be put on meta
    devices for memory efficiency purposes during initialization.

    Note that for `__init__` method, it can be unpatched once we have finished the initialization of the
    model, however, for `forward`, we need it to constantly being patched during the whole process in case
    recompile happens and torch dynamo needs meta-aware `forward` to be able to re-capture the graph.
    """

    methods_to_patch: Dict[str, Callable] = [
        ("torch.nn.Linear.__init__", meta_init(nn.Linear.__init__)),
        ("torch.nn.Embedding.__init__", meta_init(nn.Embedding.__init__)),
        ("torch.nn.Linear.forward", meta_aware_linear_forward),
        ("torch.nn.Embedding.forward", meta_aware_embedding_forward),
    ]

    def __init__(self) -> None:
        self.patching_specs = []
        for orig, patch_fn in self.methods_to_patch:
            module_qualified_name, attribute_name = orig.rsplit(".", maxsplit=1)
            try:
                module = importlib.import_module(module_qualified_name)
            except ModuleNotFoundError as e:
                module_qualified_name, module_attribute_name = module_qualified_name.rsplit(".", maxsplit=1)
                module = importlib.import_module(module_qualified_name)
                try:
                    module = getattr(module, module_attribute_name)
                except AttributeError:
                    raise e
            orig_fn = getattr(module, attribute_name)

            # Module, Attribute, Patchee, Patcher, Status
            self.patching_specs.append([module, attribute_name, orig_fn, patch_fn, False])

    def _patch(self, identifier: str):
        for spec in self.patching_specs:
            # already patched
            if spec[-1]:
                continue
            if identifier in spec[1]:
                setattr(spec[0], spec[1], spec[3])
                spec[-1] = True

    def _unpatch(self, identifier: str):
        for spec in self.patching_specs:
            # already patched
            if not spec[-1]:
                continue
            if identifier in spec[1]:
                setattr(spec[0], spec[1], spec[2])
                spec[-1] = False

    def patch_meta_init(
        self,
    ):
        self._patch("init")

    def patch_meta_forward(
        self,
    ):
        self._patch("forward")

    def unpatch_meta_init(
        self,
    ):
        self._unpatch("init")

    def unpatch_meta_forward(
        self,
    ):
        self._unpatch("forward")

    def __enter__(
        self,
    ):
        self.patch_meta_init()
        self.patch_meta_forward()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unpatch_meta_init()


def initialize_parameter_meta(model: nn.Module) -> None:
    parameter_ids = set()
    for name, tensor in model.named_parameters(remove_duplicate=False):
        key = id(tensor)
        if key not in parameter_ids:
            setattr(
                tensor,
                "meta",
                ParameterMeta(
                    dim=0,
                    mapping={HashableSlice(None, None, None): ParameterSlice(source=name, shape=tuple(tensor.shape))},
                ),
            )
            parameter_ids.add(key)
        else:
            tensor.meta.is_tied = True


@torch.no_grad
def move_model_to_device(model: nn.Module, device: Union[torch.device, str]):
    """
    Move everything except tensors on meta devices on current device
    this function should be called before `intialize_parameter_meta`
    """
    for name, tensor in chain(model.named_parameters(), model.named_buffers()):
        if tensor.device == torch.device("meta"):
            continue
        splits = name.rsplit(".", maxsplit=1)
        if len(splits) == 1:
            parent_mod = model
            attr_name = splits[0]
        else:
            qualified_name = splits[0]
            parent_mod = model.get_submodule(qualified_name)
            attr_name = splits[1]
        new_tensor = tensor.to(device)
        if isinstance(tensor, nn.Parameter):
            new_tensor = nn.Parameter(new_tensor)
        setattr(parent_mod, attr_name, new_tensor)


temp_dir = tempfile.gettempdir()


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


# adpated from vllm.model_executor.model_loader.weight_utils.py
def download_model_from_hf(
    model_name_or_path: str,
    cache_dir: Optional[str],
    revision: Optional[str] = None,
    local_files_only: bool = False,
    skip_download_weights: bool = False,
) -> str:
    """Download model weights, index and config files from Hugging Face Hub.

    Args:
        model_name_or_path (`str`): The model name or path.
        cache_dir (`Optional[str]`): The cache directory to store the model
            weights. If None, will use HF defaults.
        revision (`Optional[str]`, defaults to `None`): The revision of the model.
        local_files_only(`bool`): Should only use local files if True.
        skip_download_weights (`bool`, defaults to `False`): Whether to skip downloading weights to disk.

    Returns:
        str: The path to the downloaded files.
    """
    import huggingface_hub.constants
    from huggingface_hub import HfApi, HfFileSystem
    from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

    allow_patterns = ["*.safetensors", "*.bin"]

    if not skip_download_weights and not huggingface_hub.constants.HF_HUB_OFFLINE:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break

    if skip_download_weights:
        # only need to download config file
        allow_patterns = [CONFIG_NAME]
    elif allow_patterns[0] == "*.safetensors":
        allow_patterns = allow_patterns + [CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME]
    else:
        allow_patterns = allow_patterns + [CONFIG_NAME, WEIGHTS_INDEX_NAME]

    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        hf_folder = HfApi(user_agent=http_user_agent()).snapshot_download(
            model_name_or_path,
            allow_patterns=allow_patterns,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE or local_files_only,
            tqdm_class=DisabledTqdm,
        )
    return hf_folder


# copied from optimum.neuron.utils.misc.py
def _original_filename_to_safetensors_filename(filename: str) -> str:
    """Transforms the filename for any kind of checkpoint to a safetensors equivalent."""
    from transformers.utils import SAFE_WEIGHTS_NAME

    _, extension = filename.rsplit(".", maxsplit=1)
    pattern = rf"\w+(-[0-9]*-of-[0-9]*)?\.{extension}"
    match_ = re.match(pattern, filename)
    if not match_:
        raise ValueError(f"Could not convert {filename} to a safetensor filename.")
    group_1 = match_.group(1)
    index_out_of_total_str = group_1 if group_1 is not None else ""
    safetensor_filename, safetensor_extension = SAFE_WEIGHTS_NAME.rsplit(".", maxsplit=1)
    return f"{safetensor_filename}{index_out_of_total_str}.{safetensor_extension}"


def convert_bin_to_safetensors(
    model_name_or_path: str, cache_dir: Optional[str], weight_files: List[str], weight_map: Dict[str, str]
):
    """Convert to pytorch bin files to their safetensors equivalent."""
    from safetensors.torch import save_file

    with get_lock(model_name_or_path, cache_dir):
        for weight_file in weight_files:
            weight_file_path = Path(weight_file)
            safetensors_filename = _original_filename_to_safetensors_filename(weight_file_path.name)
            output_dir = cache_dir if cache_dir else weight_file_path.parent
            output_file_path = os.path.join(output_dir, safetensors_filename)
            if not os.path.isfile(output_file_path):
                checkpoint = torch.load(weight_file, map_location=torch.device("cpu"))
                data_pointers = set()
                for k, v in checkpoint.items():
                    if v.data_ptr() in data_pointers:
                        v = v.detach().clone()
                    v = v.contiguous()
                    checkpoint[k] = v
                    data_pointers.add(v.data_ptr())
                save_file(checkpoint, output_file_path)
            keys = [key for key, value in weight_map.items() if value == weight_file]
            for key in keys:
                weight_map[key] = output_file_path


def try_collect_weight_map(model_name_or_path: str, cache_dir: Optional[str], folder_path: str) -> Dict[str, str]:
    """Try collecting weight mapping information from the model folder."""
    from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

    weight_map = {}
    use_safetensors, weight_patterns = False, ["*safetensors", "*.bin"]
    for pattern in weight_patterns:
        if len(glob.glob(os.path.join(folder_path, pattern))) > 0:
            use_safetensors = pattern == "*.safetensors"
            break
    index_path = os.path.join(folder_path, SAFE_WEIGHTS_INDEX_NAME if use_safetensors else WEIGHTS_INDEX_NAME)
    weight_files = glob.glob(os.path.join(folder_path, "*.safetensors" if use_safetensors else "*.bin"))

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index_dict = json.load(f)
        weight_map = {k: os.path.join(folder_path, v) for k, v in index_dict["weight_map"].items()}

    # convert bin files to safetensors, modify `weight_map` meanwhile
    if not use_safetensors:
        convert_bin_to_safetensors(model_name_or_path, cache_dir, weight_files, weight_map)

    # last resort: try directly construct weight_map from weight files
    if not weight_map:
        from safetensors import safe_open

        # should have safetensors on disk in any case
        weight_files = glob.glob(os.path.join(folder_path, "*.safetensors"))
        for weight_file in weight_files:
            with safe_open(filename=weight_file, framework="pt") as f:
                for key in f.keys():
                    weight_map[key] = weight_file
    return weight_map
