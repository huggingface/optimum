# coding=utf-8
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
"""Image classification processing."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers.image_processing_utils import VALID_SIZE_DICT_KEYS, BaseImageProcessor

from .. import logging
from .base import TaskProcessor


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict
    from transformers import PretrainedConfig


class ImageClassificationProcessing(TaskProcessor):
    ACCEPTED_PREPROCESSOR_CLASSES = (BaseImageProcessor,)
    DEFAULT_DATASET_ARGS = "uoft-cs/cifar10"
    DEFAUL_DATASET_DATA_KEYS = {"image": "img"}
    ALLOWED_DATA_KEY_NAMES = {"image"}
    DEFAULT_REF_KEYS = ["answers"]

    def __init__(
        self,
        config: "PretrainedConfig",
        preprocessor: "BaseImageProcessor",
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config, preprocessor, preprocessor_kwargs=preprocessor_kwargs)
        size = None
        for dict_keys in VALID_SIZE_DICT_KEYS:
            try:
                size = tuple(self.preprocessor.size[key] for key in dict_keys)
            except KeyError:
                pass
        if size is None:
            raise ValueError(f"Could not retrieve the size information from {preprocessor}")

        self.transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                Normalize(mean=self.preprocessor.image_mean, std=self.preprocessor.image_std),
            ]
        )

    def dataset_processing_func(
        self, example: Dict[str, Any], data_keys: Dict[str, str], ref_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        image = example[data_keys["image"]]
        example["pixel_values"] = self.transforms(image.convert("RGB")).to(torch.float32).numpy()
        return example

    def try_to_guess_data_keys(self, column_names: List[str]) -> Optional[Dict[str, str]]:
        image_key_name = None
        for name in column_names:
            if "image" in name:
                image_key_name = name
                break
        if image_key_name is None and len(column_names) == 2:
            if "label" in column_names[0]:
                image_key_name = column_names[1]
            elif "label" in column_names[1]:
                image_key_name = column_names[0]

        return {"image": image_key_name} if image_key_name is not None else None

    def try_to_guess_ref_keys(self, column_names: List[str]) -> Optional[List[str]]:
        for name in column_names:
            if "label" in name:
                return [name]

    def load_dataset(
        self,
        path: str,
        data_keys: Optional[Dict[str, str]] = None,
        ref_keys: Optional[List[str]] = None,
        only_keep_necessary_columns: bool = False,
        load_smallest_split: bool = False,
        num_samples: Optional[int] = None,
        shuffle: bool = False,
        **load_dataset_kwargs,
    ) -> Union["DatasetDict", "Dataset"]:
        dataset = super().load_dataset(
            path,
            data_keys=data_keys,
            ref_keys=ref_keys,
            only_keep_necessary_columns=only_keep_necessary_columns,
            load_smallest_split=load_smallest_split,
            num_samples=num_samples,
            shuffle=shuffle,
            **load_dataset_kwargs,
        )
        # TODO: do we want to do that here?
        # eval_dataset = eval_dataset.align_labels_with_mapping(self.config.label2id, self.ref_keys[0])
        return dataset
