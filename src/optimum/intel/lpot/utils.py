#  Copyright 2021 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from torch.utils.data import DataLoader


class LpotDataLoader(DataLoader):

    @classmethod
    def from_pytorch_dataloader(cls, dataloader: DataLoader):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"Expected a PyTorch DataLoader, got: {type(dataloader)}.")
        lpot_dataloader = cls(dataloader.dataset)
        for key, value in dataloader.__dict__.items():
            lpot_dataloader.__dict__[key] = value
        return lpot_dataloader

    def __iter__(self):
        for input in super().__iter__():
            if not isinstance(input, (dict, tuple, list)):
                raise TypeError(f"Model calibration cannot use input of type {type(input)}.")
            label = input.get("labels") if isinstance(input, dict) else None
            yield input, label

