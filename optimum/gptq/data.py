#  Copyright 2023 The HuggingFace Team. All rights reserved.
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


import logging
import random
from typing import Dict, List, Union

import numpy as np
import torch
from datasets import load_dataset


def prepare_examples(
    examples: List[Dict[str, Union[List[int], torch.LongTensor]]], pad_token_id: int = None, batch_size: int = 1
):
    new_examples = []
    for example in examples:
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]
        if isinstance(input_ids, List):
            input_ids = [input_ids]
        if isinstance(attention_mask, List):
            attention_mask = [attention_mask]
        new_examples.append(
            {"input_ids": torch.LongTensor(input_ids), "attention_mask": torch.LongTensor(attention_mask)}
        )
    if batch_size > 1 and pad_token_id is None:
        raise ValueError(
            "You need to pass a `pad_token_id` in `quantize_model` if you want to have examples with batch size > 1"
        )
    new_examples = [
        collate_data(new_examples[start : start + batch_size], pad_token_id)
        for start in range(0, len(new_examples), batch_size)
    ]
    return new_examples


def collate_data(
    blocks: List[Dict[str, List[torch.LongTensor]]], pad_token_id: int = None, contain_labels: bool = False
) -> Dict[str, torch.LongTensor]:
    def pad_block(block, pads):
        return torch.cat((pads.to(block.device), block), dim=-1).long()

    input_ids_blocks = [block["input_ids"] for block in blocks]
    attention_mask_blocks = [block["attention_mask"] for block in blocks]
    if contain_labels:
        label_blocks = [block["labels"] for block in blocks]
        label_max_len = max([block.size(-1) for block in label_blocks])

    bsz = len(blocks)
    inp_max_len = max([block.size(-1) for block in input_ids_blocks])

    for i in range(bsz):
        block_bsz, block_inp_len = input_ids_blocks[i].shape
        pad_num = inp_max_len - block_inp_len
        if pad_num > 0:
            input_ids_blocks[i] = pad_block(input_ids_blocks[i], torch.ones((block_bsz, pad_num)) * pad_token_id)
            attention_mask_blocks[i] = pad_block(attention_mask_blocks[i], torch.zeros((block_bsz, pad_num)))
        if contain_labels:
            block_label_len = label_blocks[i].shape[-1]
            label_pad_num = label_max_len - block_label_len
            if label_pad_num > 0:
                label_blocks[i] = pad_block(label_blocks[i], torch.ones((block_bsz, label_pad_num)) * -100)

    data = {
        "input_ids": torch.cat(input_ids_blocks, dim=0).long(),
        "attention_mask": torch.cat(attention_mask_blocks, dim=0).long(),
    }
    if contain_labels:
        data["labels"] = torch.cat(label_blocks, dim=0).long()

    return data


def get_wikitext2(tokenizer, seqlen, nsamples):
    logger = logging.getLogger(__name__)

    wikidata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wikilist = [" \n" if s == "" else s for s in wikidata["text"]]

    text = "".join(wikilist)
    logger.info("Tokenising wikitext2")
    trainenc = tokenizer(text, return_tensors="pt")

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset


def get_c4(tokenizer, seqlen, nsamples):
    traindata = load_dataset(
        "allenai/c4",
        "allenai--c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
        use_auth_token=False,
    )
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        trainloader.append({"input_ids": inp, "attention_mask": attention_mask})

    return trainloader


def get_examples(dataset_name, tokenizer, nsamples=128, seqlen=2048, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if dataset_name == "wikitext2":
        return get_wikitext2(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen)
    elif dataset_name == "c4":
        return get_c4(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen)
    else:
        raise ValueError(f"Expected a value in ['wikitext2','c4'] but found {dataset_name}")
