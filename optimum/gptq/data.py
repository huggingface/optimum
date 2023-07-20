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


import random
from typing import Any, Dict, List, Union

import numpy as np
import torch
from datasets import load_dataset


def prepare_dataset(
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


def get_wikitext2(tokenizer, seqlen, nsamples, split="train"):
    if split == "train":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif split == "validation":
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "".join([" \n" if s == "" else s for s in data["text"]])

    enc = tokenizer(text, return_tensors="pt")
    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_c4(tokenizer, seqlen, nsamples, split="train"):
    if split == "train":
        data = load_dataset(
            "allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
        )
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def get_c4_new(tokenizer, seqlen, nsamples, split="train"):
    if split == "train":
        data = load_dataset(
            "allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
        )
    elif split == "validation":
        data = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    dataset = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(data) - 1)
            enc = tokenizer(data[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def get_ptb(tokenizer, seqlen, nsamples, split="train"):
    if split == "train":
        data = load_dataset("ptb_text_only", "penn_treebank", split="train")
    elif split == "validation":
        data = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    enc = tokenizer(" ".join(data["sentence"]), return_tensors="pt")

    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return dataset


def get_ptb_new(tokenizer, seqlen, nsamples, split="train"):
    if split == "train":
        data = load_dataset("ptb_text_only", "penn_treebank", split="train")
    elif split == "validation":
        data = load_dataset("ptb_text_only", "penn_treebank", split="test")

    enc = tokenizer(" ".join(data["sentence"]), return_tensors="pt")

    dataset = []
    for _ in range(nsamples):
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = enc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        dataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return dataset


def get_dataset(
    dataset_name: str, tokenizer: Any, nsamples: int = 128, seqlen: int = 2048, seed: int = 0, split: str = "train"
):
    """
    Get the dataset from the original paper on GTPQ

    Args:
        dataset_name (`str`):
            Dataset name. The options are ['wikitext2','c4','ptb','c4-new','ptb_new']
        tokenizer (`Any`):
            Tokenizer of the model
        nsamples (`int`, *optional*, defaults to `128`):
            Number of samples
        seqlen (`int`, *optional*, defaults to `2048`):
            The sequence length of the model
        seed (`int`, *optional*, defaults to `0`):
            Seed
        split (`str`, *optional*, defaults to `train`):
            Split of the dataset. Can be either "train" or "validation"
    Returns:
        `List[Dict[str,torch.LongTensor]]`: The tokenized dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    if split not in ["train", "test"]:
        raise ValueError(f"The split need to be 'train' or 'validation' but found {split}")
    if dataset_name == "wikitext2":
        return get_wikitext2(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen)
    elif dataset_name == "c4":
        return get_c4(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen)
    elif dataset_name == "c4-new":
        return get_c4_new(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen)
    elif dataset_name == "ptb":
        return get_ptb(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen)
    elif dataset_name == "ptb-new":
        return get_ptb_new(tokenizer=tokenizer, nsamples=nsamples, seqlen=seqlen)
    else:
        raise ValueError(f"Expected a value in ['wikitext2','c4','ptb','c4-new','ptb-new'] but found {dataset_name}")
