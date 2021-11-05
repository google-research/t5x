from pathlib import Path
from typing import Optional

import seqio
from datasets import load_dataset
import tensorflow as tf

def load_from_local(dataset_dir: Path):
    dataset_list = {
        "train": [
            "train/00.jsonl",
            "train/01.jsonl",
            "train/02.jsonl",
            "train/03.jsonl",
            "train/04.jsonl",
            "train/05.jsonl",
            "train/06.jsonl",
            "train/07.jsonl",
            "train/08.jsonl",
            "train/09.jsonl",
            "train/10.jsonl",
            "train/11.jsonl",
            "train/12.jsonl",
            "train/13.jsonl",
            "train/14.jsonl",
            "train/15.jsonl",
            "train/16.jsonl",
            "train/17.jsonl",
            "train/18.jsonl",
            "train/19.jsonl",
            "train/20.jsonl",
            "train/21.jsonl",
            "train/22.jsonl",
            "train/23.jsonl",
            "train/24.jsonl",
            "train/25.jsonl",
            "train/26.jsonl",
            "train/27.jsonl",
            "train/28.jsonl",
            "train/29.jsonl"
        ],
        "test": [
            "test.jsonl"
        ],
        "val": [
            "val.jsonl"
        ],
    }

    for split_name, filepaths in dataset_list:
        load_dataset("json", data_files=[f"{dataset_dir}/{filepath}" for filepath in filepaths], data="text")

def load_from_urls():
    remote_urls = {
        "test": [
            "https://the-eye.eu/public/AI/pile/test.jsonl.zst",
        ]
    }

    return {split_name: load_dataset("json", data_files=urls, field="text") for split_name, urls in remote_urls}

class PileDatasetFnCallable(seqio.DatasetFnCallable):
    def __init__(self):
        self.datasets = load_from_urls()

    def __call__(
        self,
        split: str,
        shuffle_files: bool,
        seed: Optional[int] = None
    ) -> tf.data.Dataset:
        datasets = load_from_urls()
        if split not in datasets:
            raise ValueError(f"Unrecognized split value, got {split} expected {datasets.keys()}")

        dataset = datasets[split]
        return dataset.to_tf_dataset(
            columns="text",
            batch_size=1000,
            shuffle=shuffle_files
        )

