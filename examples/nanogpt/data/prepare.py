# Tokenise the OpenWebText dataset and write train.bin / val.bin for GPT-2 training.
#
# Run once locally (or on a GCE VM with enough RAM and disk), then upload the
# two output files to GCS so the GKE training job can fetch them:
#
#   python prepare.py
#   gsutil -m cp train.bin val.bin gs://your-bucket/data/openwebtext/
#
# Output sizes:
#   train.bin  ~17 GB   (~9B tokens, uint16)
#   val.bin    ~8.5 MB  (~4M tokens, uint16)
#
# Requirements:
#   pip install datasets tiktoken tqdm numpy

import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Number of workers for parallel tokenisation and dataset loading.
# Good default is ~(cpu_cores // 2).
num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    # ~54 GB in the HuggingFace cache; ~8M documents.
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)

    # openwebtext only has a 'train' split; carve out a small validation set.
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    # DatasetDict({
    #   train: Dataset({features: ['text'], num_rows: 8009762})
    #   val:   Dataset({features: ['text'], num_rows: 4007})
    # })

    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)  # append <|endoftext|> (50256)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenising",
        num_proc=num_proc,
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # train.bin  ~17 GB   (~9,035,582,198 tokens)
    # val.bin    ~8.5 MB  (~4,434,897 tokens)
    print("Done. Upload to GCS:")
    print("  gsutil -m cp train.bin val.bin gs://your-bucket/data/openwebtext/")
