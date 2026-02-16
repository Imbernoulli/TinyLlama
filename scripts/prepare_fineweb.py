import json
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str = "train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    """
    Prepare FineWeb-Edu dataset for training.
    FineWeb-Edu is a filtered version of FineWeb focusing on educational content.
    """
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset
    filenames = filenames_subset

    if not filenames:
        raise RuntimeError(
            f"No files found at {source_path}. \n"
            "Make sure you download the FineWeb-Edu data first.\n"
            "You can download it using: huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset --local-dir <destination>"
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_fineweb_{process_id}",
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        # FineWeb-Edu uses parquet format
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(filepath)
            df = table.to_pandas()

            for idx, row in tqdm(df.iterrows(), total=len(df)):
                # FineWeb-Edu has 'text' column containing the content
                text = row['text']
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            # Try JSON format as fallback
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in tqdm(f):
                        data = json.loads(line)
                        text = data.get('text', '')
                        if text:
                            text_ids = tokenizer.encode(text)
                            builder.add_array(np.array(text_ids, dtype=builder.dtype))
            except Exception as e2:
                print(f"Failed to process {filepath} as JSON too: {e2}")
                continue

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/fineweb-edu"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/fineweb_edu_processed"),
    chunk_size: int = 2049 * 1024,
    split: str = "train",
    percentage: float = 1.0,
) -> None:
    """
    Main function to prepare FineWeb-Edu dataset.

    Args:
        source_path: Path to the downloaded FineWeb-Edu dataset
        tokenizer_path: Path to the tokenizer model
        destination_path: Path to save the processed dataset
        chunk_size: Size of each chunk
        split: Dataset split (train/validation/test)
        percentage: Percentage of data to use (for testing)
    """
    import time
    import glob

    # FineWeb-Edu uses parquet files
    # The structure may vary, but typically: data/train/*.parquet
    pattern = os.path.join(source_path, split, "**/*.parquet")
    filenames = glob.glob(pattern, recursive=True)

    # Also try jsonl format as fallback
    if not filenames:
        pattern = os.path.join(source_path, split, "**/*.jsonl")
        filenames = glob.glob(pattern, recursive=True)

    if not filenames:
        # Try without split subdirectory
        pattern = os.path.join(source_path, "**/*.parquet")
        filenames = glob.glob(pattern, recursive=True)

    if not filenames:
        print(f"No files found at {source_path}")
        print("Please download FineWeb-Edu dataset first using:")
        print("huggingface-cli download HuggingFaceFW/fineweb-edu --repo-type dataset --local-dir data/fineweb-edu")
        print("\nOr use the datasets library:")
        print("from datasets import load_dataset")
        print("dataset = load_dataset('HuggingFaceFW/fineweb-edu', split='train')")
        return

    filenames = filenames[:int(len(filenames) * percentage)]
    print(f"Found {len(filenames)} files to process")

    num_processes = min(cpu_count(), len(filenames))
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        if len(subset) == 0:
            continue
        p = Process(
            target=prepare_full,
            args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
