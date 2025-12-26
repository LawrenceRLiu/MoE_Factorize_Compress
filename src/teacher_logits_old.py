"""
Teacher Logits Generation and Caching

Efficiently generates and caches teacher model logits for knowledge distillation.
Stores only top-k logits and their indices to save memory.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging
from tqdm import tqdm
from dataclasses import dataclass
import json
import h5py

from transformers import AutoModelForCausalLM, PreTrainedModel
from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class LogitsCacheMetadata:
    """Metadata for cached logits."""
    num_samples: int
    max_seq_length: int
    top_k: int
    vocab_size: int
    teacher_model: str
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = None


class TeacherLogitsGenerator:
    """
    Generates and caches teacher model logits for knowledge distillation.

    Uses HDF5 for efficient storage of large logit datasets.
    Stores only top-k logits and their indices to reduce memory footprint.
    """

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        cache_dir: str,
        top_k: int = 64,
        device: str = "cuda"
    ):
        """
        Initialize teacher logits generator.

        Args:
            teacher_model: Pre-trained teacher model
            cache_dir: Directory to cache logits
            top_k: Number of top logits to store per position
            device: Device to run teacher model on
        """
        self.teacher_model = teacher_model
        self.teacher_model.eval()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.device = device

        # Move teacher to device
        if hasattr(teacher_model, 'device'):
            logger.info(f"Teacher model already on device: {teacher_model.device}")
        else:
            self.teacher_model = self.teacher_model.to(device)

    @torch.no_grad()
    def generate_and_cache_logits(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        max_length: int = 2048,
        num_workers: int = 4,
        force_regenerate: bool = False
    ) -> Path:
        """
        Generate and cache teacher logits for entire dataset.

        Args:
            dataset: HuggingFace dataset with tokenized inputs
            batch_size: Batch size for teacher inference
            max_length: Maximum sequence length
            num_workers: Number of dataloader workers
            force_regenerate: If True, regenerate even if cache exists

        Returns:
            Path to cached logits file
        """
        # Create cache filename based on dataset hash
        cache_file = self.cache_dir / "teacher_logits.h5"
        metadata_file = self.cache_dir / "metadata.json"

        # Check if cache exists
        if cache_file.exists() and not force_regenerate:
            logger.info(f"Using existing cached logits at {cache_file}")
            return cache_file

        logger.info(f"Generating teacher logits for {len(dataset)} samples...")
        logger.info(f"Cache directory: {self.cache_dir}")

        # Prepare dataset for batching
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Create HDF5 file for storing logits
        num_samples = len(dataset)

        # We'll store logits in chunks to handle variable sequence lengths
        with h5py.File(cache_file, 'w') as h5f:
            # Create datasets for storing top-k values and indices
            # Shape: (num_samples, max_length, top_k)
            logits_values_ds = h5f.create_dataset(
                'logits_values',
                shape=(num_samples, max_length, self.top_k),
                dtype='float16',
                chunks=(1, max_length, self.top_k),
                compression='gzip',
                compression_opts=4
            )

            logits_indices_ds = h5f.create_dataset(
                'logits_indices',
                shape=(num_samples, max_length, self.top_k),
                dtype='int32',
                chunks=(1, max_length, self.top_k),
                compression='gzip',
                compression_opts=4
            )

            # Store sequence lengths
            seq_lengths_ds = h5f.create_dataset(
                'seq_lengths',
                shape=(num_samples,),
                dtype='int32'
            )

            sample_idx = 0

            # Process in batches
            for i in tqdm(range(0, num_samples, batch_size), desc="Generating teacher logits"):
                batch_end = min(i + batch_size, num_samples)
                batch = dataset[i:batch_end]

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get teacher logits
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                logits = outputs.logits  # (batch_size, seq_len, vocab_size)

                # Process each sample in the batch
                for j in range(logits.size(0)):
                    sample_logits = logits[j]  # (seq_len, vocab_size)
                    seq_len = attention_mask[j].sum().item()

                    # Get top-k values and indices
                    topk_values, topk_indices = torch.topk(
                        sample_logits,
                        k=self.top_k,
                        dim=-1
                    )  # Both shape: (seq_len, top_k)

                    # Convert to CPU and store
                    topk_values_cpu = topk_values.cpu().numpy().astype(np.float16)
                    topk_indices_cpu = topk_indices.cpu().numpy().astype(np.int32)

                    # Pad to max_length if necessary
                    if seq_len < max_length:
                        pad_len = max_length - seq_len
                        topk_values_cpu = np.pad(
                            topk_values_cpu,
                            ((0, pad_len), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )
                        topk_indices_cpu = np.pad(
                            topk_indices_cpu,
                            ((0, pad_len), (0, 0)),
                            mode='constant',
                            constant_values=0
                        )

                    # Store in HDF5
                    logits_values_ds[sample_idx] = topk_values_cpu
                    logits_indices_ds[sample_idx] = topk_indices_cpu
                    seq_lengths_ds[sample_idx] = seq_len

                    sample_idx += 1

        # Save metadata
        vocab_size = self.teacher_model.config.vocab_size
        metadata = LogitsCacheMetadata(
            num_samples=num_samples,
            max_seq_length=max_length,
            top_k=self.top_k,
            vocab_size=vocab_size,
            teacher_model=self.teacher_model.config._name_or_path
        )

        with open(metadata_file, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2)

        logger.info(f"Teacher logits cached to {cache_file}")
        logger.info(f"Metadata saved to {metadata_file}")

        return cache_file


class CachedLogitsDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that provides access to cached teacher logits.

    Returns samples with both input data and corresponding teacher logits.
    """

    def __init__(
        self,
        cache_file: Path,
        original_dataset: Dataset,
        device: str = "cpu"
    ):
        """
        Initialize cached logits dataset.

        Args:
            cache_file: Path to HDF5 file with cached logits
            original_dataset: Original dataset with input_ids and attention_mask
            device: Device to load tensors on
        """
        self.cache_file = Path(cache_file)
        self.original_dataset = original_dataset
        self.device = device

        # Load metadata
        metadata_file = self.cache_file.parent / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
            self.metadata = LogitsCacheMetadata(**metadata_dict)

        # Open HDF5 file (keep it open for faster access)
        self.h5f = h5py.File(self.cache_file, 'r')
        self.logits_values = self.h5f['logits_values']
        self.logits_indices = self.h5f['logits_indices']
        self.seq_lengths = self.h5f['seq_lengths']

        logger.info(f"Loaded cached logits from {cache_file}")
        logger.info(f"Dataset size: {len(self)} samples")
        logger.info(f"Top-k: {self.metadata.top_k}")

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample with cached teacher logits.

        Returns:
            Dictionary with:
                - input_ids: Token IDs
                - attention_mask: Attention mask
                - labels: Labels for language modeling (shifted input_ids)
                - teacher_logits_values: Top-k logit values (seq_len, top_k)
                - teacher_logits_indices: Top-k logit indices (seq_len, top_k)
                - seq_length: Actual sequence length
        """
        # Get original sample
        sample = self.original_dataset[idx]

        # Get cached logits
        logits_values = torch.from_numpy(
            self.logits_values[idx][:]
        ).to(torch.float32)

        logits_indices = torch.from_numpy(
            self.logits_indices[idx][:]
        ).to(torch.long)

        seq_length = int(self.seq_lengths[idx])

        # Create labels (shifted input_ids for language modeling)
        input_ids = sample["input_ids"]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": sample["attention_mask"],
            "labels": labels,
            "teacher_logits_values": logits_values,
            "teacher_logits_indices": logits_indices,
            "seq_length": torch.tensor(seq_length, dtype=torch.long),
            "vocab_size": torch.tensor(self.metadata.vocab_size, dtype=torch.long)
        }

    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if hasattr(self, 'h5f'):
            self.h5f.close()


def reconstruct_teacher_logits(
    values: torch.Tensor,
    indices: torch.Tensor,
    vocab_size: int,
    fill_value: float = -1e4
) -> torch.Tensor:
    """
    Reconstruct full vocabulary logits from top-k cached values.

    Args:
        values: Top-k logit values (..., top_k)
        indices: Top-k logit indices (..., top_k)
        vocab_size: Size of vocabulary
        fill_value: Value to use for non-top-k positions

    Returns:
        Full logits tensor (..., vocab_size)
    """
    # Get shape
    *batch_dims, top_k = values.shape

    # Create tensor filled with small values
    full_logits = torch.full(
        (*batch_dims, vocab_size),
        fill_value,
        dtype=values.dtype,
        device=values.device
    )

    # Scatter top-k values to their positions
    full_logits.scatter_(
        dim=-1,
        index=indices,
        src=values
    )

    return full_logits
