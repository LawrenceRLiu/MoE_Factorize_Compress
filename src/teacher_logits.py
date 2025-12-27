"""
Teacher Logits Generation and Caching

Efficiently generates and caches teacher model logits for knowledge distillation.
Stores only top-k logits and their indices to save memory.

Features:
- Streaming dataset support for large pretraining datasets
- Multi-GPU parallel generation
- Progressive caching to HDF5
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import logging
from tqdm import tqdm
from dataclasses import dataclass
import json
import h5py

from transformers import AutoModelForCausalLM, PreTrainedModel
from datasets import Dataset, IterableDataset

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
    Supports streaming datasets and multi-GPU parallelization.
    """

    def __init__(
        self,
        teacher_model: PreTrainedModel,
        cache_dir: str,
        top_k: int = 64,
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
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @torch.no_grad()
    def generate_and_cache_logits(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = 4,
        max_length: int = 2048,
        force_regenerate: bool = False,
        streaming: bool = False,
        streaming_chunk_size: int = 1000,
        worker_id: int = 0,
        total_workers: int = 1
    ) -> Path:
        """
        Generate and cache teacher logits for entire dataset.

        Supports both regular datasets and streaming datasets.
        For multi-GPU parallelization, run multiple instances with different worker_id.

        Args:
            dataset: HuggingFace dataset with tokenized inputs
            batch_size: Batch size for teacher inference
            max_length: Maximum sequence length
            num_workers: Number of dataloader workers
            force_regenerate: If True, regenerate even if cache exists
            streaming: Whether dataset is streaming
            streaming_chunk_size: Number of samples to process in each chunk for streaming
            worker_id: Worker ID for multi-GPU parallelization (0-indexed)
            total_workers: Total number of workers for multi-GPU parallelization

        Returns:
            Path to cached logits file
        """
        # Create cache filename
        if total_workers > 1:
            cache_file = self.cache_dir / f"teacher_logits_worker{worker_id}.h5"
            metadata_file = self.cache_dir / f"metadata_worker{worker_id}.json"
        else:
            cache_file = self.cache_dir / "teacher_logits.h5"
            metadata_file = self.cache_dir / "metadata.json"

        # Check if cache exists
        if cache_file.exists() and not force_regenerate:
            logger.info(f"Using existing cached logits at {cache_file}")
            return cache_file

        logger.info(f"Generating teacher logits...")
        logger.info(f"Cache directory: {self.cache_dir}")
        if total_workers > 1:
            logger.info(f"Multi-GPU mode: Worker {worker_id}/{total_workers}")

        if streaming:
            return self._generate_streaming(
                dataset=dataset,
                cache_file=cache_file,
                metadata_file=metadata_file,
                batch_size=batch_size,
                max_length=max_length,
                chunk_size=streaming_chunk_size,
                worker_id=worker_id,
                total_workers=total_workers
            )
        else:
            return self._generate_regular(
                dataset=dataset,
                cache_file=cache_file,
                metadata_file=metadata_file,
                batch_size=batch_size,
                max_length=max_length,
                worker_id=worker_id,
                total_workers=total_workers
            )

    def _generate_regular(
        self,
        dataset: Dataset,
        cache_file: Path,
        metadata_file: Path,
        batch_size: int,
        max_length: int,
        worker_id: int,
        total_workers: int
    ) -> Path:
        """Generate logits for regular (non-streaming) dataset."""

        # For multi-worker: shard the dataset
        num_samples = len(dataset)
        if total_workers > 1:
            # Calculate this worker's shard
            samples_per_worker = num_samples // total_workers
            start_idx = worker_id * samples_per_worker
            if worker_id == total_workers - 1:
                # Last worker takes remainder
                end_idx = num_samples
            else:
                end_idx = start_idx + samples_per_worker

            dataset = dataset.select(range(start_idx, end_idx))
            num_samples = len(dataset)
            logger.info(f"Worker {worker_id}: processing samples {start_idx} to {end_idx}")

        logger.info(f"Processing {num_samples} samples...")

        # Prepare dataset for batching
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Create HDF5 file for storing logits
        with h5py.File(cache_file, 'w') as h5f:
            # Create datasets for storing top-k values and indices
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

            seq_lengths_ds = h5f.create_dataset(
                'seq_lengths',
                shape=(num_samples,),
                dtype='int32'
            )

            sample_idx = 0
            logger.info(f"logit indices ds shape: {logits_indices_ds.shape}")

            # Process in batches
            for i in tqdm(range(0, num_samples, batch_size), desc="Generating teacher logits"):
                batch_end = min(i + batch_size, num_samples)
                batch = dataset[i:batch_end]

                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                logger.info(f"Processing batch {i} to {batch_end}, input_ids shape: {input_ids.shape}")
                # Get teacher logits
                outputs = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                logging.info(f"Logits shape: {logits.shape}")

                # Process each sample in the batch
                for j in range(logits.size(0)):
                    sample_logits = logits[j]  # (max_length, vocab_size) - already padded
                    seq_len = attention_mask[j].sum().item()

                    # Get top-k values and indices for all positions (including padded)
                    # The logits are already at max_length due to input padding
                    topk_values, topk_indices = torch.topk(
                        sample_logits,
                        k=self.top_k,
                        dim=-1
                    )

                    # Convert to CPU and store
                    topk_values_cpu = topk_values.cpu().numpy().astype(np.float16)
                    topk_indices_cpu = topk_indices.cpu().numpy().astype(np.int32)

                    # Verify shape matches expected dimensions
                    assert topk_values_cpu.shape == (max_length, self.top_k), \
                        f"Shape mismatch: expected ({max_length}, {self.top_k}), got {topk_values_cpu.shape}"

                    # Store in HDF5
                    # Note: We store logits for all positions including padding
                    # The seq_length field indicates actual content length
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

    def _generate_streaming(
        self,
        dataset: IterableDataset,
        cache_file: Path,
        metadata_file: Path,
        batch_size: int,
        max_length: int,
        chunk_size: int,
        worker_id: int,
        total_workers: int
    ) -> Path:
        """Generate logits for streaming dataset."""

        logger.info(f"Streaming mode: processing in chunks of {chunk_size}")

        # For streaming datasets, we'll process in chunks and progressively write
        # Multi-worker support: shard by skipping samples
        if total_workers > 1:
            # For streaming, we use interleaving: worker_i takes samples i, i+total_workers, i+2*total_workers, ...
            logger.info(f"Worker {worker_id}: taking every {total_workers}-th sample starting from {worker_id}")

        with h5py.File(cache_file, 'w') as h5f:
            # Create resizable datasets
            logits_values_ds = h5f.create_dataset(
                'logits_values',
                shape=(0, max_length, self.top_k),
                maxshape=(None, max_length, self.top_k),
                dtype='float16',
                chunks=(1, max_length, self.top_k),
                compression='gzip',
                compression_opts=4
            )

            logits_indices_ds = h5f.create_dataset(
                'logits_indices',
                shape=(0, max_length, self.top_k),
                maxshape=(None, max_length, self.top_k),
                dtype='int32',
                chunks=(1, max_length, self.top_k),
                compression='gzip',
                compression_opts=4
            )

            seq_lengths_ds = h5f.create_dataset(
                'seq_lengths',
                shape=(0,),
                maxshape=(None,),
                dtype='int32'
            )

            sample_idx = 0
            chunk_buffer = []
            total_samples_processed = 0

            # For streaming datasets, we MUST have a limit to avoid infinite iteration
            # This is critical for datasets like FineWeb (13T tokens)
            max_samples_limit = chunk_size * 1000  # Default limit if not specified elsewhere
            logger.warning(
                f"Streaming dataset detected. Will process samples until stopped. "
                f"IMPORTANT: Set max_samples in config to avoid processing entire dataset!"
            )

            # Iterate through streaming dataset
            logger.info("Processing streaming dataset...")
            pbar = tqdm(desc="Processing samples", unit="samples")

            for idx, sample in enumerate(dataset):
                # Multi-worker sharding: skip samples not for this worker
                if total_workers > 1 and (idx % total_workers) != worker_id:
                    continue

                total_samples_processed += 1

                chunk_buffer.append(sample)

                # Process when buffer reaches batch_size
                if len(chunk_buffer) >= batch_size:
                    num_processed = self._process_batch_streaming(
                        chunk_buffer[:batch_size],
                        h5f, logits_values_ds, logits_indices_ds,
                        seq_lengths_ds, sample_idx, max_length
                    )
                    sample_idx += num_processed
                    chunk_buffer = chunk_buffer[batch_size:]
                    pbar.update(num_processed)

            # Process remaining samples in buffer
            if chunk_buffer:
                num_processed = self._process_batch_streaming(
                    chunk_buffer,
                    h5f, logits_values_ds, logits_indices_ds,
                    seq_lengths_ds, sample_idx, max_length
                )
                sample_idx += num_processed
                pbar.update(num_processed)

            pbar.close()

        # Save metadata
        vocab_size = self.teacher_model.config.vocab_size
        metadata = LogitsCacheMetadata(
            num_samples=sample_idx,
            max_seq_length=max_length,
            top_k=self.top_k,
            vocab_size=vocab_size,
            teacher_model=self.teacher_model.config._name_or_path
        )

        with open(metadata_file, 'w') as f:
            json.dump(metadata.__dict__, f, indent=2)

        logger.info(f"Processed {sample_idx} samples in streaming mode")
        logger.info(f"Teacher logits cached to {cache_file}")
        logger.info(f"Metadata saved to {metadata_file}")

        return cache_file

    def _process_batch_streaming(
        self,
        batch_samples: List[Dict],
        h5f,
        logits_values_ds,
        logits_indices_ds,
        seq_lengths_ds,
        start_idx: int,
        max_length: int
    ) -> int:
        """Process a batch of samples for streaming dataset."""
        batch_size = len(batch_samples)

        # Stack inputs
        input_ids = torch.stack([s["input_ids"] for s in batch_samples]).to(self.device)
        attention_mask = torch.stack([s["attention_mask"] for s in batch_samples]).to(self.device)

        # Get teacher logits
        outputs = self.teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Resize datasets if needed
        new_size = start_idx + batch_size
        if logits_values_ds.shape[0] < new_size:
            logits_values_ds.resize((new_size, max_length, self.top_k))
            logits_indices_ds.resize((new_size, max_length, self.top_k))
            seq_lengths_ds.resize((new_size,))

        # Process each sample in the batch
        for j in range(batch_size):
            sample_logits = logits[j]  # (max_length, vocab_size) - already padded
            seq_len = attention_mask[j].sum().item()

            # Get top-k values and indices for all positions (including padded)
            # The logits are already at max_length due to input padding
            topk_values, topk_indices = torch.topk(
                sample_logits,
                k=self.top_k,
                dim=-1
            )

            # Convert to CPU and store
            topk_values_cpu = topk_values.cpu().numpy().astype(np.float16)
            topk_indices_cpu = topk_indices.cpu().numpy().astype(np.int32)

            # Verify shape matches expected dimensions
            assert topk_values_cpu.shape == (max_length, self.top_k), \
                f"Shape mismatch: expected ({max_length}, {self.top_k}), got {topk_values_cpu.shape}"

            # Store in HDF5
            # Note: We store logits for all positions including padding
            # The seq_length field indicates actual content length
            sample_idx = start_idx + j
            logits_values_ds[sample_idx] = topk_values_cpu
            logits_indices_ds[sample_idx] = topk_indices_cpu
            seq_lengths_ds[sample_idx] = seq_len

        # Flush to disk
        h5f.flush()

        return batch_size


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

        # Mask padding positions in labels with -100 so they're ignored in loss computation
        # This ensures CE loss only computes on actual tokens, not padding
        attention_mask = sample["attention_mask"]
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
