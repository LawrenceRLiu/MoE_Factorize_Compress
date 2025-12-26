#!/usr/bin/env python3
"""
Generate and cache teacher model logits using Hydra configuration.

This script can be run independently to pre-generate teacher logits
before starting the distillation training. This is useful for:
- Separating teacher inference from student training
- Reusing cached logits across multiple training runs
- Reducing GPU memory requirements during training

Usage:
    python scripts/generate_teacher_logits.py

Configuration is loaded from conf/config.yaml and conf/distillation/default.yaml
You can override settings via command line:
    python scripts/generate_teacher_logits.py \
        distillation.dataset.name=c4 \
        distillation.teacher.generation.top_k=32 \
        distillation.teacher.generation.num_workers=4
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from src.distillation_utils import prepare_dataset, generate_teacher_logits
from src.utils import set_seed, log_gpu_memory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    """
    Generate and cache teacher logits.

    Args:
        config: Hydra configuration
    """
    logger.info("="*80)
    logger.info("Teacher Logits Generation")
    logger.info("="*80)

    # Use distillation config (unified config approach)
    dist_cfg = config.distillation

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(f"Teacher: {OmegaConf.to_yaml(dist_cfg.teacher)}")
    logger.info(f"Dataset: {OmegaConf.to_yaml(dist_cfg.dataset)}")

    # Log GPU information
    logger.info("\nGPU Information:")
    log_gpu_memory()

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"\nRandom seed set to: {seed}")

    # Resolve teacher model path
    teacher_model_path = dist_cfg.teacher.model_path
    if teacher_model_path is None:
        teacher_model_path = config.model.name
        logger.info(f"Using model.name as teacher: {teacher_model_path}")

    # Load tokenizer
    logger.info(f"\nLoading tokenizer from {teacher_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        teacher_model_path,
        trust_remote_code=config.model.trust_remote_code
    )
    logger.info("Tokenizer loaded successfully")

    # Prepare dataset
    logger.info("\n" + "="*80)
    logger.info("Preparing dataset...")
    logger.info("="*80)

    dataset = prepare_dataset(
        dataset_name=dist_cfg.dataset.name,
        dataset_config=dist_cfg.dataset.config_name,
        split=dist_cfg.dataset.split,
        tokenizer=tokenizer,
        max_length=dist_cfg.dataset.max_length,
        text_column=dist_cfg.dataset.text_column,
        max_samples=dist_cfg.dataset.max_samples,
        streaming=dist_cfg.dataset.streaming,
        num_proc=dist_cfg.dataset.get('num_proc', 4)
    )

    if dist_cfg.dataset.streaming:
        logger.info(f"\nDataset prepared (streaming mode)")
        logger.info(f"Chunk size: {dist_cfg.dataset.get('streaming_chunk_size', 1000)}")
    else:
        logger.info(f"\nDataset prepared: {len(dataset)} samples")
    logger.info(f"Sequence length: {dist_cfg.dataset.max_length}")

    # Generate teacher logits
    logger.info("\n" + "="*80)
    logger.info("Generating teacher logits...")
    logger.info("="*80)

    gen_cfg = dist_cfg.teacher.generation
    cache_file = generate_teacher_logits(
        teacher_model_path=teacher_model_path,
        dataset=dataset,
        cache_dir=dist_cfg.teacher.cache_dir,
        top_k=gen_cfg.top_k,
        batch_size=gen_cfg.batch_size,
        device=gen_cfg.device,
        force_regenerate=gen_cfg.force_regenerate,
        # New parameters
        num_workers=gen_cfg.get('num_workers', 1),
        devices=gen_cfg.get('devices', None),
        worker_id=gen_cfg.get('worker_id', 0),
        total_workers=gen_cfg.get('total_workers', 1),
        streaming=dist_cfg.dataset.streaming,
        streaming_chunk_size=dist_cfg.dataset.get('streaming_chunk_size', 1000)
    )

    logger.info("\n" + "="*80)
    logger.info("Teacher logits generation complete!")
    logger.info(f"Cache saved to: {cache_file}")
    logger.info("="*80)

    # Log cache information
    cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
    logger.info(f"\nCache file size: {cache_size_mb:.2f} MB")
    if not dist_cfg.dataset.streaming:
        logger.info(f"Number of samples: {len(dataset)}")
    logger.info(f"Top-k: {gen_cfg.top_k}")
    logger.info(f"Max sequence length: {dist_cfg.dataset.max_length}")

    # Calculate storage efficiency (if not streaming)
    if not dist_cfg.dataset.streaming:
        vocab_size = len(tokenizer)
        original_size_mb = (
            len(dataset) *
            dist_cfg.dataset.max_length *
            vocab_size *
            2  # float16
        ) / (1024 * 1024)
        compression_ratio = original_size_mb / cache_size_mb

        logger.info(f"\nStorage efficiency:")
        logger.info(f"  Full logits would be: {original_size_mb:.2f} MB")
        logger.info(f"  Top-k cached logits: {cache_size_mb:.2f} MB")
        logger.info(f"  Compression ratio: {compression_ratio:.1f}x")

    # Final GPU memory info
    logger.info("\nFinal GPU memory state:")
    log_gpu_memory()


if __name__ == "__main__":
    main()
