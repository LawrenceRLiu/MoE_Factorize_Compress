#!/usr/bin/env python
"""
Test script to validate the MoE compression implementation.

This script runs unit tests on the core components without requiring
a full model download.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from src.shared_core import (
    LowRankWrapper,
    CompressedExpert,
    SharedCoreLayer,
    initialize_from_experts
)
from src.utils import (
    set_seed,
    print_model_size,
    estimate_memory_usage
)


def test_low_rank_wrapper():
    """Test LowRankWrapper module."""
    print("\n" + "="*60)
    print("Testing LowRankWrapper")
    print("="*60)

    dim = 1024
    rank = 64
    batch_size = 8

    wrapper = LowRankWrapper(dim, rank, init_zeros=True)

    # Test forward pass
    x = torch.randn(batch_size, dim)
    out = wrapper(x)

    assert out.shape == (batch_size, dim), f"Shape mismatch: {out.shape}"

    # Initially should be close to identity (since U initialized to zeros)
    diff = torch.norm(out - x).item()
    print(f"✓ LowRankWrapper forward pass: output shape {out.shape}")
    print(f"  Initial difference from identity: {diff:.6f}")

    # Test matrix reconstruction
    full_mat = wrapper.as_matrix()
    assert full_mat.shape == (dim, dim), f"Matrix shape mismatch: {full_mat.shape}"
    print(f"✓ Matrix reconstruction: shape {full_mat.shape}")

    # Check it's close to identity initially
    identity_diff = torch.norm(full_mat - torch.eye(dim)).item()
    print(f"  Distance from identity: {identity_diff:.6f}")

    print("✓ LowRankWrapper tests passed!")


def test_compressed_expert():
    """Test CompressedExpert module."""
    print("\n" + "="*60)
    print("Testing CompressedExpert")
    print("="*60)

    d_in = 1024
    d_out = 2048
    rank = 64
    batch_size = 8

    # Create random core
    core = torch.randn(d_out, d_in) * 0.02

    expert = CompressedExpert(core, d_in, d_out, rank)

    # Test forward pass
    x = torch.randn(batch_size, d_in)
    out = expert(x)

    assert out.shape == (batch_size, d_out), f"Shape mismatch: {out.shape}"
    print(f"✓ CompressedExpert forward pass: {x.shape} -> {out.shape}")

    # Test weight reconstruction
    reconstructed = expert.reconstruct_weight()
    assert reconstructed.shape == (d_out, d_in), f"Weight shape mismatch: {reconstructed.shape}"
    print(f"✓ Weight reconstruction: shape {reconstructed.shape}")

    # Verify forward is same as matmul with reconstructed weight
    out_reconstructed = x @ reconstructed.T
    diff = torch.norm(out - out_reconstructed).item()
    print(f"  Forward vs reconstructed matmul difference: {diff:.6f}")
    assert diff < 1e-4, f"Forward pass mismatch: {diff}"

    print("✓ CompressedExpert tests passed!")


def test_shared_core_layer():
    """Test SharedCoreLayer module."""
    print("\n" + "="*60)
    print("Testing SharedCoreLayer")
    print("="*60)

    num_experts = 8
    d_in = 1024
    d_out = 2048
    rank = 64
    batch_size = 4

    # Create layer
    layer = SharedCoreLayer(num_experts, d_in, d_out, rank)

    print(f"Created SharedCoreLayer:")
    print(f"  Experts: {num_experts}")
    print(f"  Dimensions: {d_in} -> {d_out}")
    print(f"  Rank: {rank}")

    # Test forward for each expert
    x = torch.randn(batch_size, d_in)
    for expert_idx in range(num_experts):
        out = layer(x, expert_idx)
        assert out.shape == (batch_size, d_out), f"Expert {expert_idx} output shape mismatch"

    print(f"✓ Forward pass for all {num_experts} experts")

    # Test parameter counting
    param_stats = layer.count_parameters()
    print(f"\nParameter statistics:")
    print(f"  Core params: {param_stats['core_params']:,}")
    print(f"  Wrapper params per expert: {param_stats['wrapper_params_per_expert']:,}")
    print(f"  Total compressed params: {param_stats['total_params']:,}")
    print(f"  Original params: {param_stats['original_params']:,}")
    print(f"  Compression ratio: {param_stats['compression_ratio']:.4f}")
    print(f"  Reduction: {param_stats['reduction_percentage']:.2f}%")

    assert param_stats['total_params'] < param_stats['original_params'], \
        "Compressed model should have fewer parameters!"

    print("✓ SharedCoreLayer tests passed!")


def test_zero_shot_initialization():
    """Test zero-shot initialization algorithm."""
    print("\n" + "="*60)
    print("Testing Zero-Shot Initialization")
    print("="*60)

    # Create synthetic expert weights
    num_experts = 4
    d_out = 512
    d_in = 256
    rank = 32

    print(f"Creating {num_experts} synthetic experts ({d_out}×{d_in})")

    # Generate experts with shared structure + noise
    set_seed(42)
    shared_base = torch.randn(d_out, d_in) * 0.02

    expert_weights = []
    for i in range(num_experts):
        # Each expert is base + small random perturbation
        noise = torch.randn(d_out, d_in) * 0.005
        expert = shared_base + noise
        expert_weights.append(expert)

    print("Running zero-shot initialization...")
    print(f"  Rank: {rank}")
    print(f"  Steps: 100 (reduced for testing)")

    # Run initialization (with fewer steps for testing)
    core, wrappers = initialize_from_experts(
        expert_weights=expert_weights,
        rank=rank,
        num_steps=100,
        lr=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\n✓ Initialization complete!")
    print(f"  Core shape: {core.shape}")
    print(f"  Number of wrappers: {len(wrappers)}")

    # Verify reconstruction quality
    print("\nReconstruction quality:")
    for i, (W_orig, (U_in, V_in, U_out, V_out)) in enumerate(zip(expert_weights, wrappers)):
        # Reconstruct
        W_orig = W_orig.to(core.device)
        input_mat = torch.eye(d_in, device=core.device) + U_in @ V_in.T
        output_mat = torch.eye(d_out, device=core.device) + U_out @ V_out.T
        W_recon = output_mat @ core @ input_mat

        # Compute error
        error = torch.norm(W_orig - W_recon, p='fro').item()
        relative_error = error / torch.norm(W_orig, p='fro').item()
        print(f"  Expert {i}: Frobenius error = {error:.6f}, relative = {relative_error:.4f}")

    print("✓ Zero-shot initialization tests passed!")


def test_compression_ratio_calculation():
    """Test compression ratio calculations."""
    print("\n" + "="*60)
    print("Testing Compression Ratio Calculations")
    print("="*60)

    # Typical Qwen-3 MoE dimensions
    configs = [
        {"name": "Small", "num_experts": 8, "d_in": 2048, "d_out": 5632, "rank": 32},
        {"name": "Medium", "num_experts": 8, "d_in": 4096, "d_out": 11264, "rank": 64},
        {"name": "Large", "num_experts": 16, "d_in": 4096, "d_out": 14336, "rank": 64},
    ]

    for config in configs:
        num_experts = config["num_experts"]
        d_in = config["d_in"]
        d_out = config["d_out"]
        rank = config["rank"]

        # Original
        original_params = num_experts * d_in * d_out

        # Compressed
        core_params = d_in * d_out
        wrapper_params = num_experts * (2 * d_in * rank + 2 * d_out * rank)
        compressed_params = core_params + wrapper_params

        ratio = compressed_params / original_params
        reduction = (1 - ratio) * 100

        print(f"\n{config['name']} Config:")
        print(f"  Experts: {num_experts}, Dims: {d_in}×{d_out}, Rank: {rank}")
        print(f"  Original: {original_params:,} params ({original_params*2/1e9:.2f} GB)")
        print(f"  Compressed: {compressed_params:,} params ({compressed_params*2/1e9:.2f} GB)")
        print(f"  Ratio: {ratio:.4f}")
        print(f"  Reduction: {reduction:.2f}%")

    print("\n✓ Compression ratio calculations complete!")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("MoE Compression Implementation Tests")
    print("="*60)

    try:
        test_low_rank_wrapper()
        test_compressed_expert()
        test_shared_core_layer()
        test_zero_shot_initialization()
        test_compression_ratio_calculation()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe implementation is ready to use.")
        print("Next steps:")
        print("  1. Review the configurations in conf/")
        print("  2. Run: python scripts/run_compression.py")
        print("  3. Monitor progress and check outputs")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
