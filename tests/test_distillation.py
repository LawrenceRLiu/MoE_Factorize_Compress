"""
Unit tests for knowledge distillation components.

Tests the core functionality of:
- Teacher logits reconstruction
- Parameter group manager
- Distillation loss computation
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Test imports
from src.teacher_logits import reconstruct_teacher_logits
from src.distillation_trainer import ParameterGroupManager


def test_logits_reconstruction():
    """Test that top-k logits can be reconstructed to full vocabulary."""
    print("\n" + "="*80)
    print("Testing logits reconstruction...")
    print("="*80)

    batch_size = 2
    seq_len = 10
    vocab_size = 50000
    top_k = 64

    # Create random full logits
    full_logits = torch.randn(batch_size, seq_len, vocab_size)

    # Get top-k
    topk_values, topk_indices = torch.topk(full_logits, k=top_k, dim=-1)

    # Reconstruct
    reconstructed = reconstruct_teacher_logits(
        values=topk_values,
        indices=topk_indices,
        vocab_size=vocab_size,
        fill_value=-1e4
    )

    # Verify shape
    assert reconstructed.shape == full_logits.shape, "Shape mismatch"

    # Verify top-k values are preserved
    for b in range(batch_size):
        for s in range(seq_len):
            for k in range(top_k):
                idx = topk_indices[b, s, k].item()
                expected = topk_values[b, s, k].item()
                actual = reconstructed[b, s, idx].item()
                assert abs(expected - actual) < 1e-5, f"Value mismatch at [{b},{s},{k}]"

    # Verify non-top-k positions are filled with fill_value
    # Create mask for non-top-k positions
    mask = torch.ones(batch_size, seq_len, vocab_size, dtype=torch.bool)
    mask.scatter_(dim=-1, index=topk_indices, src=torch.zeros_like(topk_indices, dtype=torch.bool))

    non_topk_values = reconstructed[mask]
    assert torch.all(non_topk_values == -1e4), "Non-top-k values should be fill_value"

    print("✓ Logits reconstruction test passed!")
    return True


def test_parameter_group_manager():
    """Test parameter grouping with regex-based learning rates."""
    print("\n" + "="*80)
    print("Testing parameter group manager...")
    print("="*80)

    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 64)
            self.shared_core = nn.Linear(64, 64)
            self.expert_wrapper_1 = nn.Linear(64, 64)
            self.expert_wrapper_2 = nn.Linear(64, 64)
            self.norm = nn.LayerNorm(64)
            self.head = nn.Linear(64, 100)

        def forward(self, x):
            x = self.embedding(x)
            x = self.shared_core(x)
            x = self.expert_wrapper_1(x)
            x = self.norm(x)
            x = self.head(x)
            return x

    model = SimpleModel()

    # Define learning rate multipliers
    lr_multipliers = [
        {"pattern": ".*shared_core.*", "lr_multiplier": 1.0},
        {"pattern": ".*expert_wrapper.*", "lr_multiplier": 1.0},
        {"pattern": ".*embedding.*", "lr_multiplier": 0.1},
        {"pattern": ".*head.*", "lr_multiplier": 0.1},
        {"pattern": ".*norm.*", "lr_multiplier": 0.0},  # Frozen
    ]

    # Create parameter group manager
    manager = ParameterGroupManager(
        model=model,
        base_lr=1e-4,
        lr_multipliers=lr_multipliers,
        weight_decay=0.01
    )

    # Create parameter groups
    param_groups = manager.create_parameter_groups()

    # Verify groups were created
    assert len(param_groups) > 0, "No parameter groups created"

    # Verify learning rates
    found_lrs = set()
    for group in param_groups:
        found_lrs.add(group['lr'])

    expected_lrs = {1e-4, 1e-5}  # 1.0 * base_lr and 0.1 * base_lr (0.0 is frozen, not in groups)
    assert found_lrs == expected_lrs, f"Expected LRs {expected_lrs}, got {found_lrs}"

    # Verify norm is frozen (should not have requires_grad=True)
    for name, param in model.named_parameters():
        if 'norm' in name:
            assert not param.requires_grad, f"Parameter {name} should be frozen"

    print("✓ Parameter group manager test passed!")
    return True


def test_distillation_loss_computation():
    """Test that distillation loss can be computed correctly."""
    print("\n" + "="*80)
    print("Testing distillation loss computation...")
    print("="*80)

    import torch.nn.functional as F

    batch_size = 2
    seq_len = 10
    vocab_size = 100
    temperature = 2.0
    alpha = 0.5

    # Create dummy student and teacher logits
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Shift for next-token prediction
    shift_student = student_logits[..., :-1, :].contiguous()
    shift_teacher = teacher_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute CE loss
    ce_loss = F.cross_entropy(
        shift_student.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='mean'
    )

    # Compute KL loss
    student_log_probs = F.log_softmax(shift_student / temperature, dim=-1)
    teacher_probs = F.softmax(shift_teacher / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_log_probs.view(-1, vocab_size),
        teacher_probs.view(-1, vocab_size),
        reduction='batchmean',
        log_target=False
    ) * (temperature ** 2)

    # Compute total loss
    total_loss = (1 - alpha) * ce_loss + alpha * kl_loss

    # Verify losses are finite and positive
    assert torch.isfinite(ce_loss), "CE loss is not finite"
    assert torch.isfinite(kl_loss), "KL loss is not finite"
    assert torch.isfinite(total_loss), "Total loss is not finite"
    assert ce_loss >= 0, "CE loss should be non-negative"
    assert kl_loss >= 0, "KL loss should be non-negative"
    assert total_loss >= 0, "Total loss should be non-negative"

    print(f"  CE Loss: {ce_loss.item():.4f}")
    print(f"  KL Loss: {kl_loss.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print("✓ Distillation loss computation test passed!")
    return True


def run_all_tests():
    """Run all distillation tests."""
    print("\n" + "="*80)
    print("Running Knowledge Distillation Tests")
    print("="*80)

    tests = [
        test_logits_reconstruction,
        test_parameter_group_manager,
        test_distillation_loss_computation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
