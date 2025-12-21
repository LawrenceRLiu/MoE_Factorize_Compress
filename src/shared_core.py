"""
Shared Core Compression Module

Implements the core decomposition: W_e ≈ (I + U_out @ V_out^T) @ C @ (I + U_in @ V_in^T)
where C is a shared core and U, V are low-rank adapters per expert.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import math


class LowRankWrapper(nn.Module):
    """
    Low-rank wrapper implementing (I + U @ V^T) transformation.

    Acts as a residual rotation to the shared core, allowing each expert
    to shift the input/output manifold without learning a full dense matrix.

    Args:
        dim: Dimension of the input/output space
        rank: Rank of the low-rank factorization
        init_zeros: If True, initialize U to zeros (LoRA-style). Otherwise random.
    """

    def __init__(self, dim: int, rank: int, init_zeros: bool = True):
        super().__init__()
        self.dim = dim
        self.rank = rank

        # U and V matrices for low-rank factorization
        self.U = nn.Parameter(torch.zeros(dim, rank) if init_zeros else torch.randn(dim, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(dim, rank) / math.sqrt(rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply (I + U @ V^T) @ x transformation.

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Transformed tensor of shape [..., dim]
        """
        # Identity path
        identity = x

        # Low-rank path: (U @ V^T) @ x = U @ (V^T @ x)
        # This is more efficient than materializing U @ V^T
        low_rank = x @ self.V  # [..., rank]
        low_rank = low_rank @ self.U.T  # [..., dim]

        return identity + low_rank

    def as_matrix(self) -> torch.Tensor:
        """Return the full (I + U @ V^T) matrix for analysis."""
        return torch.eye(self.dim, device=self.U.device) + self.U @ self.V.T


class CompressedExpert(nn.Module):
    """
    Single compressed expert using shared core + low-rank wrappers.

    Implements: output = (I + U_out @ V_out^T) @ core @ (I + U_in @ V_in^T) @ input

    Args:
        core: Shared core weight matrix (not owned by this module)
        d_in: Input dimension
        d_out: Output dimension
        rank: Rank for low-rank adapters
    """

    def __init__(self, core: torch.Tensor, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # Register core as buffer (shared, not trained per-expert)
        self.register_buffer('core', core)

        # Input and output low-rank wrappers
        self.input_wrapper = LowRankWrapper(d_in, rank, init_zeros=True)
        self.output_wrapper = LowRankWrapper(d_out, rank, init_zeros=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through compressed expert.

        Args:
            x: Input tensor of shape [..., d_in]

        Returns:
            Output tensor of shape [..., d_out]
        """
        # Apply input wrapper: (I + U_in @ V_in^T) @ x
        x = self.input_wrapper(x)

        # Apply shared core: C @ x
        x = x @ self.core.T

        # Apply output wrapper: (I + U_out @ V_out^T) @ x
        x = self.output_wrapper(x)

        return x

    def reconstruct_weight(self) -> torch.Tensor:
        """
        Reconstruct the full weight matrix for analysis or export.

        Returns:
            W_reconstructed ≈ (I + U_out @ V_out^T) @ C @ (I + U_in @ V_in^T)
        """
        # Get wrapper matrices
        input_mat = self.input_wrapper.as_matrix()  # [d_in, d_in]
        output_mat = self.output_wrapper.as_matrix()  # [d_out, d_out]

        # Reconstruct: output_mat @ core @ input_mat
        return output_mat @ self.core @ input_mat


class SharedCoreLayer(nn.Module):
    """
    Layer of compressed experts sharing a single core.

    For a given layer and projection (e.g., gate_proj, up_proj, down_proj),
    all experts share one core matrix but have individual low-rank wrappers.

    Args:
        num_experts: Number of experts in the layer
        d_in: Input dimension
        d_out: Output dimension
        rank: Rank for low-rank adapters
        init_core: Optional initial core matrix. If None, will be initialized later.
    """

    def __init__(
        self,
        num_experts: int,
        d_in: int,
        d_out: int,
        rank: int,
        init_core: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_experts = num_experts
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank

        # Shared core (trainable)
        if init_core is not None:
            assert init_core.shape == (d_out, d_in), f"Core shape mismatch: {init_core.shape} vs ({d_out}, {d_in})"
            self.core = nn.Parameter(init_core.clone())
        else:
            # Initialize with xavier uniform (will typically be overwritten by mean of experts)
            self.core = nn.Parameter(torch.empty(d_out, d_in))
            nn.init.xavier_uniform_(self.core)

        # Create compressed experts (they will share the core buffer)
        self.experts = nn.ModuleList([
            CompressedExpert(self.core, d_in, d_out, rank)
            for _ in range(num_experts)
        ])

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        Forward pass through a specific expert.

        Args:
            hidden_states: Input tensor
            expert_idx: Index of expert to use

        Returns:
            Output from the selected expert
        """
        return self.experts[expert_idx](hidden_states)

    def get_expert(self, idx: int) -> CompressedExpert:
        """Get a specific compressed expert."""
        return self.experts[idx]

    def count_parameters(self) -> Dict[str, int]:
        """
        Count parameters in the compressed representation.

        Returns:
            Dictionary with parameter counts
        """
        # Core parameters (shared across all experts)
        core_params = self.d_in * self.d_out

        # Wrapper parameters per expert: 2 * (U + V) = 2 * (dim * rank + dim * rank)
        input_wrapper_per_expert = 2 * self.d_in * self.rank
        output_wrapper_per_expert = 2 * self.d_out * self.rank
        wrapper_params_per_expert = input_wrapper_per_expert + output_wrapper_per_expert

        # Total wrapper parameters
        total_wrapper_params = wrapper_params_per_expert * self.num_experts

        # Total parameters
        total_params = core_params + total_wrapper_params

        # Original uncompressed parameters (for comparison)
        original_params = self.num_experts * self.d_in * self.d_out

        return {
            "core_params": core_params,
            "wrapper_params_per_expert": wrapper_params_per_expert,
            "total_wrapper_params": total_wrapper_params,
            "total_params": total_params,
            "original_params": original_params,
            "compression_ratio": total_params / original_params,
            "reduction_percentage": (1 - total_params / original_params) * 100
        }


def initialize_from_experts(
    expert_weights: List[torch.Tensor],
    rank: int,
    num_steps: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda"
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Zero-shot initialization of shared core + wrappers from original expert weights.

    Algorithm:
    1. Initialize core as mean of all expert weights
    2. Initialize wrappers: U=0, V~N(0,1)
    3. Optimize to minimize ||W_e - W_hat_e||_F^2 using Adam

    Args:
        expert_weights: List of expert weight tensors [num_experts, d_out, d_in]
        rank: Rank for low-rank adapters
        num_steps: Number of optimization steps
        lr: Learning rate for Adam
        device: Device to run optimization on

    Returns:
        core: Optimized shared core matrix
        wrappers: List of (U_in, V_in, U_out, V_out) tuples per expert
    """
    expert_weights = [w.to(device) for w in expert_weights]
    num_experts = len(expert_weights)
    d_out, d_in = expert_weights[0].shape

    # Step 1: Initialize core as mean
    core = torch.stack(expert_weights).mean(dim=0).clone().requires_grad_(True)

    # Step 2: Initialize wrappers (LoRA-style)
    wrappers = []
    for _ in range(num_experts):
        U_in = torch.zeros(d_in, rank, device=device, requires_grad=True)
        V_in = torch.randn(d_in, rank, device=device, requires_grad=True) / math.sqrt(rank)
        U_out = torch.zeros(d_out, rank, device=device, requires_grad=True)
        V_out = torch.randn(d_out, rank, device=device, requires_grad=True) / math.sqrt(rank)
        wrappers.append((U_in, V_in, U_out, V_out))

    # Collect all parameters for optimizer
    params = [core]
    for U_in, V_in, U_out, V_out in wrappers:
        params.extend([U_in, V_in, U_out, V_out])

    # Step 3: Optimize with Adam
    optimizer = torch.optim.Adam(params, lr=lr)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Compute reconstruction loss for all experts
        total_loss = 0.0
        for e, (W_e, (U_in, V_in, U_out, V_out)) in enumerate(zip(expert_weights, wrappers)):
            # Reconstruct: W_hat_e = (I + U_out @ V_out^T) @ core @ (I + U_in @ V_in^T)
            # Efficiently compute without materializing full matrices

            # Input wrapper application: (I + U_in @ V_in^T)
            input_wrapped = torch.eye(d_in, device=device) + U_in @ V_in.T

            # Core @ input_wrapped
            temp = core @ input_wrapped

            # Output wrapper: (I + U_out @ V_out^T) @ temp
            output_wrapped = torch.eye(d_out, device=device) + U_out @ V_out.T
            W_hat_e = output_wrapped @ temp

            # L2 reconstruction loss
            loss = torch.norm(W_e - W_hat_e, p='fro') ** 2
            total_loss += loss

        # Backprop and step
        total_loss.backward()
        optimizer.step()

        # Logging
        if step % 100 == 0 or step == num_steps - 1:
            avg_loss = total_loss.item() / num_experts
            print(f"Step {step}/{num_steps}, Avg Reconstruction Loss: {avg_loss:.6f}")

    # Detach and return
    core_final = core.detach()
    wrappers_final = [
        (U_in.detach(), V_in.detach(), U_out.detach(), V_out.detach())
        for U_in, V_in, U_out, V_out in wrappers
    ]

    return core_final, wrappers_final
