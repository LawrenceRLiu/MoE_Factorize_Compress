"""
Shared Core Compression Module

Implements the core decomposition: W_e ≈ (I + U_out @ V_out^T) @ C @ (I + U_in @ V_in^T)
where C is a shared core and U, V are low-rank adapters per expert.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import math
import logging


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
        # print(f"self.U dtype: {self.U.dtype}, self.V dtype: {self.V.dtype}")
        return torch.eye(self.dim, device=self.U.device, dtype=self.U.dtype) + self.U @ self.V.T


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

    def __init__(self, d_in: int, d_out: int, rank: int,
                 bias: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank


        # Input and output low-rank wrappers
        self.input_wrapper = LowRankWrapper(d_in, rank, init_zeros=True)
        self.output_wrapper = LowRankWrapper(d_out, rank, init_zeros=True)

        if bias:
            print("using bias in compressed expert")
            self.bias = nn.Parameter(torch.zeros(d_out))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through compressed expert.

        Args:
            x: Input tensor of shape [..., d_in]
            core: Shared core weight matrix of shape [d_out, d_in]

        Returns:
            Output tensor of shape [..., d_out]
        """
        # Apply input wrapper: (I + U_in @ V_in^T) @ x
        x = self.input_wrapper(x)

        # Apply shared core: C @ x
        x = x @ core.T  # [..., d_out]

        # Apply output wrapper: (I + U_out @ V_out^T) @ x
        x = self.output_wrapper(x)
        if self.bias is not None:
            x = x + self.bias
        return x
    
    def reconstruct(self, core: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the full weight matrix of this expert.

        Args:
            core: Shared core weight matrix of shape [d_out, d_in]

        Returns:
            Reconstructed weight matrix of shape [d_out, d_in]
        """
        # Get full wrapper matrices
        
        input_matrix = self.input_wrapper.as_matrix()  # [d_in, d_in]
        output_matrix = self.output_wrapper.as_matrix()  # [d_out, d_out]
        
        # print(f"Core dtype: {core.dtype}, input_matrix dtype: {input_matrix.dtype}, output_matrix dtype: {output_matrix.dtype}")
        # Full reconstruction: W_hat = (I + U_out @ V_out^T) @ C @ (I + U_in @ V_in^T)
        W_hat = output_matrix @ core @ input_matrix  # [d_out, d_in]
        return W_hat


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
        init_core: Optional[torch.Tensor] = None,
        bias: bool = True
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
        # assert not bias, "bias is true"
        self.experts = nn.ModuleList([
            CompressedExpert(d_in, d_out, rank, bias=bias)
            for _ in range(num_experts)
        ])
        self.has_bias = bias

    def forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        Forward pass through a specific expert.

        Args:
            hidden_states: Input tensor
            expert_idx: Index of expert to use

        Returns:
            Output from the selected expert
        """
        return self.experts[expert_idx](hidden_states, self.core)

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
        wrapper_params_per_expert = input_wrapper_per_expert + output_wrapper_per_expert + self.d_out * self.has_bias  # include bias

        # Total wrapper parameters
        total_wrapper_params = wrapper_params_per_expert * self.num_experts

        # Total parameters
        total_params = core_params + total_wrapper_params
        
        # Double check vs actual count
        actual_count = sum(p.numel() for p in self.parameters())
        assert total_params == actual_count, f"Parameter count mismatch: calculated {total_params}, actual {actual_count}"

        # Original uncompressed parameters (for comparison)
        original_params = self.num_experts * (self.d_in + self.has_bias) * self.d_out

        return {
            "core_params": core_params,
            "wrapper_params_per_expert": wrapper_params_per_expert,
            "total_wrapper_params": total_wrapper_params,
            "total_params": total_params,
            "original_params": original_params,
            "compression_ratio": total_params / original_params,
            "reduction_percentage": (1 - total_params / original_params) * 100
        }

    def reconstruct_experts(self) -> List[torch.Tensor]:
        """
        Reconstruct full weight matrices for all experts.

        Returns:
            List of reconstructed weight matrices [num_experts] each of shape [d_out, d_in]
        """
        return [
            expert.reconstruct(self.core)
            for expert in self.experts
        ]

def initialize_from_experts(
    expert_weights: List[torch.Tensor],
    rank: int,
    num_steps: int = 1000,
    lr: float = 1e-3,
    device: str = "cuda",
    logger:Optional[logging.getLogger]=None,
    layer_name: str = "",
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Zero-shot initialization of shared core + wrappers from original expert weights.

    Algorithm:
    1. Initialize core as mean of all expert weights
    2. Initialize wrappers: U=0, V~N(0,1)
    3. Optimize to minimize ||W_e - W_hat_e||_F^2 using Adam

    This implementation uses batched operations for efficiency.

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
    # Stack expert weights into a single tensor [num_experts, d_out, d_in]
    expert_weights_stacked = torch.stack([
        w.to(device).detach().to(torch.float32)
        for w in expert_weights
    ])
    num_experts, d_out, d_in = expert_weights_stacked.shape

    # Step 1: Initialize core as mean
    core = expert_weights_stacked.mean(dim=0).clone()
    core.requires_grad_(True)

    # Step 2: Initialize wrappers (LoRA-style) - batched across all experts
    # Shape: [num_experts, d_in, rank]
    U_in_batch = torch.zeros(num_experts, d_in, rank, device=device, dtype=torch.float32)
    U_in_batch.requires_grad_(True)

    V_in_batch = torch.randn(num_experts, d_in, rank, device=device, dtype=torch.float32) / math.sqrt(rank)
    V_in_batch.requires_grad_(True)

    # Shape: [num_experts, d_out, rank]
    U_out_batch = torch.zeros(num_experts, d_out, rank, device=device, dtype=torch.float32)
    U_out_batch.requires_grad_(True)

    V_out_batch = torch.randn(num_experts, d_out, rank, device=device, dtype=torch.float32) / math.sqrt(rank)
    V_out_batch.requires_grad_(True)

    # Collect all parameters for optimizer
    params = [core, U_in_batch, V_in_batch, U_out_batch, V_out_batch]

    # Step 3: Optimize with Adam
    optimizer = torch.optim.Adam(params, lr=lr)

    # Pre-create identity matrices (reused across iterations)
    I_in = torch.eye(d_in, device=device, dtype=torch.float32)  # [d_in, d_in]
    I_out = torch.eye(d_out, device=device, dtype=torch.float32)  # [d_out, d_out]
    
    inital_loss = (expert_weights_stacked**2).sum().item() / num_experts

    for step in range(num_steps):
        optimizer.zero_grad()

        # Batched reconstruction for all experts
        # Input wrapper: (I + U_in @ V_in^T) for each expert
        # Shape: [num_experts, d_in, rank] @ [num_experts, rank, d_in] -> [num_experts, d_in, d_in]
        input_low_rank = torch.bmm(U_in_batch, V_in_batch.transpose(1, 2))  # [num_experts, d_in, d_in]
        input_wrapped = I_in.unsqueeze(0) + input_low_rank  # [num_experts, d_in, d_in]

        # Apply core to all input-wrapped matrices
        # core @ input_wrapped: [d_out, d_in] @ [num_experts, d_in, d_in]
        # We need to broadcast: [num_experts, d_out, d_in] @ [num_experts, d_in, d_in]
        core_expanded = core.unsqueeze(0).expand(num_experts, -1, -1)  # [num_experts, d_out, d_in]
        temp = torch.bmm(core_expanded, input_wrapped)  # [num_experts, d_out, d_in]

        # Output wrapper: (I + U_out @ V_out^T) @ temp for each expert
        # Shape: [num_experts, d_out, rank] @ [num_experts, rank, d_out] -> [num_experts, d_out, d_out]
        output_low_rank = torch.bmm(U_out_batch, V_out_batch.transpose(1, 2))  # [num_experts, d_out, d_out]
        output_wrapped = I_out.unsqueeze(0) + output_low_rank  # [num_experts, d_out, d_out]

        # Final reconstruction: [num_experts, d_out, d_out] @ [num_experts, d_out, d_in]
        W_hat = torch.bmm(output_wrapped, temp)  # [num_experts, d_out, d_in]

        # Compute total reconstruction loss (batched Frobenius norm)
        # ||W_e - W_hat_e||_F^2 for all experts
        reconstruction_error = expert_weights_stacked - W_hat
        total_loss = (reconstruction_error ** 2).sum()  # Sum over all elements

        # Backprop and step
        total_loss.backward()
        optimizer.step()

        # Logging
        if step % 50 == 0 or step == num_steps - 1:
            avg_loss = total_loss.item() / num_experts
            log_str = f"{layer_name} Step {step}/{num_steps}, Avg Reconstruction Loss: {avg_loss:.6f} relative recon loss: {avg_loss/inital_loss:.6f}"
            if logger:
                logger.info(log_str)
            else:
                print(log_str)  
    # Detach and return as list of tuples (for backward compatibility)
    core_final = core.detach()
    wrappers_final = [
        (
            U_in_batch[i].detach(),
            V_in_batch[i].detach(),
            U_out_batch[i].detach(),
            V_out_batch[i].detach()
        )
        for i in range(num_experts)
    ]

    return core_final, wrappers_final
