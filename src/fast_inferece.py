import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor
import time
import tqdm
import numpy as np

from typing import Optional
from src.ARMOR_compress import ARMOR_Linear

# force CUTLASS use if ``cuSPARSELt`` is not available
SparseSemiStructuredTensor._FORCE_CUTLASS = True
torch.manual_seed(100)

class BlockDiagonalLinear(nn.Module):
    """
    A PyTorch Linear layer with a block diagonal weight matrix.

    This layer is memory and computationally efficient for models where interactions
    between features are known to be sparse and grouped. Instead of a dense
    (d_out, d_in) weight matrix, it uses a (num_blocks, block_size, block_size)
    tensor.

    Args:
        d (int): The total number of input and output features. d_in and d_out are
                 both equal to d.
        block_size (int): The size of each square block on the diagonal.
                          `d` must be divisible by `block_size`.
        bias (bool, optional): If True, adds a learnable bias to the output.
                               Default: True.
    """
    def __init__(self,diag_blocks: torch.Tensor,
                 d:int,
                 bias:bool = True):
        super().__init__()
        block_size = diag_blocks.shape[-1]
        if d % block_size != 0:
            raise ValueError(f"Dimension 'd' ({d}) must be divisible by 'block_size' ({block_size}).")

        self.d = d
        
        self.block_size = block_size
        self.num_blocks = d // block_size
        assert diag_blocks.shape == (self.num_blocks, self.block_size, self.block_size), f"diag_blocks.shape {diag_blocks.shape}"
        self.use_bias = bias

        # Parameters are stored efficiently. Instead of a (d, d) matrix,
        # we store a (num_blocks, block_size, block_size) tensor.
        self.diag_blocks = nn.Parameter(diag_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): The input tensor of shape (..., d), where '...'
                              denotes any number of leading dimensions.

        Returns:
            torch.Tensor: The output tensor of shape (..., d).
        """
        input_shape = x.shape
        # if input_shape[-1] != self.d:
        #     raise ValueError(
        #         f"Expected last dimension of input to be {self.d}, but got {input_shape[-1]}."
        #     )

        # 1. Reshape the input for block-wise processing
        # The input tensor x of shape (..., d) is viewed as
        # (..., num_blocks, block_size)
        leading_dims = input_shape[:-1]
        x_reshaped = x.view(*leading_dims, self.num_blocks, self.block_size)
        # print("x_reshaped.shape", x_reshaped.shape)
        # 2. Perform the block-diagonal matrix multiplication efficiently
        # We use torch.einsum for a highly optimized batched matrix multiplication.
        # '...ni,nji->...nj' translates to:
        # For each element in the leading dimensions '...':
        #   For each block 'n':
        #     Perform a matrix-vector product between the input block '...ni'
        #     and the transposed weight block 'nji' (transposing i and j).
        # The result has the same leading dims and block structure '...nj'.
        # print("self.diag_blocks", self.diag_blocks.shape)
        out_reshaped = torch.einsum('...ni,nji->...nj', x_reshaped, self.diag_blocks)
        
        # 3. Reshape the output back to the original layout
        # (..., num_blocks, block_size) -> (..., d)
        out = out_reshaped.reshape(*leading_dims, self.d)

        return out

    def extra_repr(self) -> str:
        """
        Provides a string representation of the layer's configuration.
        """
        return f'd={self.d}, block_size={self.block_size}, bias={self.use_bias}'
    

class BlockSpareFastLinear(nn.Module):

    def __init__(self, 
                 sparse_weight: torch.Tensor,
                 B: torch.Tensor,
                 A: torch.Tensor,
                 bias: Optional[torch.Tensor] = None,
                 ):
        
        super().__init__()
        self.d_out, self.d_in = sparse_weight.shape
        
        self.sparse_linear = nn.Linear(self.d_in, self.d_out, bias=False)
        self.sparse_linear.weight = nn.Parameter(to_sparse_semi_structured(sparse_weight))
        self.B  = BlockDiagonalLinear(B, d=self.d_in, bias=False)
        self.A  = BlockDiagonalLinear(A, d=self.d_out, bias=False)
        self.bias = bias


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.B(x)
        x = self.sparse_linear(x)
        x = self.A(x)
        if self.bias is not None:
            x = x + self.bias
        return x

@torch.no_grad()
def convert_from_naive_to_fast(
    naive_layer: ARMOR_Linear,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> BlockSpareFastLinear:
    """
    Convert a ARMOR_Linear layer to a BlockSpareFastLinear layer.

    Args:
        naive_layer (ARMOR_Linear): The ARMOR_Linear layer to convert.
        device (torch.device, optional): The device to place the new layer on.

    Returns:
        BlockSpareFastLinear: The converted BlockSpareFastLinear layer.
    """
    assert isinstance(naive_layer, ARMOR_Linear), "Input layer must be an instance of ARMOR_Linear"
    
    naive_layer.to(device).to(torch.float16)
    sparse_weight = naive_layer.naive_compression_module.reconstruct(denormalize=True)
    
    dout, din = sparse_weight.shape
    B = naive_layer.B.diag_blocks.detach().clone()
    A = naive_layer.A.diag_blocks.detach().clone()
    
    #denormalize them appropriately
    B = B.permute(1,0,2)
    B = naive_layer.normalizer.denormalize_otf_in(B.reshape(-1, din)).reshape_as(B).permute(1,0,2)
    A = naive_layer.normalizer.denormalize_otf_out(A.reshape(dout, -1).T).T.reshape_as(A)
    
    A = A.contiguous()
    B = B.contiguous()
    
    
    fast_layer = BlockSpareFastLinear(
        sparse_weight = sparse_weight.detach().clone(),
        B = B,
        A = A,
        bias = naive_layer.bias
    )
    
    
    #check the outputs are the same
    x = torch.randn(2, din, dtype=torch.float16).to(device)
    naive_out = naive_layer(x)
    fast_out = fast_layer(x)
    if not torch.allclose(naive_out, fast_out, atol=1e-1, rtol=1e-1):
        print("naive out", naive_out)
        print("fast_out", fast_out)
        print("Max absolute difference:", (naive_out - fast_out).abs().max().item())
        raise ValueError("Outputs of the naive and fast layers do not match!")
    
    return fast_layer
        
                