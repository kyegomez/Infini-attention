import torch 
from torch import nn, Tensor
from zeta.nn.attention import LinearAttention, Attention

class InfiniAttention(nn.Module):
    """
    InfiniAttention module applies attention mechanism to the input tensor.

    Args:
        dim (int): The input dimension.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dim_head (int, optional): The dimension of each attention head. Defaults to 64.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.head_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        
        # Attention
        self.attention = Attention(
            dim,
            dim_head,
            heads,
            causal=True,
            qk_norm=True,
            kv_heads=4,
            dropout=dropout
        )
        
        # Linear Attention
        self.l_attn = LinearAttention(
            dim,
            heads,
            dim_head,
            dropout=dropout
        )
        
        # Projection
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the InfiniAttention module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying attention mechanism.
        """
        q = self.proj(x)
    
        # KV Attention
        attended, _ = self.attention(x, q)
        
        # Linear Attention
        linear_attended = self.l_attn(x)
        
        # Concatenate
        out = torch.cat([attended, linear_attended], dim=1)
        print(out.shape)
        
        return self.proj(out)
    
    
# x = torch.randn(1, 32, 64)

# attn = InfiniAttention(64)
# out = attn(x)
# print(out.shape)