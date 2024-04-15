[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Infini Attention


## install
`$ pip install infini-torch`


## usage

```python
import torch
from infini_torch.attention import InfiniAttention

# Create a random tensor of shape (1, 32, 64)
x = torch.randn(1, 32, 64)

# Create an instance of InfiniAttention with input size 64
attn = InfiniAttention(64)

# Apply the attention mechanism to the input tensor
out = attn(x)

# Print the shape of the output tensor
print(out.shape)
```

# License
MIT
