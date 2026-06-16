import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Normalizes each token's representation so its magnitude is ~1.0
    prevents values from growing/shrnking across deep networks

    used in LLaMA 1/2/3, Mistral, Gemma, Qwen
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        """
        learnable scale per dimension
        after forcing RMS=1, the model can learn to amplify
        important dimensions and dampen unimportant ones
        starts at 1.0 (no change initially)
        """
        self.weight = nn.Parameter(data=torch.ones(d_model))
        self.eps = eps # prevents division by zero

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        compute 1/sqrt(mean(x^2))
        rsqrt is 1/sqrt - cpomputed as a single CUDA kernel
        for speed. the mean is over the last dimension (d_model)
        keepdim=True preserves the dimension for broadcasting.
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True)) + self.eps
        return x * rms * self.weight
    