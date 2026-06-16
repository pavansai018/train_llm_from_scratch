import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU - gated version of swish activation function.
    the 'gate' (right side of multiplication) learns to selectively
    pass or block information - like a faucet

    standard FFN: output = W2(ReLU(W1(x)))
    SwiGLU FFN: output = W3(SiLU(W1(x))) * (W2(x))

    the gate multiples values: if gate ~0, block info
                               if gate ~1, pass info
                               if gate ~0.5, partial pass

    the gating mechanism is what makes SwiGLU outperform ReLU and GELU
    the model learns where to apply non-linearity

    used in LLaMA 1/2/3, PaLM, Gemini
    """

    def __init__(self, d_model: int, expansion_factor: int = 4):
        super().__init__()
        """
        hidden dim is 4x input/output - the expansion bottleneck
        expand -> process -> contract is more expressive than same-size
        784 -> 3072 -> 784 lets the FFN learn ~4x more complex patterns
        """
        hidden_dim = expansion_factor * d_model

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False) # projects to values
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False) # projects to gates
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False) # projects back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU(w1(x)) are the values, W2(x) are the gates
        SiLU (also called as Swish) = x * sigmoid(x)
        it's smooth (unlike ReLU which has a sharp corner at 0),
        which makes gradients flow better during training.
        Gate multiplies values element-wise, selectively passing info.
    
        """
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
    