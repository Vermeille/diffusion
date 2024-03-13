import torch
import torch.nn as nn
import torchelie.nn as tnn


class GLU(nn.Module):
    def __init__(self, act_fn):
        super().__init__()
        self.act_fn = act_fn

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return self.act_fn(x1) * x2


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_size, causal):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attention = tnn.SelfAttention(
            hidden_dim, num_heads, head_size, causal
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            GLU(nn.SiLU()),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, head_size, causal):
        super().__init__()
        self.gating = nn.Linear(hidden_dim, 6 * hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.self_attention = tnn.ConditionalSelfAttention(
            hidden_dim, num_heads, head_size, causal
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            GLU(nn.SiLU()),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x, t):
        scale_1, shift_1, out_scale_1, scale_2, shift_2, out_scale_2 = torch.chunk(
            self.gating(x), 6, dim=-1
        )
        x = x * out_scale_1 + self.self_attention(self.norm1(x, t) * scale_1 + shift_1)
        x = x * out_scale_2 + self.mlp(self.norm2(x, t) * scale_2 + shift_2)
        return x


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, hidden_dim, num_heads, head_size, num_layers, causal
    ):
        super().__init__()
        self.vocab = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.Sequential(
            *[
                TransformerBlock(hidden_dim, num_heads, head_size, causal)
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.vocab(x)
        x = self.layers(x)
        return self.proj(x)

    @property
    def num_layers(self):
        return len(self.layers)


class Denoiser(nn.Module):
    def __init__(
        self, in_features, hidden_dim, num_heads, head_size, num_layers, causal
    ):
        super().__init__()
        self.vocab = nn.Linear(in_features, hidden_dim)
        self.layers = tnn.CondSeq(
            [
                ConditionalTransformerBlock(hidden_dim, num_heads, head_size, causal)
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(hidden_dim, in_features)

    def forward(self, x, t):
        x = self.vocab(x)
        x = self.layers(x, t)
        return self.proj(x)

    @property
    def num_layers(self):
        return len(self.layers)
