import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # Number of heads for Q
    n_kv_heads: Optional[int] = None  # Number of heads for K and V
    vocab_size: int = -1  # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # For KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: Optional[str] = None


def precompute_theta_pos_frequencies(
    head_dim: int, seq_len: int, device: str, theta: float = 10000.0
) -> torch.Tensor:
    # Assert that the `head_dim` is even because rotary positional embedding cannot
    # be applied to embeddings with odd number of dimensions (as written in the paper)
    assert head_dim % 2 == 0, "Number of dimensions must be even"

    # Compute theta parameters according to the formula in the paper:
    # theta_i = 10000 ^ (-2 * (i - 1) / d) for i = [1, 2, ..., d/2]
    # where d is the number of dimension
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim / 2, )
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the m parameter - see paper)
    # Shape: (seq_len, )
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # Shape: (seq_len, ) outer product with (head_dim / 2, ) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # Compute the complex numbers in the polar form: c = R * exp(i * m * theta) where R = 1
    # Shape: (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex


def apply_rotary_embeddings(
    x: torch.Tensor, freqs_complex: torch.Tensor, device: str
) -> torch.Tensor:
    # Shape: (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, num_heads, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Shape: (seq_len, head_dim / 2) -> (batch_size, seq_len, num_heads, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Shape: (batch_size, seq_len, num_heads, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (batch_size, seq_len, num_heads, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # Shape: (batch_size, seq_len, num_heads, head_dim / 2) -> (batch_size, seq_len, num_heads, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # Shape: (batch_size, seq_len, num_heads, head_dim / 2, 2) -> (batch_size, seq_len, num_heads, head_dim)
    x_out = x_out.reshape(*x_out.shape)

    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # gamma parameter (learnable)

    def _norm(self, x: torch.Tensor):
        # Formula as stated in paper: a_i = (a_i / RMS(a)) * gamma_i
        # where RMS(a) -> sqrt((1/n) * sum(a_i)) for i = [1, 2, ..., head_dim]
        # Shape: (batch_size, seq_len, head_dim)
        return x * torch.rsqrt(
            x.pow(2).mean(dim=-1, keepdims=True) + self.eps
        )  # torch.rsqrt -> reciprocal of RMS -> 1 / RMS(a)

    def forward(self, x: torch.Tensor):
        # Shape: (head_dim, ) * (batch_size, seq_len, head_dim) -> (batch_size, seq_len, head_dim)
        return self.weight * self._norm(x.float()).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape()
    if n_rep == 1:  # do not need to repeat
        return x
    else:
        # Shape: (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_kv_heads, 1, head_dim)
        return (
            x[:, :, :, None, :]
            .expand(
                batch_size, seq_len, n_kv_heads, n_rep, head_dim
            )  # repeat along the new dimension `n_rep` times
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates the number of times the heads of the Keys and Values
        # should be repeated to match the number of heads for the Queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        (
            batch_size,
            seq_len,
            _,
        ) = x.shape()  # (batch_size, seq_len, dim) where seq_len = 1

        # Apply the Wq, Wk and Wv matrices to Queries, Keys and Values
        # Shape: (batch_size, 1, dim) -> (batch_size, 1, n_heads_q * head_dim)
        xq = self.wq(x)
        # Shape: (batch_size, 1, dim) -> (batch_size, 1, n_kv_heads * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # Shape: (batch_size, 1, n_heads_q * head_dim) -> (batch_size, 1, n_heads_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # Shape: (batch_size, 1, n_kv_heads * head_dim) -> (batch_size, 1, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Does not change the shape of the tensors (See `apply_rotary_embeddings` function)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # Retrieve all the cache Keys and Values stored so far
        # Shape: (batch_size, seq_len, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Repeat the heads of the Keys and Values to match with the number of heads of the Queries
        # Shape: (batch_size, seq_len, n_kv_heads, head_dim) -> (batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # Shape: (batch_size, 1, n_heads_q, head_dim) -> (batch_size, n_heads_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        # Shape: (batch_size, seq_len, n_kv_heads * n_rep, head_dim) -> (batch_size, n_kv_heads * n_rep, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Shape: (batch_size, n_heads_q, 1, head_dim) @ (batch_size, n_kv_heads * n_rep, head_dim, seq_len) -> (batch_size, n_heads_q, 1, seq_len)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Shape: (batch_size, n_heads_q, 1, seq_len) @ (batch_size, n_kv_heads * n_rep, seq_len, head_dim) -> (batch_size, n_heads_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # Shape: (batch_size, n_heads_q, 1, head_dim) -> (batch_size, 1, n_heads_q * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Shape: (batch_size, 1, n_heads_q * head_dim) or (batch_size, 1, dim)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round up the hidden_dim to the nearest multiple_of parameter
        # Example: We have a hidden_dim = 7 and a multiple_of = 5, Answer = 10
        # ((7 + 5 - 1) // 5) * 5 = 2 * 5 = 10
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # As in LLaMA: FFN_SwiGLU(x, W1, W2, W3) = (Swish(xW1) * xW3)W2
        # Shape: (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 4) -> (batch_size, seq_len, hidden_dim)
        swish = F.silu(self.w1(x))
        # Shape: (batch_size, seq_len, dim) -> (batch_size, seq_len, dim * 4) -> (batch_size, seq_len, hidden_dim)
        x_V = self.w3(x)
        # Shape: (batch_size, seq_len, hidden_dim)
        x = swish * x_V
        # Shape: (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, dim)
        x = self.w2(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization BEFORE self-attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization AFTER self-attention and BEFORE feed forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor
    ) -> torch.Tensor:
        # Shape: (batch_size, seq_len, dim) + (batch_size, 1, dim) -> (batch_size, seq_len, dim)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        # Shape: (batch_size, seq_len, dim) + (batch_size, seq_len, dim) -> (batch_size, seq_len, dim)
        out = h + self.feed_forward(self.ffn_norm(h))

        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocabulary size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(
            self.args.dim // self.args.n_heads,
            self.args.max_seq_len * 2,
            device=self.args.device,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch_size, seq_len)
        _, seq_len = tokens.shape
        # With KV cache, we can pass in one token at a time (excluding previous tokens)
        # Therefore, assert sequence length is equal to 1
        assert seq_len == 1, "Only one token can be processed at a time"

        # This model is only good for intferencing and not training (we are using pre-trained LLaMA 7B weights)
        # because for training, we need to be able to process multiple tokens (so we can't use KV cache)

        # Shape: (batch_size, seq_len) -> # (batch_size, seq_len, dim) (depending on model size, dim can be different)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)

        # Shape: (batch_size, seq_len, dim) -> (batch_size, seq_len, vocab_size)
        output = self.output(h).float()

        return output
