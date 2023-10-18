import json
import time
from pathlib import Path
from typing import Optional

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer


class LLaMA:
    def __init__(
        self,
        model: Transformer,
        tokenizer: SentencePieceProcessor,
        model_args: ModelArgs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert (
                len(checkpoints) > 0
            ), f"No checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {(time.time() - prev_time):.2f}s")

        prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            device=device,
            **params,
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            # # The only unmatched key in the checkpoint is rope.freqs because we are
            # computing the frequencies for the rotary positional embeddings
            # using `precompute_theta_pos_frequencies`
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded weights in {(time.time() - prev_time):.2f}s")

        return LLaMA(model, tokenizer, model_args)

    def text_completion(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ) -> tuple[list]:
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len

        # Convert each prompt into tokens using the tokenizer
        prompt_tokens = [
            self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
            for prompt in prompts
        ]
        # Make sure the `batch_size` is not too large
        batch_size = len(prompt_tokens)
        assert (
            batch_size <= self.args.max_batch_size
        ), "Batch size should not exceed the maximum batch size"

        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the `max_prompt_len` is not more than the `max_seq_len`
        assert (
            max_prompt_len <= self.args.max_seq_len
        ), "Maximum prompt length should be less than or equal to the maximum sequence length"

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=device
        )
        for k, t in enumerate(prompt_tokens):
            # Populate the initial tokens with the prompt tokens
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = (
            tokens != pad_id
        )  # True if token is a prompt token else False

        for curr_pos in tqdm(range(1, total_len), desc="Generating tokens"):
            with torch.no_grad():
                # Shape: (batch_size, 1, vocab_size)
                logits = self.model(tokens[:, curr_pos - 1], curr_pos)
            if temperature > 0:
                # Temperature is applied BEFORE the softmax
                probs = torch.softmax(
                    logits[:, -1] / temperature, dim=-1
                )  # Shape: (batch_size, vocab_size) -> probability distribution over vocabulary
                next_token = self._sample_top_p(probs, top_p)
            else:
                # If no temperature specified, we use greedy search
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace token if it is a padding token (do it over the batches)
            next_token = torch.where(
                prompt_tokens_mask[:, curr_pos], tokens[:, curr_pos], next_token
            )
            # This replaces the padding tokens across the batches with the predicted tokens
            tokens[:, curr_pos] = next_token

            # EOS is reached only if we found an EOS token for a padding token
            # Basically, EOS is only reached when we find it for one of the tokens
            # that we actually want to carry out inference on
            eos_reached |= (~prompt_tokens_mask[:, curr_pos]) & (
                next_token == self.tokenizer.eos_id()
            )
            if all(eos_reached):  # if all batches have EOS predicted, we break
                break

        out_tokens = []
        out_text = []
        for current_prompt_tokens in tokens.tolist():
            # Cut to the EOS, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)

    def _sample_top_p(self, probs: torch.Tensor, top_p: float):
        # Sort probabilities in descending order
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # Find the cumulative probabilities
        probs_cumsum = torch.cumsum(probs_sort, dim=-1)
        # Shift the cumulative probabilities to the right where the first item will now be 0.0
        # We do this by taking the cumulative probabilities - sorted probabilities
        # Then we generate a mask which gives us the posiiton of items exceeding `top_p`
        mask = probs_cumsum - probs_sort > top_p
        # We zero out these positions
        probs_sort[mask] = 0.0
        # Then we redistribute the probabilities among surviving tokens
        probs_sort.div_(probs_sort.sum(dim=-1, keepdims=True))
        next_token_idx = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, dim=-1, index=next_token_idx)

        return next_token


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot prompt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Elon Musk
        Decision: 
        """,
    ]

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device,
    )

    # Inference
    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
    assert len(out_texts) == len(prompts)

    for i in range(len(out_texts)):
        print(f"{out_texts[i]}")
        print("-" * 50)
