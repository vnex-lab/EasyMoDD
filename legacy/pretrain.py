"""Train the custom decoder-only Transformer from random initialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from byte_tokenizer import ByteTokenizer
from transformer_model import TransformerChatbot


def load_tokens(args, tokenizer):
    if args.stage == "dialogue":
        rows = json.loads(Path(args.dialogue).read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Dialogue JSON must be a list of user/bot objects")
        tokens = []
        for index, row in enumerate(rows):
            user = row.get("user", row.get("prompt", row.get("input")))
            bot = row.get("bot", row.get("assistant", row.get("response")))
            if not user or not bot:
                raise ValueError(f"Dialogue row {index} needs user/bot text")
            tokens.extend(tokenizer.encode_dialogue(str(user), str(bot)))
        return tokens
    return tokenizer.encode("\n\n".join(Path(path).read_text(encoding="utf-8") for path in args.corpus))


def windows(tokens, length):
    if length < 2:
        raise ValueError("--seq-len must be at least 2")
    return [np.asarray(tokens[start:start + length + 1], dtype=np.int32)
            for start in range(0, len(tokens) - length, length)
            if len(tokens[start:start + length + 1]) == length + 1]


def save(model, tokenizer, directory, step, train_loss, val_loss):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path / f"step_{step:07d}.bin"))
    model.save(str(path / "best.bin"))
    tokenizer.save(str(path / "tokenizer.bin"))
    (path / "training_state.json").write_text(json.dumps({"step": step, "train_loss": train_loss, "val_loss": val_loss}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["pretrain", "dialogue"], default="pretrain")
    parser.add_argument("--corpus", nargs="*", default=[])
    parser.add_argument("--dialogue", default=None)
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()
    if args.stage == "pretrain" and not args.corpus:
        parser.error("--corpus is required for pretraining")
    if args.stage == "dialogue" and not args.dialogue:
        parser.error("--dialogue is required for dialogue training")

    tokenizer = ByteTokenizer()
    data = windows(load_tokens(args, tokenizer), args.seq_len)
    if len(data) < 2:
        raise ValueError("Dataset is too small; provide more text or reduce --seq-len")
    split = max(1, int(len(data) * 0.9))
    train_data, validation_data = data[:split], data[split:]
    model = TransformerChatbot.load(args.init_from) if args.init_from else TransformerChatbot(
        vocab_size=tokenizer.vocab_size, embed_dim=args.embed_dim, num_heads=args.heads,
        num_layers=args.layers, ff_dim=args.ff_dim, max_seq_len=args.seq_len,
        learning_rate=args.learning_rate, decoder_only=True)
    if model.vocab_size != tokenizer.vocab_size or not model.decoder_only:
        raise ValueError("Checkpoint must be a decoder-only byte-token model")
    best = math.inf
    rng = np.random.default_rng(42)
    print(f"Parameters: {model._count_parameters():,}; train windows: {len(train_data):,}; validation: {len(validation_data):,}")
    for step in range(1, args.steps + 1):
        item = train_data[int(rng.integers(len(train_data)))]
        train_loss = model.train_batch([item], [item])
        if step % args.validate_every and step != args.steps:
            continue
        sample = validation_data[: min(32, len(validation_data))]
        losses = []
        for sequence in sample:
            logits = model.forward(sequence[:-1].reshape(1, -1), sequence[:-1].reshape(1, -1))
            shifted = logits - np.max(logits, axis=-1, keepdims=True)
            probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
            losses.append(float(-np.mean(np.log(probabilities[0, np.arange(len(sequence) - 1), sequence[1:]] + 1e-10))))
        val_loss = float(np.mean(losses))
        print(f"step={step:,} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            save(model, tokenizer, args.checkpoint_dir, step, train_loss, val_loss)


if __name__ == "__main__":
    main()
