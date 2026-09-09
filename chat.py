"""Interactive terminal chat for a trained custom checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from byte_tokenizer import ByteTokenizer
from transformer_model import TransformerChatbot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/best.bin")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model = TransformerChatbot.load(args.checkpoint)
    tokenizer = ByteTokenizer.load(Path(args.checkpoint).with_name("tokenizer.bin"))
    print("Custom model chat. Type /exit to quit.")
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.lower() in {"/exit", "/quit"}:
            break
        if prompt:
            response = model.generate(prompt, tokenizer, max_length=args.max_tokens, temperature=args.temperature)
            print(f"Model: {response}")


if __name__ == "__main__":
    main()
