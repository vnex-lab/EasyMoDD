"""Lossless byte tokenizer for custom from-scratch language models."""

import pickle


class ByteTokenizer:
    PAD = 256
    BOS = 257
    EOS = 258
    USER = 259
    ASSISTANT = 260
    vocab_size = 261

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        return [self.BOS, *tokens, self.EOS] if add_special_tokens else tokens

    def encode_prompt(self, user: str) -> list[int]:
        return [self.BOS, self.USER, *user.encode("utf-8"), self.ASSISTANT]

    def encode_dialogue(self, user: str, assistant: str) -> list[int]:
        return [self.BOS, self.USER, *user.encode("utf-8"), self.ASSISTANT, *assistant.encode("utf-8"), self.EOS]

    def decode(self, tokens: list[int], *, skip_special_tokens: bool = True) -> str:
        raw = bytearray()
        for token in tokens:
            if 0 <= int(token) <= 255:
                raw.append(int(token))
            elif not skip_special_tokens:
                raw.extend(f"<token:{int(token)}>".encode("ascii"))
        return raw.decode("utf-8", errors="replace")

    def save(self, filepath: str) -> None:
        with open(filepath, "wb") as handle:
            pickle.dump({"vocab_size": self.vocab_size}, handle)

    @classmethod
    def load(cls, filepath: str) -> "ByteTokenizer":
        with open(filepath, "rb") as handle:
            data = pickle.load(handle)
        if data.get("vocab_size") != cls.vocab_size:
            raise ValueError("Unsupported byte tokenizer vocabulary")
        return cls()
