import math

from dataclasses import dataclass


class SampleRequest:
    def __init__(self, context: list[dict], continuation: str):
        self.args = (context, continuation)


@dataclass
class Token:
    token: str
    logprob: float | None
    rank: int | None


@dataclass
class Sequence:
    tokens: list[Token]

    def __post_init__(self):
        n: int = len([t for t in self.tokens if t.rank])

        self.text: str = "".join([t.token for t in self.tokens])
        self.logprob: float | None = sum(ll for token in self.tokens if (ll := token.logprob)) if n else None
        self.avg_logprob: float | None = self.logprob / n if n else None
        self.ppl: float | None = math.exp(-self.avg_logprob) if n else None
        self.ranks: list[int] = [rank for token in self.tokens if (rank := token.rank)]

    def __str__(self):
        return_val = f"text: {self.text}\n"
        return_val += f"logprob: {self.logprob}\n"
        return_val += f"avg_logprob: {self.avg_logprob}\n"
        return_val += f"PPL: {self.ppl}\n"
        return_val += f"ranks: {self.ranks}\n"
        return return_val

    @property
    def data(self) -> dict[str, str | float | list[int]]:
        return {
            "text": self.text.strip(),
            "logprob": self.logprob,
            "avg_logprob": self.avg_logprob,
            "perplexity": self.ppl,
            "ranks": self.ranks,
        }

    def __repr__(self):
        return self.__str__()


@dataclass
class Prompt:
    context: Sequence | None
    continuation: Sequence | None
    next_tokens: list[Token] | None

    def __str__(self):
        return_val = "Context:\n"
        return_val += str(self.context) + "\n"
        return_val += "Continuation:\n"
        return_val += str(self.continuation) + "\n"
        return_val += str(self.next_tokens)

        return return_val
