import json
import math

from typing import Self, TypedDict
from operator import itemgetter
from functools import cached_property
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCard:
    """vLLM /v1/models response"""

    id: str
    root: str

    def __str__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, ModelCard):
            return self.root == other.root
        return False

    def __hash__(self):
        return hash(self.root)

    def __getstate__(self):
        return {"id": self.id, "root": self.root}

    def __setstate__(self, state):
        object.__setattr__(self, "id", state["id"])
        object.__setattr__(self, "root", state["root"])

    @property
    def payload(self) -> dict:
        return {"model": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            **{
                "id": data.get("id"),
                "root": data.get("root"),
            }
        )


class SequenceData(TypedDict):
    text: str
    logprob: float | None
    avg_logprob: float | None
    perplexity: float | None
    ranks: list[int]
    token_strings: list[str]


class SampleRequest:
    def __init__(self, context: list[dict], continuation: str):
        self.args = (context, continuation)


@dataclass
class Token:
    token: str
    logprob: float | None
    rank: int | None

    @classmethod
    def from_prompt_logprob(cls, prompt_logprob: dict) -> Self:
        token_data = next(iter(prompt_logprob.values()))
        return cls(
            token=token_data.get("decoded_token"),
            logprob=token_data.get("logprob"),
            rank=token_data.get("rank"),
        )

    @classmethod
    def from_top_logprobs(cls, top_logprobs: dict[str, float]) -> list[Self]:
        sorted_logprobs = dict(sorted(top_logprobs.items(), key=itemgetter(1), reverse=True))

        return [
            cls(token=token, logprob=logprob, rank=rank)
            for rank, (token, logprob) in enumerate(sorted_logprobs.items(), start=1)
        ]

    def __str__(self):
        return f"Token(rank={self.rank}, token={self.token}, logprob={self.logprob})"

    def __repr__(self):
        return self.__str__()


@dataclass
class Sequence:
    tokens: list[Token]

    def __post_init__(self):
        n = len([t for t in self.tokens if t.rank])

        self.text = "".join([t.token for t in self.tokens])
        self.logprob = sum(token.logprob for token in self.tokens) if n else None
        self.avg_logprob = self.logprob / n if n else None
        self.ppl = math.exp(-self.avg_logprob) if n else None
        self.ranks = [token.rank for token in self.tokens]

    @property
    def data(self) -> SequenceData:
        return {
            "text": self.text.strip(),
            "logprob": self.logprob,
            "avg_logprob": self.avg_logprob,
            "perplexity": self.ppl,
            "ranks": self.ranks,
            "token_strings": [t.token for t in self.tokens],  # ← Add this
        }

    def __str__(self):
        return json.dumps(self.data, indent=2)

    def __repr__(self):
        return self.__str__()

    @classmethod
    def parse_prompt_logprobs(cls, prompt_logprobs: list[dict]) -> Self:
        tokens = [Token.from_prompt_logprob(prompt) for prompt in prompt_logprobs]
        return cls(tokens=tokens)


@dataclass
class Prompt:
    choice: dict
    ctxlen: int

    def __post_init__(self):
        *_, self.top_logprobs = self.choice["logprobs"]["top_logprobs"]
        self.prompt_logprobs = self.choice["prompt_logprobs"]

    @cached_property
    def context(self) -> Sequence | None:
        if self.ctxlen <= 1:
            return None
        return Sequence.parse_prompt_logprobs(
            prompt_logprobs=self.prompt_logprobs[1 : self.ctxlen],
        )

    @cached_property
    def continuation(self) -> Sequence | None:
        if self.ctxlen >= len(self.prompt_logprobs):
            return None
        return Sequence.parse_prompt_logprobs(
            prompt_logprobs=self.prompt_logprobs[self.ctxlen :],
        )

    @cached_property
    def next_tokens(self) -> list[Token]:
        return Token.from_top_logprobs(top_logprobs=self.top_logprobs)

    def __str__(self):
        return json.dumps(
            {
                "Context": self.context,
                "Continuation": self.continuation,
                "Next Tokens": self.next_tokens,
            },
            indent=2,
        )

    def __repr__(self):
        return self.__str__()
