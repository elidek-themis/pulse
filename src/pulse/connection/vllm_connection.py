import math

from typing import Final
from operator import itemgetter
from dataclasses import dataclass

import requests

from streamlit import logger
from streamlit.connections import BaseConnection
from lm_eval.models.api_models import TemplateAPI

from pulse.utils.tools import ModelCard

_LOGGER: Final = logger.get_logger(__name__)


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


class SampleRequest:
    def __init__(self, context: list[dict], continuation: str):
        self.args = (context, continuation)


class VLLMConnection(BaseConnection):
    def _connect(self, seed=2025, **kwargs) -> "VLLMConnection":
        """Searches for credentials in `kwargs` or `streamlit secrets`."""

        if "base_url" in kwargs:
            _LOGGER.info("Using base_url from kwargs.")
            self._base_url = kwargs["base_url"]
        elif hasattr(self._secrets, "base_url"):
            _LOGGER.info("Using base_url from secrets.")
            self._base_url = self._secrets["base_url"]
        else:
            raise ValueError("No base_url provided in kwargs or secrets.")

        if "token" in kwargs:
            _LOGGER.info("Using token from kwargs.")
            self._token = kwargs["token"]
        elif hasattr(self._secrets, "token"):
            _LOGGER.info("Using token from secrets.")
            self._token = self._secrets["token"]
        else:
            _LOGGER.warning("No token provided in kwargs or secrets. Falling back to `EMPTY`.")
            self._token = "EMPTY"

        self._seed = seed

    @property
    def chat_template(self) -> str:
        assert hasattr(self, "lm"), "No model has been assigned. Use `assign_model` first."
        return self.lm.tokenizer.chat_template

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def assign_model(self, model_card: ModelCard) -> None:
        base_url = f"{self._base_url}/v1/completions"
        self.lm = VLLMCompletions(
            base_url=base_url,
            api_key=self._token,
            model_card=model_card,
            seed=self._seed,
        )
        self.max_logprobs = self._get_max_logprobs()

    def get_models(self) -> requests.Response:
        resp = requests.get(f"{self._base_url}/v1/models", headers=self.headers)
        resp.raise_for_status()
        return resp

    def get_model_config(self) -> requests.Response:
        payload = {"model": self.lm.model}
        resp = requests.get(f"{self._base_url}/model_config", json=payload, headers=self.headers)
        if not resp.ok:
            _LOGGER.warning(
                "/model_config endpoint not enabled, serve with --middleware utils.CustomRouteMiddleware",
            )
        return resp

    def _get_max_logprobs(self, vllm_default: int = 20) -> int:
        resp = self.get_model_config()
        model_config = resp.json()

        return model_config.get("max_logprobs", vllm_default)

    def sample(self, requests: list[SampleRequest], **kwargs) -> list[Prompt]:
        return self.lm.sample(requests=requests, **kwargs)


class VLLMCompletions(TemplateAPI):
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model_card: ModelCard = None,
        tokenizer_backend: str = "huggingface",
        **kwargs,
    ):
        model = model_card.root  # root name for tokenizer
        super().__init__(base_url=base_url, tokenizer_backend=tokenizer_backend, model=model, **kwargs)
        self.model = model_card.id  # served model name
        self.api_key = api_key

    def sample(
        self,
        requests: list[SampleRequest],
        parse_context: bool = False,
        parse_continuation: bool = False,
        parse_next_tokens: bool = True,
        **kwargs,
    ) -> list[Prompt]:
        assert self.tokenized_requests
        extra_body = kwargs.get("extra_body", {})
        add_generation_prompt = extra_body.pop("add_generation_prompt", True)

        sample_requests = []
        for chat, continuation in [req.args for req in requests]:
            context = self.apply_chat_template(
                chat_history=chat,
                add_generation_prompt=add_generation_prompt,
            )

            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
            sample_requests.append((None, context_enc, continuation_enc))

        inputs, ctxlens, _ = self.batch_loglikelihood_requests([sample_requests])
        outputs = self.model_call(messages=inputs, generate=False, **kwargs)
        parsed = self.parse_logprobs(
            outputs=outputs,
            tokens=inputs,
            ctxlens=ctxlens,
            parse_context=parse_context,
            parse_continuation=parse_continuation,
            parse_next_tokens=parse_next_tokens,
            **kwargs,
        )

        return parsed

    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        generate=False,
        gen_kwargs: dict | None = None,
        seed: int = 2025,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            raise NotImplementedError
        print(kwargs)
        extra_body = kwargs.pop("extra_body", {})

        to_ret = {
            "model": self.model,
            "prompt": messages,
            "temperature": 1,
            "max_tokens": 1,
            "logprobs": 20,
            "seed": seed,
            "echo": True,
            "prompt_logprobs": 1,
            **extra_body,  # will overwrite
        }

        print(to_ret)
        return to_ret

    @staticmethod
    def parse_logprobs(
        outputs: dict | list[dict],
        tokens: list[list[int]] = None,
        ctxlens: list[int] = None,
        parse_context: bool = False,
        parse_continuation: bool = True,
        parse_next_tokens: bool = False,
        **kwargs,
    ) -> list[list[Token]]:
        results = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            choice_ctxlen = zip(sorted(out["choices"], key=itemgetter("index")), ctxlens)
            for choice, ctxlen in choice_ctxlen:
                (top_logprobs,) = choice["logprobs"]["top_logprobs"]
                prompt_logprobs = choice["prompt_logprobs"]

                results.append(
                    Prompt(
                        context=VLLMCompletions._parse_logprobs(prompt_logprobs[1:ctxlen]) if parse_context else None,
                        continuation=VLLMCompletions._parse_logprobs(prompt_logprobs[ctxlen:])
                        if parse_continuation
                        else None,
                        next_tokens=VLLMCompletions._parse_next_tokens(top_logprobs) if parse_next_tokens else None,
                    )
                )

        return results

    @staticmethod
    def _parse_logprobs(prompt_logprobs: list[dict]) -> Sequence:
        seq = []
        for prompt in prompt_logprobs:
            token = next(iter(prompt.values()))
            seq.append(
                Token(
                    token=token["decoded_token"],
                    logprob=token["logprob"],
                    rank=token["rank"],
                )
            )

        return Sequence(tokens=seq)

    @staticmethod
    def _parse_next_tokens(top_logprobs: dict[str, float]) -> list[Token]:
        next_tokens = []
        top_logprobs = dict(sorted(top_logprobs.items(), key=itemgetter(1), reverse=True))
        for i, token in enumerate(top_logprobs, start=1):
            next_tokens.append(
                Token(
                    token=token,
                    logprob=top_logprobs[token],
                    rank=i,
                )
            )

        return next_tokens

    @staticmethod
    def parse_generations(outputs: dict | list[dict], **kwargs) -> Prompt:
        raise NotImplementedError
