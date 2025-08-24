import math

from typing import Final
from operator import itemgetter
from dataclasses import dataclass

import requests

from tqdm import tqdm
from openai import OpenAI
from streamlit import logger
from openai.resources.chat import Chat
from streamlit.connections import BaseConnection
from openai.resources.models import Models
from lm_eval.models.api_models import TemplateAPI
from openai.resources.completions import Completions

_LOGGER: Final = logger.get_logger(__name__)


class VLLMConnection(BaseConnection):
    """Connection to a vLLM server with OpenAI-compatible API endpoints."""

    def _connect(self, **kwargs) -> "VLLMConnection":
        """Searches for credentials in `kwargs` or `self._secrets`."""

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

        self._client = OpenAI(base_url=self._base_url + "/v1/", api_key=self._token)

    def assign_model(self, model: str) -> None:
        base_url = f"{self._base_url}/v1/completions"
        self.model = VLLMCompletions(base_url=base_url, model=model)

    @property
    def chat_template(self) -> str:
        assert hasattr(self, "model"), "No model has been assigned. Use `assign_model` first."
        return self.model.tokenizer.chat_template

    @property
    def token(self) -> str:
        return self._token

    @token.setter
    def token(self, value: str) -> None:
        _LOGGER.info("Setting token for VLLMConnection.")
        self._token = value
        # also update the OpenAI client
        self._client = OpenAI(base_url=self._base_url + "/v1", api_key=value)

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @property
    def client(self) -> OpenAI:
        """Access the underlying OpenAI client"""
        return self._client

    # Standard OpenAI-compatible endpoints (delegated to the OpenAI client)
    @property
    def chat(self) -> Chat:
        return self._client.chat

    @property
    def completions(self) -> Completions:
        return self._client.completions

    @property
    def models(self) -> Models:
        return self._client.models

    def get_openapi_spec(self) -> requests.Response:
        """Grab the OpenAPI specification for the vLLM server."""
        r = requests.get(f"{self._base_url}/openapi.json", headers=self.headers)
        r.raise_for_status()
        return r

    def get_swagger_ui(self) -> requests.Response:
        """Fetch the Swagger UI HTML (/docs endpoint)."""
        r = requests.get(f"{self._base_url}/docs", headers=self.headers)
        r.raise_for_status()
        return r

    def get_redoc(self) -> requests.Response:
        """Fetch the ReDoc HTML (/redoc endpoint)."""
        r = requests.get(f"{self._base_url}/redoc", headers=self.headers)
        r.raise_for_status()
        return r

    def health(self) -> requests.Response:
        """Check server health."""
        r = requests.get(f"{self._base_url}/health", headers=self.headers)
        r.raise_for_status()
        return r

    def load_model(self, model: str) -> requests.Response:
        """Load a specific model (used for pre-loading/warmup)."""
        r = requests.get(f"{self._base_url}/load?model={model}", headers=self.headers)
        r.raise_for_status()
        return r

    def ping(self) -> requests.Response:
        """Ping the server."""
        r = requests.get(f"{self._base_url}/ping", headers=self.headers)
        r.raise_for_status()
        return r

    def tokenize(self, prompt: str, model: str | None = None) -> requests.Response:
        """Tokenize prompt

        .. note::
           not defining `model` will choose the first model in the models list

        """
        payload = {"prompt": prompt}
        if model:
            payload["model"] = model

        r = requests.post(f"{self._base_url}/tokenize", json=payload, headers=self.headers)
        r.raise_for_status()
        return r

    def detokenize(self, token_ids: list[int], model: str | None = None) -> requests.Response:
        """Detokenize token IDs

        .. note::
           not defining `model` will choose the first model in the models list

        """
        payload = {"tokens": token_ids}
        if model:
            payload["model"] = model

        r = requests.post(f"{self._base_url}/detokenize", json=payload, headers=self.headers)
        r.raise_for_status()
        return r

    def get_models(self) -> requests.Response:
        """List available models."""
        r = requests.get(f"{self._base_url}/v1/models", headers=self.headers)
        r.raise_for_status()
        return r

    def get_version(self) -> requests.Response:
        """Get server version."""
        r = requests.get(f"{self._base_url}/version", headers=self.headers)
        r.raise_for_status()
        return r


@dataclass
class Token:
    token: str
    logprob: float | None
    rank: int | None

    def __post_init__(self):
        self.prob: float | None = math.exp(self.logprob) if self.logprob else None
        self.is_greedy: bool = self.rank == 1

    def __str__(self):
        return f"Token(token={self.token}, logprob={self.logprob}, rank={self.rank}, prob={self.prob}, is_greedy={self.is_greedy})"


@dataclass
class Sequence:
    tokens: list[Token]

    def __post_init__(self):
        n: int = len([t for t in self.tokens if t.rank])

        self.text: str = "".join([t.token for t in self.tokens])
        self.logprob: float = sum([logprob for token in self.tokens if (logprob := token.logprob)])
        self.avg_logprob: float = self.logprob / n
        self.ppl: float = math.exp(-self.avg_logprob)
        self.ranks: list[int] = [rank for token in self.tokens if (rank := token.rank)]

    def __str__(self):
        return_val = f"text: {self.text}\n"
        return_val += f"logprob: {self.logprob}\n"
        return_val += f"avg_logprob: {self.avg_logprob}\n"
        return_val += f"PPL: {self.ppl}\n"
        return_val += f"ranks: {self.ranks}\n"
        return return_val

    def __repr__(self):
        return self.__str__()


@dataclass
class Prompt:
    context: Sequence
    continuation: Sequence
    next_tokens: list[Token]


# @register_model("vllm-completions")
class VLLMCompletions(TemplateAPI):
    def __init__(
        self,
        base_url: str = None,
        tokenizer_backend: str = "huggingface",
        **kwargs,
    ):
        super().__init__(base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs)

    def loglikelihood(self, requests, disable_tqdm: bool = False, **kwargs) -> list[Prompt]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":  # BOS or EOS as context
                context_enc, continuation_enc = ([self.prefix_token_id], self.tok_encode(continuation))
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm, **kwargs)

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs) -> list[Prompt]:
        assert self.tokenizer is not None, "Tokenizer is required for loglikelihood tasks."

        inputs, ctxlens, cache_keys = self.batch_loglikelihood_requests([requests])
        outputs = self.model_call(messages=inputs, generate=False, **kwargs)
        if isinstance(outputs, dict):
            outputs = [outputs]

        parsed = self.parse_logprobs(outputs=outputs, tokens=inputs, ctxlens=ctxlens)
        assert len(requests) == len(inputs) == len(ctxlens) == len(parsed)

        results = []
        pbar = tqdm(desc="Requesting API", total=len(requests))
        for answer_, cache_key in zip(parsed, cache_keys):
            if answer_ is not None:
                results.append(answer_)
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer_)
            pbar.update(1)
        return results

    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        generate=False,
        gen_kwargs: dict | None = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            raise NotImplementedError

        extra_body = kwargs.pop("extra_body", {})

        return {
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

    @staticmethod
    def parse_logprobs(
        outputs: dict | list[dict],
        tokens: list[list[int]] = None,
        ctxlens: list[int] = None,
        **kwargs,
    ) -> list[list[Token]]:
        results = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            choice_ctxlen = zip(sorted(out["choices"], key=itemgetter("index")), ctxlens)
            for choice, ctxlen in choice_ctxlen:
                (first_token, *_) = choice["logprobs"]["tokens"]  # _, *(ctx + cont + prediction)
                *_, top_logprobs = choice["logprobs"]["top_logprobs"]  # *(_, ctx + cont), prediction
                _, *prompt_logprobs = choice["prompt_logprobs"]  # _, *(ctx + cont)

                next_tokens = []
                for i, token in enumerate(top_logprobs, start=1):
                    next_tokens.append(
                        Token(
                            token=token,
                            logprob=top_logprobs[token],
                            rank=i,
                        )
                    )

                # first token doesn't have a logprob or rank
                ctx = [Token(token=first_token, logprob=None, rank=None)]
                for prompt in prompt_logprobs[:ctxlen]:
                    token = next(iter(prompt.values()))
                    ctx.append(
                        Token(
                            token=token["decoded_token"],
                            logprob=token["logprob"],
                            rank=token["rank"],
                        )
                    )
                context = Sequence(tokens=ctx)

                cont = []
                for prompt in prompt_logprobs[ctxlen:]:
                    token = next(iter(prompt.values()))
                    cont.append(
                        Token(
                            token=token["decoded_token"],
                            logprob=token["logprob"],
                            rank=token["rank"],
                        )
                    )
                continuation = Sequence(tokens=cont)

                results.append(
                    Prompt(
                        context=context,
                        continuation=continuation,
                        next_tokens=next_tokens,
                    )
                )

        return results

    @staticmethod
    def parse_generations(outputs: dict | list[dict], **kwargs) -> list[list[Token]]:
        raise NotImplementedError
