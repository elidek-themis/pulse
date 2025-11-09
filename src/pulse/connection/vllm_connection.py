from http import HTTPStatus
from typing import Final
from functools import cached_property
from dataclasses import field, dataclass

import requests

from streamlit import logger

from pulse.connection.types import (
    Prompt,
    ModelCard,
    SampleRequest,
)
from pulse.connection.vllm_completions import VLLMCompletions

_LOGGER: Final = logger.get_logger(__name__)


@dataclass(frozen=True)
class VLLMInstance:
    base_url: str
    model_card: ModelCard
    token: str = field(default="EMPTY", compare=False, hash=False)
    max_logprobs: int = 20
    seed: int = 2025

    @cached_property
    def model(self) -> str:
        return self.model_card.id

    @cached_property
    def completions_endpoint(self) -> str:
        return f"{self.base_url}/v1/completions"

    @cached_property
    def lm(self) -> VLLMCompletions:
        _LOGGER.info(f"Accessing vLLM model: {self.model_card.id}")
        return VLLMCompletions(
            base_url=self.completions_endpoint,
            api_key=self.token,
            model_card=self.model_card,
            seed=self.seed,
        )

    def sample(self, requests: list[SampleRequest], **kwargs) -> list[Prompt]:
        return self.lm.sample(requests=requests, **kwargs)


@dataclass(frozen=True)
class VLLMConnection:
    base_url: str
    token: str | None = "EMPTY"
    seed: int = 2025

    def __post_init__(self):
        # health check
        pass

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def get_models(self) -> tuple[ModelCard]:
        resp = requests.get(
            f"{self.base_url}/v1/models",
            headers=self.headers,
        )
        resp.raise_for_status()

        data = resp.json().get("data", {})
        return tuple(ModelCard.from_dict(m) for m in data)

    def get_vllm_client(self, model_card: ModelCard) -> VLLMInstance:
        max_logprobs = self._get_max_logprobs(model_card=model_card)

        return VLLMInstance(
            base_url=self.base_url,
            token=self.token,
            model_card=model_card,
            max_logprobs=max_logprobs,
            seed=self.seed,
        )

    def _get_max_logprobs(self, model_card: ModelCard) -> int | None:
        resp = self._get_model_config(payload=model_card.payload)
        model_config = resp.json()

        return model_config.get("max_logprobs", 20)

    def _get_model_config(self, payload: dict) -> requests.Response:
        resp = requests.get(
            f"{self.base_url}/model_config",
            json=payload,
            headers=self.headers,
        )
        if not resp.ok:
            _LOGGER.warning(  # noqa: PLE1205
                "/model_config endpoint not enabled,",
                "serve with --middleware router.CustomRouteMiddleware",
            )
        return resp

    @staticmethod
    def is_alive(url: str) -> int:
        health_endpoint = url.rstrip("/") + "/health"

        try:
            resp = requests.get(url=health_endpoint, timeout=5)
            return resp.status_code
        except requests.RequestException:
            return HTTPStatus.NOT_FOUND
