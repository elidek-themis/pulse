from typing import Final

import requests

from openai import OpenAI
from streamlit import logger
from openai.resources.chat import Chat
from streamlit.connections import BaseConnection
from openai.resources.models import Models
from openai.resources.completions import Completions
from lm_eval.models.openai_completions import LocalCompletionsAPI

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
            self.base_url = self._secrets["base_url"]
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

    def assign_model(self, model: str) -> None:
        base_url = f"{self._base_url}/v1/completions"
        self.model = LocalCompletionsAPI(base_url=base_url, model=model)

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
        self._client = OpenAI(base_url=self._base_url, api_key=value)

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
