import json

from enum import Enum
from pprint import pformat
from dataclasses import asdict, is_dataclass

import torch

from fastapi import FastAPI, Request, Response
from matplotlib.path import Path
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware


class VLLMConfigMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.custom_routes = {
            "/config": self.get_config,
            # TODO: chat_template from /tokenizer_info?chat_template
        }

    async def dispatch(self, request: Request, call_next):
        if request.url.path.endswith("_config"):
            response = await self.get_config(request=request)
            return response

        return await call_next(request)

    async def get_config(self, request: Request) -> JSONResponse:
        if not hasattr(self, "config"):
            vllm_config = request.app.state.vllm_config
            self.config = self._serialize_obj(vllm_config)

        group, _ = request.url.path.strip("/").split("_config")
        fields = request.query_params.getlist("fields")

        try:
            if group == "vllm":
                return JSONResponse(content={"vllm_config": self.config})
            else:
                config = self.config.get(group, {})

            if fields:
                filtered = {key: config.get(key, None) for key in fields}
                return JSONResponse(content=filtered)

            return JSONResponse(content=config)

        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)

    # async def redirect_typer():
    #     return RedirectResponse("https://typer.tiangolo.com")

    def _serialize_obj(self, obj):  # noqa: PLR0911
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, torch.device):
            return str(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif is_dataclass(obj):
            return {k: self._serialize_obj(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: self._serialize_obj(v) for k, v in obj.items()}
        elif isinstance(obj, list | tuple):
            return [self._serialize_obj(i) for i in obj]
        else:
            try:
                json.dumps(obj)  # sanity test
                return obj
            except TypeError:
                return str(obj)
