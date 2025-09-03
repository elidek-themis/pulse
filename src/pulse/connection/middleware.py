import enum

from dataclasses import asdict

import torch

from fastapi import FastAPI, Request
from vllm.config import ModelConfig
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class CustomRouteMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/model_config":
            response = await self.get_model_config(request=request)
            return response

        return await call_next(request)

    async def get_model_config(self, request: Request) -> JSONResponse:
        model_config: ModelConfig = request.app.state.vllm_config.model_config

        try:
            model_config_dict = asdict(model_config)
            return JSONResponse(content=serialize(model_config_dict))
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)


def serialize(obj):
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, torch.dtype):
        return str(obj)
    return obj
