import json
import asyncio
import logging

from contextlib import asynccontextmanager

import click
import httpx
import uvicorn

from fastapi import FastAPI, Request, Response, HTTPException

from router.model import VLLMModelList
from router.router import Router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")


router = Router()


async def refresh_router():
    while True:
        try:
            router.refresh()
            logger.info(f"Refreshed router. Found {len(router.model_map)} active vLLM servers.")
        except Exception as e:
            logger.error(f"Error refreshing router: {e}")
        await asyncio.sleep(10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting vLLM Router on port {Router.PORT}")
    logger.info(f"Will discover vLLM servers on ports {list(Router.PORT_RANGE)}")
    refresh_task = asyncio.create_task(refresh_router())

    try:
        yield
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="vLLM Router",
    description="Router for vLLM OpenAI-compatible servers",
    version="1.0.0",
    lifespan=lifespan,
)


async def get_model_from_request(request: Request) -> str | None:
    try:
        body = await request.json()
        return body.get("model")
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


@app.get("/")
async def root():
    return {"message": "vLLM Router", "active_servers": len(router.model_map)}


def manual_refresh():
    router.refresh()
    return {"message": "Router refreshed", "active_servers": len(router.model_map)}


@app.get("/health")
async def health_check():
    return manual_refresh()


@app.post("/refresh")
async def refresh():
    return manual_refresh()


@app.api_route("/v1/models", methods=["GET", "POST"], response_model=VLLMModelList)
async def list_models(request: Request):
    # Model specified - Forward to specific vLLM server
    if model_name := await get_model_from_request(request):
        return await proxy_to_vllm(path="v1/models", request=request, model_name=model_name)

    # No model specified - return aggregated list from router
    return router.models


@app.api_route("/{path:path}", methods=["GET", "POST", "HEAD"])
async def proxy_to_vllm(path: str, request: Request, model_name: str | None = None):
    """Forward request to the correct vLLM server based on model name."""
    if not router.model_map:
        raise HTTPException(status_code=503, detail="No vLLM servers available")

    if not model_name:
        model_name = await get_model_from_request(request)

    if not model_name:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model name required for /{path}. "
                "Specify 'model' in the request body. "
                f"Available models: {router.models}"
            ),
        )

    if not router.model_exists(model_name):
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' not found. Available models: {router.models}"
        )

    target_port = router.get_model_port(model_name)
    if target_port is None:
        raise HTTPException(status_code=503, detail="No backend available for model")

    target_url = f"http://localhost:{target_port}/{path}"

    # TODO: support streaming responses
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=request.method,
            url=target_url,
            headers={k: v for k, v in request.headers.items() if k.lower() != "host"},
            content=await request.body(),
            params=request.query_params,
        )

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.headers.get("content-type"),
    )


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to run the router on")
@click.option("--vllm-port-start", default=8001, type=int, help="Start of vLLM port range")
@click.option("--vllm-port-end", default=8010, type=int, help="End of vLLM port range")
def run(host: str, port: int, vllm_port_start: int, vllm_port_end: int):
    """Start the vLLM router server."""
    Router.PORT = port
    Router.PORT_RANGE = range(vllm_port_start, vllm_port_end + 1)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run()
