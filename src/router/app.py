import json
import asyncio
import logging
import argparse

from contextlib import asynccontextmanager

import httpx
import uvicorn

from fastapi import FastAPI, Request, Response, HTTPException

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


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": router.models}


@app.post("/refresh")
async def manual_refresh():
    router.refresh()
    return {"message": "Router refreshed", "active_servers": len(router.model_map)}


@app.api_route("/{path:path}", methods=["GET", "POST", "HEAD"])
async def proxy_to_vllm(path: str, request: Request):
    """Forward request to the correct vLLM server based on model name."""

    if not router.model_map:
        raise HTTPException(status_code=503, detail="No vLLM servers available")

    # require `model` for all requests that fall through here
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

    # TODO: support streaming responses - no buffering (consume body once)
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


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM Router")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the router on (default: 8000)")
    parser.add_argument("--vllm-port-start", type=int, default=8001, help="Start of vLLM port range (default: 8001)")
    parser.add_argument("--vllm-port-end", type=int, default=8010, help="End of vLLM port range (default: 8010)")
    return parser


def run():
    parser = setup_parser()
    args = parser.parse_args()

    Router.PORT = args.port
    Router.PORT_RANGE = range(args.vllm_port_start, args.vllm_port_end + 1)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    run()
