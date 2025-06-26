### PULSE – Polling Using LLM-based Sentiment Extraction (Demo at CIKM 2025)

Packaged with [uv](https://github.com/astral-sh/uv) \
For dependencies, see [pyproject.toml](pyproject.toml)

```bash
# install dependencies
uv sync | uv sync --extra dev

# activate venv
source .venv/bin/activate

# run the app
pulse
```

You can serve a model locally using [vLLM](https://docs.vllm.ai/en/stable/) by running:
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host localhost \
  --port 8000 \
  --gpu_memory_utilization 0.8 \
  --max_model_len 1024 \
```
or connect to an OpenAI compatible server.
