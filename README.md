### PULSE – Polling Using LLM-based Sentiment Extraction (Demo at ICDM 2025)

Packaged with [uv](https://github.com/astral-sh/uv) 

PULSE is build with
* [vLLM](https://github.com/vllm-project/vllm)
* [lm-eval](https://github.com/vllm-project/vllm)
* [Streamlit](https://github.com/streamlit/streamlit)

All dependencies, [pyproject.toml](pyproject.toml)

```bash
# install dependencies
uv sync

# activate venv
source .venv/bin/activate
```

Use the router package to serve multiple models on a single host. \
vLLM [configuration files](data/model/) are provided for reproducibility.
```bash
# i.e. serve on http://localhost:8000 and discover models in 8001-8010
router --host localhost --port 8000 --vllm-port-start 8001 --vllm-port-end 8010
serve --config data/models/llama_3_1_8b_it.yaml --port 8001
serve --config data/models/gemma_2_27b_it.yaml --port 8002
serve --config data/models/mistral_7b_v03.yaml --port 8003
```

Run PULSE \
The Streamlit web interface will automatically launch at: [http://localhost:8501](http://localhost:8501)

```bash
pulse
```
