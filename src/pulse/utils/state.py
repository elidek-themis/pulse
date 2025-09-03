import json

from pathlib import Path

from streamlit import session_state as ss

if "vllm_conn" not in ss:
    ss.vllm_conn = None

completions_path = Path("data") / "completions.json"
if "completions" not in ss:
    ss.completions = json.load(open(completions_path))

if "completion" not in ss:
    ss.completion = None

if "credentials" not in ss:
    ss.url = None
    ss.api_key = None
    ss.credentials = {}

# # class StateWrapper:
