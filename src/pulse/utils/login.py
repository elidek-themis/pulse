import os

from huggingface_hub import login


def hf_login():
    if hf_token := os.environ.get("HF_TOKEN"):
        login(token=hf_token)
