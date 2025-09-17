import time

from dataclasses import asdict

import streamlit as st

from streamlit import session_state as ss
from streamlit.delta_generator import DeltaGenerator

from pulse.utils.tools import ModelCard
from pulse.data.pulse_task import PulseConfig
from pulse.data.repository import Repository
from pulse.connection.vllm_connection import VLLMConnection


def init_session_state() -> None:
    if "task_config" not in ss:
        ss.task_config = PulseConfig()

    if "repo" not in ss:
        ss.repo = Repository()

    if "credentials" not in ss:
        ss.url = None
        ss.api_key = None
        ss.credentials = {}


def persist_session_state() -> None:
    if "selected_model" in ss:
        # avoid streamlit-dataclass serialization
        ss.selected_model = ModelCard(**asdict(ss.selected_model))

    if "vllm_conn" in ss:
        ss.vllm_conn = ss.vllm_conn

    if "description" in ss:
        ss.description = ss.description

    if "doc_to_text" in ss:
        ss.doc_to_text = ss.doc_to_text

    if "gen_prefix" in ss:
        ss.gen_prefix = ss.gen_prefix

    if "selected_completions" in ss:
        ss.selected_completions = ss.selected_completions

    if "selected_persona" in ss:
        ss.selected_persona = ss.selected_persona


def st_md(
    text: str,
    container: DeltaGenerator | None = None,
    font_size: str = "16px",
    **styles: str,
) -> None:
    styles = {"font-size": font_size, **styles}
    style_str = "; ".join(f"{k}: {v}" for k, v in styles.items() if v is not None)

    target = container if container is not None else st
    target.markdown(
        f"<div style='{style_str}'>{text}</div>",
        unsafe_allow_html=True,
    )


def connect(url: str, api_key: str) -> None:
    credentials = {"base_url": url, "token": api_key}

    ss.vllm_conn = VLLMConnection("vllm", type=VLLMConnection, **credentials)
    # ss.vllm_conn = st.connection(name="vllm", type=VLLMConnection, **credentials)
    ss.credentials = credentials


def get_models() -> list[str]:
    def _parse_model(model: dict) -> ModelCard:
        return ModelCard(id=model.get("id"), root=model.get("root"))

    resp = ss.vllm_conn.get_models().json()

    models = [_parse_model(model) for model in resp.get("data", [])]
    return models


def assign_model() -> None:
    ss.selected_model = ss._selected_model

    # batch_size = st.secrets.sampling.BATCH_SIZE
    ss.vllm_conn.assign_model(model_card=ss.selected_model)
    st.toast(f"Assigned model: {ss.selected_model}")
    time.sleep(0.5)


def sidebar_connection() -> None:
    with st.form("connection_form"):
        url = st.text_input(
            label="URL",
            value=ss.credentials.get("base_url"),
            placeholder="http://localhost:8000",
        )
        api_key = st.text_input(
            label="API Key",
            value=ss.credentials.get("token"),
            placeholder="EMPTY",
            type="password",
        )

        if st.form_submit_button("Connect"):
            connect(url=url, api_key=api_key)

    if ss.get("vllm_conn"):
        models = get_models()
        index = models.index(ss.selected_model) if ss.get("selected_model") else None

        st.selectbox(
            label="Select a model",
            options=models,
            index=index,
            on_change=assign_model,
            key="_selected_model",
        )


def get_chat() -> str | None:
    chat_history = []

    if ss.description:
        chat_history.append({"role": "system", "content": ss.description})

    if ss.doc_to_text:
        chat_history.append({"role": "user", "content": ss.doc_to_text})

    if ss.gen_prefix:
        chat_history.append({"role": "assistant", "content": ss.gen_prefix})

    if chat_history:
        return chat_history

    st.toast("No chat to submit.")
