import time

from pathlib import Path

import streamlit as st

from streamlit import session_state as ss
from streamlit.delta_generator import DeltaGenerator

from pulse.pages.guard import GUARD
from pulse.data.pulse_task import PulseConfig
from pulse.data.repository import Repository
from pulse.connection.ranker import Ranker
from pulse.connection.vllm_connection import (
    ModelCard,
    HTTPStatus,
    VLLMInstance,
    VLLMConnection,
)

DELAY = 0.5

# Caching


@st.cache_data
def _read_css(file_path: Path) -> str:
    with open(file_path) as f:
        css = f"<style>{f.read()}</style>"
    return css


@st.cache_resource
def get_ranker(
    client: VLLMConnection,
    chat: list[dict[str, str]],
    completions: list[str],
    v_pct: float,
    min_p: float,
) -> Ranker:
    return Ranker(
        vllm=client,
        chat=chat,
        completions=completions,
        v_pct=v_pct,
        min_p=min_p,
    )


def load_css(file_path: Path) -> None:
    css_string = _read_css(file_path)
    st.markdown(css_string, unsafe_allow_html=True)


def init_session_state() -> None:
    if "task_config" not in ss:
        ss.task_config = PulseConfig()

    if "repo" not in ss:
        ss.repo = Repository()

    if "credentials" not in ss:
        ss.url = None
        ss.api_key = None
        ss.credentials = {}

    if "rank_flag" not in ss:
        ss.rank_flag = True


def persist_session_state() -> None:
    if selected_model := ss.get("selected_model"):
        if isinstance(selected_model, dict):
            ss.selected_model = ModelCard.from_dict(selected_model)
        else:
            ss.selected_model = selected_model

    persist_session_keys = (
        "vllm_conn",
        "persona",
        "question",
        "answer",
        "selected_completions",
        "selected_persona",
        "selected_task",
    )

    for key in persist_session_keys:
        if value := ss.get(key):
            setattr(ss, key, value)


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
    code = VLLMConnection.is_alive(url=url)

    if code == HTTPStatus.OK:
        credentials = {"base_url": url, "token": api_key}

        ss.vllm_conn = VLLMConnection(**credentials)
        ss.credentials = credentials
        ss.selected_model = None
        st.toast(f"vLLM connection - {url} - {code}")
    else:
        st.toast(f"{url} - {code}")


def get_models(conn: VLLMConnection) -> tuple[ModelCard]:
    return conn.get_models()


def assign_model() -> None:
    vllm_conn: VLLMConnection = ss.vllm_conn
    selected_model: ModelCard = ss._selected_model
    client: VLLMInstance = vllm_conn.get_vllm_client(model_card=selected_model)

    ss.client = client
    ss.selected_model = selected_model

    st.toast(f"Assigned model: {selected_model}")
    time.sleep(DELAY)


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

        if st.form_submit_button("Connect") and url:
            connect(url=url, api_key=api_key)

    if GUARD.is_connected:
        models = get_models(conn=ss.vllm_conn)
        index = models.index(model) if (model := ss.get("selected_model")) else None

        st.selectbox(
            label="Select a model",
            options=models,
            index=index,
            on_change=assign_model,
            key="_selected_model",
        )


def get_chat() -> list[dict[str, str]] | None:
    if not (clause := GUARD.has_prompts):
        st.toast(clause.msg)
        return None

    return [
        {"role": "system", "content": ss.persona},
        {"role": "user", "content": ss.question},
        {"role": "assistant", "content": ss.answer},
    ]
