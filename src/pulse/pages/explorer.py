import json
import time

from typing import Final
from pathlib import Path

import pandas as pd
import streamlit as st

from streamlit import logger
from streamlit import session_state as ss
from streamlit.delta_generator import DeltaGenerator

from pulse.utils.paths import COMPLETIONS
from pulse.utils.tools import Placeholder as ph
from pulse.connection.vllm_connection import SampleRequest, VLLMConnection

_LOGGER: Final = logger.get_logger(__name__)

st.title("PULSE - Polling Using LLM-based Sentiment Extraction")
st.subheader("Explorer")

if "vllm_conn" not in ss:
    ss.vllm_conn = None

completions_path = COMPLETIONS / "completions_json"
if "completions" not in ss:
    ss.completions = json.load(open(completions_path))

if "completion" not in ss:
    ss.completion = None


if "credentials" not in ss:
    ss.url = None
    ss.api_key = None
    ss.credentials = {}

# persist start
if "selected_model" in st.session_state:
    st.session_state.selected_model = st.session_state.selected_model

if "description" in st.session_state:
    st.session_state.description = st.session_state.description

if "doc_to_text" in st.session_state:
    st.session_state.doc_to_text = st.session_state.doc_to_text

if "gen_prefix" in st.session_state:
    st.session_state.gen_prefix = st.session_state.gen_prefix

if "selected_completions" in st.session_state:
    st.session_state.selected_completions = st.session_state.selected_completions
# persist stop


def store_key(key: str):
    assert key.startswith("_")
    st_key = key.removeprefix("_")
    _LOGGER.info(f"Storing `{st_key}`")
    ss[st_key] = ss[key]


def persist(key: str) -> dict:
    return {"key": key, "on_change": store_key, "args": (key,)}


def sidebar_connection() -> None:
    with st.form("connection_form"):
        url = st.text_input(
            label="url",
            value=ss.credentials.get("base_url"),
            placeholder="http://localhost:8000",
        )
        api_key = st.text_input(
            label="api_key",
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


def connect(url: str, api_key: str) -> None:
    credentials = {"base_url": url, "token": api_key}

    ss.selected_model = None
    ss.vllm_conn = VLLMConnection("vllm", type=VLLMConnection, **credentials)
    ss.credentials = credentials


def get_models() -> list[str]:
    r: dict = ss.vllm_conn.get_models().json()
    models: list = [d["id"] for d in r["data"]]
    return models


def assign_model() -> None:
    store_key(key="_selected_model")
    # ss.selected_model = ss._selected_model

    ss.vllm_conn.assign_model(ss.selected_model)
    st.toast(f"Assigned model: {ss.selected_model}")
    # rerun()
    ss.completion = None
    ss.rankings = None
    time.sleep(0.5)


with st.sidebar.expander("Connection", expanded=True):
    sidebar_connection()


@st.dialog("Create Completions", width="large")
def create_completions() -> None:
    st.write("Create a new set of completions.")
    name = st.text_input("Name", placeholder="e.g. elections")

    columns = ["a", "b", "alias"]
    completions_df = pd.DataFrame(columns=columns)
    changed = st.data_editor(completions_df, num_rows="dynamic")

    if st.button("Save"):
        if not name:
            st.error("Please provide a name for the completions.")
            st.stop()

        if name in ss.completions:
            st.error(f"Completions '{name}' already exists.")
            st.stop()

        if changed.empty:
            st.error("Empty dataframe. Add some rows.")
            st.stop()

        ss.completions[name] = changed.to_dict(orient="records")
        with open(completions_path, "w") as f:
            json.dump(ss.completions, f, indent=4)
        st.toast("Updated completions.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Edit Completions File", width="large")
def edit_completions(selected_completions) -> None:
    st.write(selected_completions.capitalize())
    completions = ss.completions[selected_completions]
    changed = st.data_editor(completions, num_rows="dynamic")

    if st.button("Save"):
        ss.completions[selected_completions] = changed
        with open(completions_path, "w") as f:
            json.dump(ss.completions, f, indent=4)
        st.toast("Updated completions.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Completions", width="small")
def delete_completions(selected_completions) -> None:
    st.error(f"Are you sure you want to delete the completions '{selected_completions}'?")
    if st.button("Confirm"):
        del ss.completions[selected_completions]
        with open(completions_path, "w") as f:
            json.dump(ss.completions, f, indent=4)
        st.toast("Deleted completions")
        time.sleep(0.5)
        st.rerun()


# def draw_completions() -> None:
#     st.selectbox(label="Select completions", options=ss.completions.keys(), index=None, key="selected_completions")
#     new_col, edit_col, del_col = st.columns(3)
#     new_col.button("New", on_click=create_completions, use_container_width=True)
#     # if a completion is selected, add view/edit & delete btns
#     if selected_completions := ss.get("selected_completions"):
#         edit_col.button(
#             label="View/Edit", on_click=edit_completions, args=(selected_completions,), use_container_width=True
#         )
#         del_col.button(
#             label="Delete", on_click=delete_completions, args=(selected_completions,), use_container_width=True
#         )


@st.fragment()
def prompt_container():
    st.text_input(label="Persona", placeholder=ph.persona, key="description")
    st.text_input(label="Question", placeholder=ph.question, key="doc_to_text")
    answer_col, comp_col = st.columns(2)

    answer_col.text_input(label="Answer", placeholder=ph.answer, key="gen_prefix")
    comp_col.text_input(
        label="completion", placeholder=" the Democrats", key="completion", help="Mind the leading whitespace!"
    )
    # draw_completions()


@st.fragment()
def draw_params():
    max_logprobs = ss.vllm_conn.max_logprobs
    st.number_input(
        label=f"No. of log probs (max: {max_logprobs})",
        min_value=5,
        max_value=max_logprobs,
        value=10,
        step=1,
        key="logprobs",
    )

    add_generation_prompt = False if (ss.gen_prefix or ss.completion) else True

    ss.extra_body = {
        "extra_body": {
            "logprobs": ss.logprobs,
            "add_generation_prompt": add_generation_prompt,
        }
    }


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
    else:
        st.toast("No chat to submit.")
        return None


@st.fragment()
def sample() -> None:
    if chat := get_chat():
        request = SampleRequest(context=chat, continuation=ss.completion or "")
        (prompt,) = ss.vllm_conn.sample(requests=[request], **ss.extra_body)
        ss.sample_df = pd.DataFrame(prompt.next_tokens).set_index("rank")
        # (numRows + 1) * 35 + 3
    else:
        ss.sample_df = None


@st.fragment()
def rank() -> None:
    if not (chat := get_chat()):
        ss.rankings = None
        return

    if not (completions := ss.get("selected_completions")):
        st.toast("No selected completions.")
        ss.rankings = None
        return

    completions = st.session_state.completions[completions]
    rankings = pd.DataFrame(completions).to_dict(orient="list")
    choices = rankings["a"] + rankings["b"]
    rank_requests = [SampleRequest(context=chat, continuation=f" {choice}") for choice in choices]
    prompts = ss.vllm_conn.sample(rank_requests, **ss.extra_body)

    conts = [prompt.continuation.data for prompt in prompts]
    mid = len(conts) // 2
    ss.rankings = tuple(map(pd.DataFrame, [conts[:mid], conts[mid:]]))


if not ss.vllm_conn:
    st.warning("Enter vLLM server credentials.")
    st.stop()

if not ss.get("selected_model"):
    st.warning("Select one of the available models.")
    st.stop()

if ss.vllm_conn.lm.tokenizer.chat_template is None:
    st.error("Selected model has no chat template.")
    st.stop()

prompt_col, params_col, next_col = st.columns((0.425, 0.15, 0.425))

prompt_col.markdown("<div> Prompt </div>", unsafe_allow_html=True)
with prompt_col.container(border=True, height=280):
    prompt_container()


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


with next_col:
    st_md("Next token")
    next_container = next_col.container(border=True, height=420)
    if "sample_df" in ss:
        next_container.dataframe(ss.sample_df, height=415)

with params_col.container(border=False, height=420) as cont:
    st.empty().container(border=False, height=138)
    draw_params()
    st.button(
        label="Sample next token",  # 🕵️‍♂️👉
        on_click=sample,
        use_container_width=True,
    )


if rankings := ss.get("rankings"):
    side_a, side_b = st.columns(2)
    group_a, group_b = rankings
    side_a.dataframe(group_a, hide_index=True)
    side_b.dataframe(group_b, hide_index=True)
