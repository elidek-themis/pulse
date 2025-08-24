import json
import math
from pathlib import Path
import time

from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from annotated_text import annotation, annotated_text

from utils.tools import format_jinja, Placeholder as ph
from utils.vllm_connection import VLLMConnection

st.title("PULSE - Polling Using LLM-based Sentiment Extraction")
st.subheader("Explorer")

if "vllm_conn" not in st.session_state:
    st.session_state.vllm_conn = None

completions_path = Path("data") / "completions.json"
if "completions" not in st.session_state:
    st.session_state.completions = json.load(open(completions_path))

if "completion" not in st.session_state:
    st.session_state.completion = None

if "credentials" not in st.session_state:
    st.session_state.url = None
    st.session_state.api_key = None
    st.session_state.credentials = {}


def sidebar_connection() -> None:
    with st.form("connection_form"):
        url = st.text_input(
            label="url", value=st.session_state.credentials.get("base_url", None), placeholder="http://localhost:8000"
        )
        api_key = st.text_input(
            label="api_key", value=st.session_state.credentials.get("token", None), placeholder="EMPTY", type="password"
        )

        if st.form_submit_button("Connect"):
            connect(url, api_key)

    if st.session_state.get("vllm_conn", None):
        is_healthy()

        models = get_models()
        index = models.index(st.session_state.selected_model) if st.session_state.get("selected_model", False) else None

        st.selectbox(
            label="Select a model",
            options=models,
            index=index,
            on_change=assign_model,
            key="_selected_model",
            help="Selecting a model allows to access its tokenizer and chat template.",
        )


@st.fragment(run_every=30)
def is_healthy() -> bool:
    timestamp = time.time()
    timestamp = datetime.fromtimestamp(timestamp)
    timestamp = timestamp.strftime("%d/%m/%Y - %H:%M:%S")

    try:
        r = st.session_state.vllm_conn.health()
        if r.status_code == 200:  # noqa: PLR2004
            st.write(f"🟢 {timestamp}")
            return True
        else:
            st.write(f"🔴 {timestamp}")
            return False
    except requests.exceptions.vllm_ConnectionError:
        st.write(f"🔴 {timestamp}")
        return False


def get_models() -> list[str]:
    r = st.session_state.vllm_conn.get_models().json()
    models = [d["id"] for d in r["data"]]
    return models


def assign_model() -> None:
    st.session_state.selected_model = st.session_state._selected_model

    st.session_state.vllm_conn.assign_model(st.session_state.selected_model)
    st.toast(f"Assigned model: {st.session_state.selected_model}")
    time.sleep(0.5)  # interactivity hack


def connect(url: str, api_key: str) -> None:
    credentials = {"base_url": url, "token": api_key}

    st.session_state.vllm_conn = VLLMConnection("vllm", type=VLLMConnection, **credentials)
    st.session_state.credentials = credentials


def submit() -> None:
    if not st.session_state.get("selected_model", False):
        st.error("No selected model")

    if st.session_state.prompt:
        st.session_state.completion = st.session_state.vllm_conn.completions.create(
            model=st.session_state.selected_model,
            prompt=st.session_state.prompt,
            logprobs=20,
            max_tokens=1,
            temperature=0,
            extra_body={
                "prompt_logprobs": 20,
            },
        )
    else:
        st.session_state.completion = None
        st.toast("Oops")
        time.sleep(0.75)


def get_hex_color(value: float) -> str:
    rgba = plt.get_cmap("RdYlGn")(value)
    return mcolors.rgb2hex(rgba)  # Convert to HEX (e.g., '#a6d96a')


def tokenize():
    chat_history = []

    if st.session_state.description:
        chat_history.append({"role": "system", "content": st.session_state.description})

    if st.session_state.doc_to_text:
        chat_history.append({"role": "user", "content": st.session_state.doc_to_text})

    if st.session_state.gen_prefix:
        chat_history.append({"role": "assistant", "content": st.session_state.gen_prefix})

    if chat_history:
        tok = st.session_state.vllm_conn.model.apply_chat_template(
            chat_history=chat_history, add_generation_prompt=st.session_state.gen_prompt
        )

        st.session_state.prompt = tok
        submit()
    else:
        st.toast("No chat history to submit.")
        st.session_state.completion = None


def has_chat_template() -> bool:
    if st.session_state.vllm_conn.model.tokenizer.chat_template is not None:
        return True
    return False


@st.dialog("asd")
def chat_template_dialog():
    st.markdown("### Chat Template")
    template = st.session_state.vllm_conn.model.tokenizer.chat_template
    fmt_template = format_jinja(template)
    st.code(fmt_template.strip(), height=508, language="jinja2")


with st.sidebar.expander("Connection", expanded=True):
    sidebar_connection()
    # TODO: if has_chat_template(): st.button("Chat template", on_click=chat_template_dialog)


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

        if name in st.session_state.completions:
            st.error(f"Completions '{name}' already exists.")
            st.stop()

        if changed.empty:
            st.error("Empty dataframe. Add some rows.")
            st.stop()

        st.session_state.completions[name] = changed.to_dict(orient="records")
        with open(completions_path, "w") as f:
            json.dump(st.session_state.completions, f, indent=4)
        st.toast("Updated completions.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Edit Completions File", width="large")
def edit_completions(selected_completions) -> None:
    st.write(selected_completions.capitalize())
    completions = st.session_state.completions[selected_completions]
    changed = st.data_editor(completions, num_rows="dynamic")

    if st.button("Save"):
        st.session_state.completions[selected_completions] = changed
        with open(completions_path, "w") as f:
            json.dump(st.session_state.completions, f, indent=4)
        st.toast("Updated completions.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Completions", width="small")
def delete_completions(selected_completions) -> None:
    st.error(f"Are you sure you want to delete the completions '{selected_completions}'?")
    if st.button("Confirm"):
        del st.session_state.completions[selected_completions]
        with open(completions_path, "w") as f:
            json.dump(st.session_state.completions, f, indent=4)
        st.toast("Deleted completions")
        time.sleep(0.5)
        st.rerun()


if st.session_state.vllm_conn and st.session_state.get("selected_model", False):
    text_col, next_col, template_col = st.columns((0.3, 0.2, 0.6))
    text_col.markdown("<div> Prompt </div>", unsafe_allow_html=True)
    with text_col.container(border=True, height=420):
        st.text_input(
            label="System prompt",
            placeholder="You are a citizen of the United States of America.",
            key="description",
        )
        st.text_input(
            label="User prompt",
            placeholder="What is your opinion on abortion?",
            key="doc_to_text",
        )
        st.text_input(
            label="Assistant prompt",
            placeholder="I believe that",
            key="gen_prefix",
        )

        selected_completions = st.selectbox("Select completions", st.session_state.completions.keys(), index=None)
        new_c, edit_c, del_c = st.columns([0.33, 0.33, 0.33])
        new_c.button("New", on_click=create_completions, use_container_width=True)
        if selected_completions:
            edit_c.button(
                label="View/Edit",
                on_click=edit_completions,
                args=(selected_completions,),
                key="edit_completions",
                use_container_width=True,
            )
            del_c.button(
                label="Delete",
                on_click=delete_completions,
                args=(selected_completions,),
                key="delete_completions",
                use_container_width=True,
            )
    next_col.markdown("<div> Next token hyperparameters </div>", unsafe_allow_html=True)
    with next_col.container(border=True, height=420):
        st.slider(
            label="temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.05,
            key="temp",
            help=ph.temperature,
        )
        st.slider(
            label="top-k (token cutoff)",
            min_value=-1,
            max_value=1024,
            value=-1,
            step=1,
            key="top_k",
            help=ph.top_k,
        )
        st.slider(
            label="min-p",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
            key="min_p",
            help=ph.min_p,
        )
        st.slider(
            label="top-p (nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.01,
            key="top_p",
            help=ph.top_p,
        )
        st.slider(
            label="logprobs",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
            key="logprobs",
            help=ph.top_p,
        )
        st.checkbox(
            label="add_generation_prompt",
            value=True,
            key="gen_prompt",
            help=ph.add_gen_prompt,
        )
    with template_col:
        st.markdown("<div> Hyperparameters </div>", unsafe_allow_html=True)
        params_container = st.container(border=True, height=420)

    btn_col, sample_col, gen_col = st.columns(3)
    btn_col.button("Sample", on_click=tokenize, use_container_width=True)
    sample_col.button("Generate", on_click=tokenize, use_container_width=True)
    gen_col.button("Rank", on_click=tokenize, use_container_width=True)

    if st.session_state.completion:
        (next_token,) = st.session_state.completion.choices
        (next_logprobs,) = next_token.logprobs.top_logprobs
        prompt_logprobs = next_token.prompt_logprobs

        encoded_prompt = []
        for prompt in prompt_logprobs[1:]:
            token_id = next(iter(prompt))
            encoded_prompt.append({"token": token_id} | prompt[token_id])

        encoded_prompt_df = pd.DataFrame(encoded_prompt)
        encoded_prompt_df["probability"] = encoded_prompt_df.logprob.apply(lambda x: math.exp(x))

        annotations = []
        for _, row in encoded_prompt_df.iterrows():
            token = row.decoded_token
            rank = f"r: {row['rank']}"
            prob = row.probability
            color = get_hex_color(prob)
            annotations.append(annotation(token, rank, color="black", border=f"; background: {color}"))
        annotated_text(*annotations)

        encoded_prompt_df = encoded_prompt_df.set_index("decoded_token").T

        next_tab, prompt_tab = st.tabs(["next token", "prompt"])
        with next_tab:
            next_logprobs = pd.Series(next_logprobs)
            next_logprobs = next_logprobs.to_frame("logprob").reset_index()
            next_logprobs["probability"] = next_logprobs.logprob.apply(lambda x: math.exp(x))
            st.dataframe(next_logprobs)
        with prompt_tab:
            st.dataframe(encoded_prompt)


tost = st.session_state.get("tost", None)
st.write(tost)
