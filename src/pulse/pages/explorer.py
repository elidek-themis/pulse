from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

from streamlit import session_state as ss

from pulse.pages.state import (
    st_md,
    get_chat,
    init_session_state,
    sidebar_connection,
    persist_session_state,
)
from pulse.utils.tools import Placeholder as ph
from pulse.connection.vllm_connection import SampleRequest

st.header("PULSE - Polling Using LLM-based Sentiment Extraction")

init_session_state()
persist_session_state()

with st.sidebar:
    sidebar_connection()


def prompt_container():
    max_logprobs = ss.vllm_conn.max_logprobs

    with st.form("prompt_form"):
        st.text_input(
            label="Persona",
            placeholder=ph.persona,
            key="description",
        )
        st.text_input(
            label="Question",
            placeholder=ph.question,
            key="doc_to_text",
        )
        answer_col, comp_col = st.columns(2)

        answer_col.text_input(
            label="Answer",
            placeholder=ph.answer,
            key="gen_prefix",
        )
        comp_col.text_input(
            label="completion",
            placeholder=ph.completion,
            key="completion",
            help="Mind the leading whitespace!",
        )
        add_generation_prompt = False if (ss.gen_prefix or ss.completion) else True

        inp_col, btn_col = st.columns(2, vertical_alignment="bottom")

        logprobs = inp_col.number_input(
            label=f"Num. of tokens (max: {max_logprobs})",
            min_value=5,
            max_value=max_logprobs,
            value=128,
            step=1,
        )

        extra_body = {
            "extra_body": {
                "logprobs": logprobs,
                "add_generation_prompt": add_generation_prompt,
                "echo": False,
            }
        }

        btn_col.form_submit_button(
            label="Sample next token",  # 🕵️‍♂️
            on_click=sample,
            args=(extra_body,),
            use_container_width=True,
        )


@st.cache_data
def get_next_tokens(context: list[dict], continuation: str, extra_body: dict) -> pd.DataFrame:
    request = SampleRequest(context=context, continuation=continuation)
    (prompt,) = ss.vllm_conn.sample(requests=[request], **extra_body)
    next_tokens = prompt.next_tokens

    logprobs = np.array([token.logprob for token in next_tokens])
    # softmax - sub max for numerical stability
    probs = np.exp(logprobs - logprobs.max())
    probs /= probs.sum()
    # logprobs already sorted
    cum_sum = np.cumsum(probs)

    df = pd.DataFrame([asdict(token) for token in next_tokens]).set_index("rank")
    df.index.name = "Rank"
    df.rename(columns={"token": "Token"}, inplace=True)
    df["Probability"] = probs
    df["Cumulative Probability"] = cum_sum

    return df.drop("logprob", axis=1)


def sample(extra_body: dict) -> None:
    if chat := get_chat():
        ss.sample_df = get_next_tokens(context=chat, continuation=ss.completion, extra_body=extra_body)
    else:
        ss.sample_df = None


if not ss.get("vllm_conn"):
    st.warning("Enter vLLM server credentials.")
    st.stop()

if not ss.get("selected_model"):
    st.warning("Select one of the available models.")
    st.stop()

if ss.vllm_conn.lm.tokenizer.chat_template is None:
    st.error("Selected model has no chat template.")
    st.stop()

_, column, _ = st.columns((0.2, 0.2, 0.2))
column.subheader("Explorer")

st_md(text="Prompt", container=column)
with column:
    prompt_container()

st_md("Next token", container=column)
next_container = column.container(border=True, height=165)
sample_df: pd.DataFrame = ss.get("sample_df")
if sample_df is not None:
    next_container.dataframe(sample_df, height=415)

st.write("")
