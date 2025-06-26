import json
import time

from pathlib import Path
from datetime import datetime
from functools import reduce
from itertools import product

import pandas as pd
import requests
import streamlit as st

from lm_eval import evaluate

from utils.task import PulseResults, PulseMultipleChoice, referendum_task
from utils.vllm_connection import VLLMConnection

# Session state setup start
if "task_config" not in st.session_state:
    st.session_state.task_config = referendum_task

if "results" not in st.session_state:
    st.session_state.results = set()

personas_path = Path("data") / "personas.json"
if personas_path not in st.session_state:
    st.session_state.personas = json.load(open(personas_path))

completions_path = Path("data") / "completions.json"
if "completions" not in st.session_state:
    st.session_state.completions = json.load(open(completions_path))

if "merged_docs" not in st.session_state:
    st.session_state.merged_docs = None

if "credentials" not in st.session_state:
    st.session_state.url = None
    st.session_state.api_key = None
    st.session_state.credentials = {}
# Session state setup end


def connect(url: str, api_key: str) -> None:
    credentials = {"base_url": url, "token": api_key}

    st.session_state.vllm_conn = VLLMConnection("vllm", type=VLLMConnection, **credentials)
    st.session_state.credentials = credentials


@st.fragment(run_every=30)
def is_healthy() -> None:
    timestamp = time.time()
    timestamp = datetime.fromtimestamp(timestamp)
    timestamp = timestamp.strftime("%d/%m/%Y - %H:%M:%S")

    try:
        r = st.session_state.vllm_conn.health()
        if r.status_code == 200:  # noqa: PLR2004
            st.write(f"🟢 {timestamp}")
        else:
            st.write(f"🔴 {timestamp}")
    except requests.exceptions.ConnectionError:
        st.write(f"🔴 {timestamp}")


def get_models() -> list[str]:
    r = st.session_state.vllm_conn.get_models().json()
    models = [d["id"] for d in r["data"]]
    return models


def assign_model() -> None:
    st.session_state.selected_model = st.session_state._selected_model

    st.session_state.vllm_conn.assign_model(st.session_state.selected_model)
    st.toast(f"Assigned model: {st.session_state.selected_model}")
    time.sleep(0.5)  # interactivity hack


@st.dialog("New Persona", width="large")
def new_persona() -> None:
    name = st.text_input("Name")
    st.write(name)

    columns = st.text_input("Columns (comma-separated)", placeholder="demographic, group, persona")
    columns = list(map(str.strip, columns.split(",")))
    persona_df = pd.DataFrame(columns=columns)

    changed = st.data_editor(persona_df, num_rows="dynamic")

    if st.button("Save"):
        if name in st.session_state.personas:
            st.error(f"Persona '{name}' already exists.")
            st.stop()

        if not name:
            st.error("Please provide a name for the persona.")
            st.stop()

        if not any(columns):
            st.error("Please provide at least one column name.")
            st.stop()

        if changed.empty:
            st.error("Empty dataframe. Add some rows.")
            st.stop()

        st.session_state.personas[name] = changed.to_dict(orient="records")
        with open(personas_path, "w") as f:
            json.dump(st.session_state.personas, f, indent=4)
        st.toast("Updated personas.json")
        time.sleep(0.5)  # interactivity hack
        st.rerun()  # close the dialog


@st.dialog("Edit Persona", width="large")
def edit_persona(selected_persona) -> None:
    st.write(selected_persona.capitalize())
    persona = st.session_state.personas[selected_persona]
    changed = st.data_editor(persona, num_rows="dynamic")

    if st.button("Save"):
        st.session_state.personas[selected_persona] = changed
        with open(personas_path, "w") as f:
            json.dump(st.session_state.personas, f, indent=4)
        st.toast("Updated personas.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Persona", width="small")
def delete_persona(selected_persona) -> None:
    st.error(f"Are you sure you want to delete the persona '{selected_persona}'?")
    if st.button("Confirm"):
        del st.session_state.personas[selected_persona]
        with open(personas_path, "w") as f:
            json.dump(st.session_state.personas, f, indent=4)
        st.toast("Deleted persona")
        time.sleep(0.5)
        st.rerun()


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


def update_task_config(key) -> None:
    setattr(st.session_state.task_config, key, st.session_state.get(key))


@st.dialog("Task Config")
def show_config() -> None:
    st.write(st.session_state.task_config.to_dict())


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
        st.button("Config", on_click=show_config)


def prompt_container() -> None:
    st.write("### Prompts")

    st.text_input(
        label="System prompt",
        placeholder="You are {{ persona }}.",
        on_change=update_task_config,
        args=("description",),
        key="description",
    )
    st.text_input(
        label="User prompt",
        placeholder="What is your opinion on {{ subject }}?",
        on_change=update_task_config,
        args=("doc_to_text",),
        key="doc_to_text",
    )
    st.text_input(
        label="Assistant prompt",
        placeholder="I believe that",
        on_change=update_task_config,
        args=("gen_prefix",),
        key="gen_prefix",
    )


def completions_container() -> None:
    selected_completions = st.selectbox("Edit completions", st.session_state.completions.keys(), index=None)
    new_c, edit_c, del_c = st.columns([0.33, 0.33, 0.33])
    with new_c:
        st.button("New", on_click=create_completions, key="new_completions", use_container_width=True)
    if selected_completions:
        with edit_c:
            st.button(
                label="View/Edit",
                on_click=edit_completions,
                args=(selected_completions,),
                key="edit_completions",
                use_container_width=True,
            )
        with del_c:
            st.button(
                label="Delete",
                on_click=delete_completions,
                args=(selected_completions,),
                key="delete_completions",
                use_container_width=True,
            )
    st.selectbox(
        label="Select completions", options=st.session_state.completions.keys(), index=None, key="selected_completions"
    )


def batch_container() -> None:
    selected_persona = st.selectbox("Edit Persona File", st.session_state.personas.keys(), index=None)
    new_c, edit_c, del_c = st.columns([0.33, 0.33, 0.33])
    with new_c:
        st.button(label="New", on_click=new_persona, key="new_persona", use_container_width=True)
    if selected_persona:
        with edit_c:
            st.button(
                label="View/Edit",
                on_click=edit_persona,
                args=(selected_persona,),
                key="edit_persona",
                use_container_width=True,
            )
        with del_c:
            st.button(
                label="Delete",
                on_click=delete_persona,
                args=(selected_persona,),
                key="delete_persona",
                use_container_width=True,
            )
    st.multiselect(
        label="Select personas",
        options=st.session_state.personas.keys(),
        # on_change=setattr(st.session_state, "merged_docs", None),
        key="selected_personas",
    )


@st.dialog("Task Arguments", width="large")
def task_arguments() -> None:
    task = PulseMultipleChoice(config=st.session_state.task_config)
    task.build_all_requests(
        apply_chat_template=True, chat_template=st.session_state.vllm_conn.model.apply_chat_template
    )

    arguments = [instance.arguments for instance in task.instances]
    st.write([{"context": ctx, "completion": comp} for ctx, comp in arguments])


@st.dialog("lm-eval", width="small")
def run_task(name: str) -> None:
    with st.spinner(f"Running {name} task "):
        task = PulseMultipleChoice(config=st.session_state.task_config)

        results = evaluate(
            lm=st.session_state.vllm_conn.model,
            task_dict={name: task},
            # cache_requests=True,
            write_out=True,
            log_samples=True,
            apply_chat_template=True,
            verbosity="INFO",
            confirm_run_unsafe_code=False,
        )

        st.session_state.results.add(PulseResults(model=st.session_state.selected_model, task=name, results=results))
        st.toast(f"Run completed for task '{name}' with model '{st.session_state.selected_model}'")
        time.sleep(0.25)
    st.rerun()


def save() -> None:
    if st.button("Save 💾", use_container_width=True):
        selected_personas = st.session_state.get("selected_personas", None)
        selected_completions = st.session_state.get("selected_completions", None)
        # GUARDS start
        if not st.session_state.name:
            st.toast("Please provide a name for the dataset.")
        elif st.session_state.name in {result.task for result in st.session_state.results}:
            st.toast(f"Task '{st.session_state.name}' already exists. Please choose a different name.")
        elif not selected_completions:
            st.toast("Please select a completion set.")
        # GUARDS end
        else:
            completions = st.session_state.completions[selected_completions]
            completions = pd.DataFrame(completions).to_dict(orient="list")

            if selected_personas:
                docs = [st.session_state.personas[persona] for persona in selected_personas]
                all_docs = list(product(*docs))  # cartesian product of input iterables
                st.toast(f"{st.session_state.selected_personas} produced {len(all_docs)} documents.")

                merged_docs = []
                for combo in all_docs:
                    merged_doc = reduce(lambda x, y: x | y, combo)
                    merged_docs.append(merged_doc)
            else:
                merged_docs = None

            st.session_state.task_config.task = st.session_state.name
            st.session_state.task_config.dataset_kwargs.update({"docs": merged_docs})
            st.session_state.task_config.dataset_kwargs.update({"completions": completions})


with st.sidebar:
    with st.expander("Connection", expanded=True):
        sidebar_connection()

st.header("PULSE - Polling Using LLM-based Sentiment Extraction")
st.subheader("Create a Poll")

name_col, _ = st.columns([0.4, 0.4])
with name_col:
    st.text_input("Name", placeholder="e.g. referendum", key="name")

task_col, batch_col = st.columns([0.4, 0.4])
with task_col.container(border=True, height=600):
    prompt_container()

    with st.empty().container(border=True):
        completions_container()

with batch_col.container(border=True, height=600):
    st.write("### Batch Polling")
    with st.empty().container(border=True):
        batch_container()

save_c, build_c, run_c = st.columns([0.33, 0.33, 0.33])

with save_c:
    save()

with build_c:
    if st.button("Build 🛠️", use_container_width=True):
        task_arguments()

with run_c:
    if st.button("Run 🏃", use_container_width=True):
        run_task(name=st.session_state.task_config.task)
