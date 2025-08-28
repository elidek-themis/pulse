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
from streamlit import session_state as ss

from utils.task import PulseResults, ReferendumConfig, PulseMultipleChoice
from utils.vllm_connection import VLLMConnection

# Session state setup start
if "task_config" not in ss:
    ss.task_config = ReferendumConfig()

if "results" not in ss:
    ss.results = set()

personas_path = Path("data") / "personas.json"
if personas_path not in ss:
    ss.personas = json.load(open(personas_path))

completions_path = Path("data") / "completions.json"
if "completions" not in ss:
    ss.completions = json.load(open(completions_path))

if "merged_docs" not in ss:
    ss.merged_docs = None

if "credentials" not in ss:
    ss.url = None
    ss.api_key = None
    ss.credentials = {}
# Session state setup end

# persist start
if "description" in st.session_state:
    st.session_state.description = st.session_state.description

if "doc_to_text" in st.session_state:
    st.session_state.doc_to_text = st.session_state.doc_to_text

if "gen_prefix" in st.session_state:
    st.session_state.gen_prefix = st.session_state.gen_prefix

if "selected_completions" in st.session_state:
    st.session_state.selected_completions = st.session_state.selected_completions
# persist stop


def connect(url: str, api_key: str) -> None:
    credentials = {"base_url": url, "token": api_key}

    ss.vllm_conn = VLLMConnection("vllm", type=VLLMConnection, **credentials)
    ss.credentials = credentials


@st.fragment(run_every=30)
def is_healthy() -> None:
    timestamp = time.time()
    timestamp = datetime.fromtimestamp(timestamp)
    timestamp = timestamp.strftime("%d/%m/%Y - %H:%M:%S")

    try:
        r = ss.vllm_conn.health()
        if r.status_code == 200:  # noqa: PLR2004
            st.write(f"🟢 {timestamp}")
        else:
            st.write(f"🔴 {timestamp}")
    except requests.exceptions.ConnectionError:
        st.write(f"🔴 {timestamp}")


def get_models() -> list[str]:
    r = ss.vllm_conn.get_models().json()
    models = [d["id"] for d in r["data"]]
    return models


def assign_model() -> None:
    ss.selected_model = ss._selected_model

    ss.vllm_conn.assign_model(ss.selected_model)
    st.toast(f"Assigned model: {ss.selected_model}")
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
        if name in ss.personas:
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

        ss.personas[name] = changed.to_dict(orient="records")
        with open(personas_path, "w") as f:
            json.dump(ss.personas, f, indent=4)
        st.toast("Updated personas.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Edit Persona", width="large")
def edit_persona(selected_persona) -> None:
    st.write(selected_persona.capitalize())
    persona = ss.personas[selected_persona]
    changed = st.data_editor(persona, num_rows="dynamic")

    if st.button("Save"):
        ss.personas[selected_persona] = changed
        with open(personas_path, "w") as f:
            json.dump(ss.personas, f, indent=4)
        st.toast("Updated personas.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Persona", width="small")
def delete_persona(selected_persona) -> None:
    st.error(f"Are you sure you want to delete the persona '{selected_persona}'?")
    if st.button("Confirm"):
        del ss.personas[selected_persona]
        with open(personas_path, "w") as f:
            json.dump(ss.personas, f, indent=4)
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


def update_task_config(key) -> None:
    setattr(ss.task_config, key, ss.get(key))


@st.dialog("Task Config")
def show_config() -> None:
    st.write(ss.task_config.to_dict())


def sidebar_connection() -> None:
    with st.form("connection_form"):
        url = st.text_input(
            label="url", value=ss.credentials.get("base_url", None), placeholder="http://localhost:8000"
        )
        api_key = st.text_input(
            label="api_key", value=ss.credentials.get("token", None), placeholder="EMPTY", type="password"
        )

        if st.form_submit_button("Connect"):
            connect(url, api_key)

    if ss.get("vllm_conn", None):
        is_healthy()

        models = get_models()
        index = models.index(ss.selected_model) if ss.get("selected_model", False) else None

        st.selectbox(
            label="Select a model",
            options=models,
            index=index,
            on_change=assign_model,
            key="_selected_model",
            help="Selecting a model allows to access its tokenizer and chat template.",
        )


# def prompt_container() -> None:
#     # st.write("#### Prompts")
#     # st.markdown("<div> Prompt </div>", unsafe_allow_html=True)

#     st.text_input(
#         label="System prompt",
#         value=ss.get("description", ""),
#         placeholder="You are {{ persona }}.",
#         on_change=update_task_config,
#         args=("description",),
#         key="description",
#     )
#     with st.expander("Personas"):
#         batch_container()
#     st.text_input(
#         label="User prompt",
#         value=ss.get("doc_to_text", ""),
#         placeholder="What is your opinion on {{ subject }}?",
#         on_change=update_task_config,
#         args=("doc_to_text",),
#         key="doc_to_text",
#     )
#     st.text_input(
#         label="Assistant prompt",
#         placeholder="I believe that",
#         on_change=update_task_config,
#         args=("gen_prefix",),
#         key="gen_prefix",
#     )


def completions_container() -> None:
    st.selectbox(label="Select completions", options=ss.completions.keys(), index=None, key="selected_completions")
    new_col, edit_col, del_col = st.columns(3)
    new_col.button("New", on_click=create_completions, use_container_width=True)
    # if a completion is selected, add view/edit & delete btns
    if selected_completions := ss.get("selected_completions", None):
        edit_col.button(
            label="View/Edit", on_click=edit_completions, args=(selected_completions,), use_container_width=True
        )
        del_col.button(
            label="Delete", on_click=delete_completions, args=(selected_completions,), use_container_width=True
        )


def batch_container() -> None:
    selected_col, assign_col = st.columns(2)
    with selected_col:
        selected_persona = st.selectbox(
            "Edit Persona File",
            ss.personas.keys(),
            index=None,
        )
        new_c, edit_c, del_c = st.columns(3)
        new_c.button(label="New", on_click=new_persona, key="new_persona", use_container_width=True)
        if selected_persona:
            edit_c.button(
                label="View/Edit",
                on_click=edit_persona,
                args=(selected_persona,),
                key="edit_persona",
                use_container_width=True,
            )
            del_c.button(
                label="Delete",
                on_click=delete_persona,
                args=(selected_persona,),
                key="delete_persona",
                use_container_width=True,
            )
    assign_col.multiselect(
        label="Select personas",
        options=ss.personas.keys(),
        # on_change=setattr(ss, "merged_docs", None),
        key="selected_personas",
    )


@st.dialog("Task Arguments", width="large")
def task_arguments() -> None:
    task = PulseMultipleChoice(config=ss.task_config)
    task.build_all_requests(apply_chat_template=True, chat_template=ss.vllm_conn.model.apply_chat_template)

    arguments = [instance.arguments for instance in task.instances]
    st.write([{"context": ctx, "completion": comp} for ctx, comp in arguments])


@st.dialog("lm-eval", width="small")
def run_task(name: str) -> None:
    with st.spinner(f"Running {name} task "):
        task = PulseMultipleChoice(config=ss.task_config)

        results = evaluate(
            lm=ss.vllm_conn.model,
            task_dict={name: task},
            write_out=True,
            log_samples=True,
            apply_chat_template=True,
            verbosity="INFO",
            confirm_run_unsafe_code=False,
        )

        ss.results.add(PulseResults(model=ss.selected_model, task=name, results=results))
        st.toast(f"Run completed for task '{name}' with model '{ss.selected_model}'")
        time.sleep(0.25)
    st.rerun()


def save() -> None:
    if st.button("Save 💾", use_container_width=True):
        selected_personas = ss.get("selected_personas", None)
        selected_completions = ss.get("selected_completions", None)
        # GUARDS start
        if not ss.task:
            st.toast("Please provide a name for the poll.")
        # TODO: in task_manager
        elif ss.task in {result.task for result in ss.results}:
            st.toast(f"Task '{ss.task}' already exists. Please choose a different name.")
        elif not selected_completions:
            st.toast("Please select a completion set.")
        # GUARDS end
        else:
            completions = ss.completions[selected_completions]
            completions = pd.DataFrame(completions).to_dict(orient="list")

            if selected_personas:
                st.write(selected_personas)
                docs = [ss.personas[persona] for persona in selected_personas]
                all_docs = list(product(*docs))  # cartesian product of input iterables
                st.toast(f"{ss.selected_personas} produced {len(all_docs)} documents.")

                merged_docs = []
                for combo in all_docs:
                    merged_doc = reduce(lambda x, y: x | y, combo)
                    merged_docs.append(merged_doc)
            else:
                merged_docs = None

            ss.task_config.task = ss.task
            ss.task_config.dataset_kwargs.update({"docs": merged_docs})
            ss.task_config.dataset_kwargs.update({"completions": completions})


with st.sidebar.expander("Connection", expanded=True):
    sidebar_connection()
    st.button("Config", on_click=show_config)

st.header("PULSE - Polling Using LLM-based Sentiment Extraction")

# name_col, _ = st.columns([0.4, 0.4])

task_col, batch_col = st.columns([0.4, 0.4])
task_col.subheader("Create a Poll")
with task_col.container(border=True, height=680):
    st.text_input(
        "Name",
        placeholder="e.g. referendum",
        on_change=update_task_config,
        args=("task",),
        key="task",
    )
    st.markdown("<div style='font-size:14px;'>Prompts</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.text_input(
            label="System",
            # value=ss.get("description", ""),
            placeholder="You are {{ persona }}.",
            on_change=update_task_config,
            args=("description",),
            key="description",
        )
        with st.expander("Batch personas"):
            batch_container()
        st.text_input(
            label="User",
            value=ss.get("doc_to_text", ""),
            placeholder="What is your opinion on {{ subject }}?",
            on_change=update_task_config,
            args=("doc_to_text",),
            key="_doc_to_text",
        )
        st.text_input(
            label="Assistant",
            placeholder="I believe that",
            on_change=update_task_config,
            args=("gen_prefix",),
            key="gen_prefix",
        )

    # with st.empty().container(border=True):
    with st.expander("Completions", expanded=True):
        completions_container()

batch_col.subheader("Select Poll")
with batch_col.container(border=True, height=680):
    data = {
        "Task Name": ["Task_1", "Task_2", "Task_3", "Task_4", "Task_5"],
        "Hash": ["53737d94", "27c6f55c", "6e76ade5", "aa640488", "8f60a8bd"],
        "Status": ["Low", "Low", "Done", "Done", "High"],
    }
    df = pd.DataFrame.from_dict(data)
    st.dataframe(df)
    # with st.empty().container(border=True):
    #     batch_container()

save_c, build_c, run_c = st.columns([0.33, 0.33, 0.33])

with save_c:
    save()
    st.write(ss.task_config)
    ss.task_config.to_yaml()

if build_c.button("Build 🛠️", use_container_width=True):
    task_arguments()

if run_c.button("Run 🏃", use_container_width=True):
    run_task(name=ss.task_config.task)
