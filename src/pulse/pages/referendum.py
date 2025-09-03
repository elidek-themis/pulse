import time

from dataclasses import asdict

import pandas as pd
import streamlit as st

from lm_eval import evaluate
from streamlit import session_state as ss

from pulse.data.pulse_task import PulseTask, PulseConfig, PulseResults
from pulse.data.repository import Repository
from pulse.data.file_manager import FileStatus
from pulse.data.task_manager import TaskStatus
from pulse.connection.vllm_connection import VLLMConnection

# Session state setup start
if "task_config" not in ss:
    ss.task_config = PulseConfig()

if "results" not in ss:
    ss.results = set()

if "repo" not in ss:
    ss.repo = Repository()

if "merged_docs" not in ss:
    ss.merged_docs = None

if "credentials" not in ss:
    ss.url = None
    ss.api_key = None
    ss.credentials = {}
# Session state setup end

# persist start
if "selected_model" in ss:
    ss.selected_model = ss.selected_model

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
# persist stop


def connect(url: str, api_key: str) -> None:
    credentials = {"base_url": url, "token": api_key}

    ss.vllm_conn = VLLMConnection("vllm", type=VLLMConnection, **credentials)
    ss.credentials = credentials


def get_models() -> list[str]:
    r = ss.vllm_conn.get_models().json()
    models = [d["id"] for d in r["data"]]
    return models


def assign_model() -> None:
    ss.selected_model = ss._selected_model

    ss.vllm_conn.assign_model(ss.selected_model)
    st.toast(f"Assigned model: {ss.selected_model}")
    time.sleep(0.5)  # interactivity hack


@st.dialog("Create persona", width="large")
def new_persona() -> None:
    name = st.text_input("Name")
    st.file_uploader("Upload completions", type=["csv", "json"])

    columns = st.text_input("Columns (comma-separated)", placeholder="demographic, group, persona")
    columns = list(map(str.strip, columns.split(",")))
    persona_df = pd.DataFrame(columns=columns)

    changed = st.data_editor(persona_df, num_rows="dynamic")

    if st.button("Save"):
        if name in ss.repo.personas:
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

        status = ss.repo.personas.add(name=name, df=changed)
        if status == FileStatus.OK:
            st.toast(f"Created persona {name}")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error(f"Failed with status: {status.value}")


@st.dialog("Edit Persona", width="large")
def edit_persona(selected_persona) -> None:
    st.write(selected_persona.capitalize())
    persona = ss.repo.personas[selected_persona].df
    changed = st.data_editor(persona, num_rows="dynamic")

    if st.button("Save"):
        ss.repo.personas.update(name=selected_persona, df=changed)
        st.toast("Updated personas.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Persona", width="small")
def delete_persona(selected_persona) -> None:
    st.error(f"Are you sure you want to delete the persona '{selected_persona}'?")
    if st.button("Confirm"):
        ss.repo.personas.delete(name=selected_persona)
        ss.selected_persona = None
        st.toast("Deleted persona")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Create completions", width="large")
def create_completions() -> None:
    name = st.text_input("Name", placeholder="e.g. elections")
    st.file_uploader("Upload completions", type=["csv", "json"])

    columns = ["A", "B", "alias"]
    completions_df = pd.DataFrame(columns=columns)
    changed = st.data_editor(completions_df, num_rows="dynamic")

    if st.button("Save"):
        if not name:
            st.error("Please provide a name for the completions.")
            st.stop()

        if name in ss.repo.completions:
            st.error(f"Completions '{name}' already exists.")
            st.stop()

        if changed.empty:
            st.error("Empty dataframe. Add some rows.")
            st.stop()

        status = ss.repo.completions.add(name=name, df=changed)
        if status == FileStatus.OK:
            st.toast(f"Created completions {name}")
            time.sleep(0.5)
            st.rerun()
        else:
            st.error(f"Failed with status: {status.value}")


@st.dialog("Edit Completions File", width="large")
def edit_completions(selected_completions) -> None:
    st.write(selected_completions.capitalize())
    completions = ss.repo.completions[selected_completions].df
    changed = st.data_editor(completions, num_rows="dynamic")

    if st.button("Save"):
        ss.repo.completions.update(name=selected_completions, df=changed)
        st.toast("Updated completions.json")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Completions", width="small")
def delete_completions(selected_completions) -> None:
    st.error(f"Are you sure you want to delete the completions '{selected_completions}'?")
    if st.button("Confirm"):
        ss.repo.completions.delete(name=selected_completions)
        ss.selected_completions = None
        st.toast("Deleted completions")
        time.sleep(0.5)
        st.rerun()


def update_task_config(key: str) -> None:
    setattr(ss.task_config, key, ss.get(key))


def update_dataset_kwargs_docs() -> None:
    selection = ss.get("selected_persona")
    personas = ss.repo.personas[selection].to_dict(orient="records")
    ss.task_config.dataset_kwargs.update({"docs": personas})


def update_dataset_kwargs_completions() -> None:
    selection = ss.get("selected_completions")
    completions = ss.repo.completions[selection].to_dict(orient="list")
    ss.task_config.dataset_kwargs.update({"completions": completions})


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
            connect(url, api_key)

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


def completions_container() -> None:
    st.selectbox(
        label="Select completions",
        options=ss.repo.all_completions,
        index=None,
        on_change=update_dataset_kwargs_completions,
        key="selected_completions",
    )
    new_col, edit_col, del_col = st.columns(3)
    new_col.button("New/Upload", on_click=create_completions, use_container_width=True)
    # if a completion is selected, add view/edit & delete btns
    if selected_completions := ss.get("selected_completions"):
        edit_col.button(
            label="View/Edit", on_click=edit_completions, args=(selected_completions,), use_container_width=True
        )
        del_col.button(
            label="Delete", on_click=delete_completions, args=(selected_completions,), use_container_width=True
        )


def batch_container() -> None:
    st.selectbox(
        label="Edit Persona File",
        options=ss.repo.all_personas,
        index=None,
        on_change=update_dataset_kwargs_docs,
        key="selected_persona",
    )
    new_col, edit_col, del_col = st.columns(3)
    new_col.button(label="New/Create", on_click=new_persona, key="new_persona", use_container_width=True)
    if selected_persona := ss.get("selected_persona"):
        edit_col.button(
            label="View/Edit",
            on_click=edit_persona,
            args=(selected_persona,),
            key="edit_persona",
            use_container_width=True,
        )
        del_col.button(
            label="Delete",
            on_click=delete_persona,
            args=(selected_persona,),
            key="delete_persona",
            use_container_width=True,
        )


@st.dialog("lm-eval", width="small")
def run_task(name: str) -> None:
    # if not exists

    with st.spinner(f"Running {name} task "):
        task = ss.repo.task_manager[ss.selected_task]
        task = PulseTask(config=task.to_eval_dict())

        results = evaluate(
            lm=ss.vllm_conn.lm,
            task_dict={name: task},
            write_out=True,
            log_samples=True,
            apply_chat_template=True,
            verbosity="INFO",
            confirm_run_unsafe_code=False,
        )

        ss.repo.add_results(model=ss.vllm_conn.lm.model, results=results)
        # ss.results.add()
        st.toast(f"Run completed for task '{name}' with model '{ss.selected_model}'")
        time.sleep(0.4)
    st.rerun()


def save() -> None:
    # GUARDS start
    if not ss.task:
        st.toast("Please provide a name for the poll.")
    elif not ss.get("selected_completions"):
        st.toast("Please select a completion set.")
    else:
        if selected_persona := ss.get("selected_persona"):
            ss.task_config.num = ss.repo.personas[selected_persona].num
            ss.task_config.alpha_num = ss.repo.personas[selected_persona].alpha_num

        status = ss.repo.task_manager.add(task_config=ss.task_config)
        if status == TaskStatus.OK:
            st.toast("Task saved successfully 👌.")
            st.rerun()
        else:
            st.toast(f"Save failed with status: {status}")


@st.dialog("Task Config", width="large")
def show_config() -> None:
    task_config = ss.selected_task
    st.json(ss.repo.task_manager[task_config].to_json())


with st.sidebar:
    with st.expander("Connection", expanded=True):
        sidebar_connection()
    st.divider()
    tasks = ss.repo.task_manager.tasks
    st.selectbox(
        label=f"Tasks ({len(tasks)})",
        options=tasks,
        index=None,
        key="selected_task",
    )
    run_col, view_col, del_col = st.columns(3)
    run_col.button("Run", use_container_width=True)
    if view_col.button("View", use_container_width=True):
        show_config()
    del_col.button("Delete", use_container_width=True)

st.header("PULSE - Polling Using LLM-based Sentiment Extraction")

task_col, batch_col = st.columns(2)
task_col.markdown("#### Create a Poll")
with task_col.container(border=True, height=680):
    st.text_input(
        "Name",
        placeholder="e.g. referendum",
        on_change=update_task_config,
        args=("task",),
        key="task",
    )
    st.markdown("<div style='font-size:16px;'>Prompts</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.text_input(
            label="Persona",
            placeholder="You are {{ persona }}.",
            on_change=update_task_config,
            args=("description",),
            key="description",
        )
        with st.expander("Batch personas"):
            batch_container()
        st.text_input(
            label="Question",
            placeholder="What is your opinion on {{ subject }}?",
            on_change=update_task_config,
            args=("doc_to_text",),
            key="doc_to_text",
        )
        st.text_input(
            label="Answer",
            placeholder="I believe that",
            on_change=update_task_config,
            args=("gen_prefix",),
            key="gen_prefix",
        )

    with st.expander("Completions", expanded=True):
        completions_container()


batch_col.markdown("#### ?")
with batch_col.container(border=True, height=680):
    pass


if task_col.button("Save 💾", use_container_width=True):
    save()


if batch_col.button("Run 🏃", use_container_width=True):
    run_task(name=ss.selected_task)
