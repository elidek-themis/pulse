import time

import pandas as pd
import streamlit as st

from lm_eval import evaluate
from streamlit import session_state as ss

from pulse.pages.state import (
    st_md,
    get_chat,
    init_session_state,
    sidebar_connection,
    persist_session_state,
)
from pulse.utils.tools import apply_html
from pulse.data.pulse_task import PulseTask
from pulse.data.file_manager import FileStatus
from pulse.data.task_manager import TaskStatus
from pulse.connection.sampler import (
    get_elbows,
    get_position_df,
    get_rankings_df,
    get_completions_metrics,
)

init_session_state()
persist_session_state()


@st.dialog("Create persona", width="large")
def new_persona() -> None:
    name = st.text_input("Name")
    uploaded_files = st.file_uploader("Upload persona file", type=["csv", "json"])

    if uploaded_files:
        # add checks
        _, ext = uploaded_files.name.split(".")
        persona_df = pd.read_json(uploaded_files) if ext == "json" else pd.read_csv(uploaded_files)
    else:
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
    name = st.text_input("Name")
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


def completions_container() -> None:
    st.selectbox(
        label="Select completions",
        options=ss.repo.all_completions,
        index=None,
        on_change=update_dataset_kwargs_completions,
        key="selected_completions",
    )
    new_col, edit_col, del_col = st.columns(3)
    new_col.button("Create", on_click=create_completions, use_container_width=True)
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
        label="Select personas",
        options=ss.repo.all_personas,
        index=None,
        on_change=update_dataset_kwargs_docs,
        key="selected_persona",
    )
    new_col, edit_col, del_col = st.columns(3)
    new_col.button(label="Create", on_click=new_persona, key="new_persona", use_container_width=True)
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
        st.toast(f"Run completed for task '{name}' with model '{ss.selected_model}'")
        time.sleep(0.4)
    st.rerun()


@st.dialog("Save Task", width="small")
def save() -> None:
    st.text_input("Poll Name", key="task")

    if not ss.task:
        st.toast("Please provide a name for the poll.")
    elif not ss.get("selected_completions"):
        st.toast("Please select a completion set.")

    else:
        update_task_config("task")
        status = ss.repo.task_manager.add(task_config=ss.task_config)
        if status == TaskStatus.OK:
            st.toast("Task saved successfully 👌.")
            time.sleep(0.5)
            st.rerun()
        else:
            st.toast(f"Save failed with status: {status}")


@st.dialog("Task Config", width="large")
def show_config() -> None:
    task_config = ss.selected_task
    st.json(ss.repo.task_manager[task_config].to_json())


with st.sidebar:
    sidebar_connection()


st.header("PULSE - Polling Using LLM-based Sentiment Extraction")

task_col, comp_col = st.columns(2)
task_col.markdown("#### Create a Poll")
comp_col.markdown("#### Completion Analysis")
comp_cont = comp_col.container(border=True, height=800)

with task_col.container(border=True, height=800):
    tasks = ss.repo.task_manager.tasks
    t_col, btn_col = st.columns((0.4, 0.6), vertical_alignment="bottom")
    t_col.selectbox(
        label="Polls",
        options=tasks,
        index=None,
        placeholder="Choose a Poll",
        key="selected_task",
    )
    with btn_col:
        save_col, run_col, del_col = st.columns(3)
        if save_col.button("Save", use_container_width=True, key="save_task"):
            save()
        if run_col.button("Run", use_container_width=True, key="run_task"):
            run_task(name=ss.selected_task)
        if del_col.button("Delete", use_container_width=True, key="delete_task"):
            ss.repo.task_manager.delete(task_name=ss.selected_task)
            st.rerun()

    st_md(text="Prompts")
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
            placeholder="What will you vote for in the 2024 U.S. presidential election?",
            on_change=update_task_config,
            args=("doc_to_text",),
            key="doc_to_text",
        )
        st.text_input(
            label="Answer",
            placeholder="I will vote for",
            on_change=update_task_config,
            args=("gen_prefix",),
            key="gen_prefix",
        )

    with st.container(border=True):
        completions_container()


if st.button("Rank completions"):
    with comp_cont:
        if not (chat := get_chat()):
            st.warning("Provide a prompt to analyze.")
            st.stop()

        if not (completions := ss.get("selected_completions")):
            st.warning("Select a completions set to analyze.")
            st.stop()

        completions = ss.repo.completions[completions].to_dict(orient="list")
        completions = completions["A"] + completions["B"]

        elbows = get_elbows(
            lm=ss.vllm_conn.lm,
            context=chat,
            completions=completions,
            v_size=ss.vllm_conn.max_logprobs,
            v_pct=st.secrets.V_PCT,
            min_p=st.secrets.MIN_P,
        )
        metrics = get_completions_metrics(lm=ss.vllm_conn.lm, context=chat, completions=completions)
        A_df, B_df = get_rankings_df(metrics=metrics, elbows=elbows)

        A_pos = get_position_df(rankings=A_df)
        B_pos = get_position_df(rankings=B_df)

        st_md(text="Side A", font_size="18px", **{"text-align": "center"})
        with st.container(border=False, height=350):
            st.table(apply_html(styler=A_pos, cell_text_color="white"))

        st_md(text="Side B", font_size="18px", **{"text-align": "center"})
        with st.container(border=False, height=350):
            st.table(apply_html(styler=B_pos, cell_text_color="white"))
