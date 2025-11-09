import time

from typing import Any, Literal, TypedDict
from collections.abc import Callable

import pandas as pd
import streamlit as st

from lm_eval import evaluate
from streamlit import session_state as ss
from streamlit.logger import get_logger
from streamlit.delta_generator import DeltaGenerator
from streamlit.runtime.uploaded_file_manager import UploadedFile

from pulse.pages.guard import GUARD
from pulse.pages.state import (
    DELAY,
    st_md,
    get_chat,
    get_ranker,
    init_session_state,
    sidebar_connection,
    persist_session_state,
)
from pulse.utils.tools import Placeholder as ph, apply_html
from pulse.data.pulse_task import PulseTask, PulseConfig
from pulse.data.repository import Repository
from pulse.pages.shortcuts import activate_shortcuts
from pulse.connection.types import ModelCard
from pulse.connection.ranker import Ranker
from pulse.data.file_manager import FileStatus, FileManager
from pulse.data.task_manager import TaskStatus, TaskManager
from pulse.connection.sampler import get_position_table

# region CONFIGURATION
logger = get_logger(__name__)
HEADER = "PULSE - Polling Using LLM-based Sentiment Extraction"
HEIGHT = 800
init_session_state()
persist_session_state()
# endregion


# region HELPERS
class InputKwargs(TypedDict):
    """Kwargs for st.text_input with on_change callback."""

    on_change: Callable[[str], None]
    args: tuple[str]
    key: str


def input_args(key: str, session_key: str | None = None) -> InputKwargs:
    return {
        "on_change": update_task_config,
        "args": (key, session_key) if session_key else (key, key),
        "key": session_key if session_key else key,
    }


def update_task_config(key: str, session_key: str) -> None:
    """Updates task config attribute from session state."""
    setattr(ss.task_config, key, ss.get(session_key))


def _load_dataframe_from_file(uploaded_file: UploadedFile) -> pd.DataFrame:
    """Loads a DataFrame from an uploaded CSV or JSON file."""
    if uploaded_file.name.endswith(".json"):
        return pd.read_json(uploaded_file)
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    raise ValueError("Unsupported file type. Use CSV or JSON.")


def _validate_entity_creation(
    name: str,
    entity_type: Literal["personas", "completions"],
    repo_accessor: FileManager,
    changed_df: pd.DataFrame,
    columns: list[str] | None,
    has_default_columns: bool,
) -> None:
    """Validate entity creation inputs and raise errors if invalid."""
    if not name:
        st.error(f"Please provide a name for the {entity_type}.")
        st.stop()

    if name in repo_accessor:
        st.error(f"{entity_type.capitalize()} '{name}' already exists.")
        st.stop()

    if not has_default_columns and columns and not any(columns):
        st.error("Please provide at least one column name.")
        st.stop()

    if changed_df.empty:
        st.error("Empty dataframe. Add some rows.")
        st.stop()


# endregion


# region CRUD
def create_entity(
    entity_type: Literal["personas", "completions"],
    repo_accessor: FileManager,
    default_columns: list[str] | None = None,
    column_placeholder: str = "",
) -> None:
    """Generic function to create new personas or completions."""
    name = st.text_input("Name")
    uploaded_file = st.file_uploader(f"Upload {entity_type} file", type=("csv", "json"))

    # Load or create DataFrame
    if uploaded_file:
        try:
            df = _load_dataframe_from_file(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()
    else:
        if default_columns:
            columns = default_columns
        else:
            columns_input = st.text_input("Columns (comma-separated)", placeholder=column_placeholder)
            columns = list(map(str.strip, columns_input.split(",")))
        df = pd.DataFrame(columns=columns)

    changed = st.data_editor(df, num_rows="dynamic")

    if st.button("Save"):
        _validate_entity_creation(
            name,
            entity_type,
            repo_accessor,
            changed,
            columns if not uploaded_file else None,
            bool(default_columns),
        )

        status = repo_accessor.add(name=name, df=changed.reset_index(drop=True))
        if status == FileStatus.OK:
            st.toast(f"Created {entity_type} '{name}'")
            time.sleep(DELAY)
            st.rerun()
        else:
            st.error(f"Failed with status: {status.value}")


def edit_entity(
    entity_type: Literal["personas", "completions"],
    repo_accessor: FileManager,
    selected_name: str,
) -> None:
    """Generic function to edit existing personas or completions."""
    st.write(selected_name.capitalize())
    entity_df = repo_accessor[selected_name].df
    changed = st.data_editor(entity_df, num_rows="dynamic")

    if st.button("Save"):
        repo_accessor.update(name=selected_name, df=changed)
        st.toast(f"Updated {entity_type} '{selected_name}'")
        time.sleep(DELAY)
        st.rerun()


def delete_entity(
    entity_type: Literal["personas", "completions", "task"],
    repo_accessor: FileManager | TaskManager,
    selected_name: str,
    session_key: str,
) -> None:
    """Generic function to delete personas or completions."""
    st.error(f"Confirm: Delete {entity_type} '{selected_name}'?")
    if st.button("Confirm"):
        repo_accessor.delete(name=selected_name)
        setattr(ss, session_key, None)
        st.toast(f"Deleted {entity_type} '{selected_name}'")
        time.sleep(DELAY)
        st.rerun()


@st.dialog("Create personas", width="large")
def new_persona() -> None:
    create_entity(
        entity_type="personas",
        repo_accessor=ss.repo.personas,
        default_columns=None,
        column_placeholder="demographic, group, persona",
    )


@st.dialog("Edit Persona", width="large")
def edit_persona(selected_persona: str) -> None:
    edit_entity(
        entity_type="persona",
        repo_accessor=ss.repo.personas,
        selected_name=selected_persona,
    )


@st.dialog("Delete Persona", width="small")
def delete_persona(selected_persona: str) -> None:
    delete_entity(
        entity_type="persona",
        repo_accessor=ss.repo.personas,
        selected_name=selected_persona,
        session_key="selected_persona",
    )


@st.dialog("Delete Poll", width="small")
def delete_dialog():
    delete_entity(
        entity_type="task",
        repo_accessor=ss.repo.task_manager,
        selected_name=ss.get("selected_task"),
        session_key="selected_task",
    )


@st.dialog("Create completions", width="large")
def create_completions() -> None:
    create_entity(
        entity_type="completions",
        repo_accessor=ss.repo.completions,
        default_columns=["A", "B", "alias"],
        column_placeholder="A, B, alias",
    )


@st.dialog("Edit Completions File", width="large")
def edit_completions(selected_completions: str) -> None:
    edit_entity(
        entity_type="completions",
        repo_accessor=ss.repo.completions,
        selected_name=selected_completions,
    )


@st.dialog("Delete Completions", width="small")
def delete_completions(selected_completions: str) -> None:
    delete_entity(
        entity_type="completions",
        repo_accessor=ss.repo.completions,
        selected_name=selected_completions,
        session_key="selected_completions",
    )


# endregion


# region TASK OPERATIONS
def reset_config() -> None:
    """Reset selected task and related session state."""
    attrs = ("persona", "question", "answer", "selected_persona", "selected_completions")
    ss.task_config = PulseConfig()

    for attr in attrs:
        setattr(ss, attr, None)


def sync_config_key(session_key: str, task_key: str, val: Any = None) -> None:
    task_config: PulseConfig = ss.get("task_config")
    setattr(ss, session_key, val)
    setattr(task_config, task_key, val)


def load_selected_task() -> None:
    """Load selected task configuration into session state and UI."""
    if not (task_name := ss.get("selected_task")):
        reset_config()
        return

    # Load task config from task manager
    task_config = ss.repo.task_manager[task_name]
    # Update session state with task config
    ss.task_config = task_config
    # Update UI fields
    ss.persona = task_config.persona
    ss.question = task_config.question
    ss.answer = task_config.answer

    # Update file selections
    if not (clause_1 := GUARD.has_valid_personas):
        st.toast(clause_1.msg)
        sync_config_key(session_key="selected_persona", task_key="docs")
    else:
        ss.selected_persona = task_config.docs

    if not (clause_2 := GUARD.has_valid_completions):
        st.toast(clause_2.msg)
        sync_config_key(session_key="selected_completions", task_key="completions")
    else:
        ss.selected_completions = task_config.completions


def add_task(task_config: PulseConfig, op: Literal["saved", "updated"]) -> None:
    """Add or update task configuration in the repository."""
    status = ss.repo.task_manager.add(task_config=task_config)

    if status == TaskStatus.OK:
        st.toast(f"Task {op} successfully.")
        time.sleep(DELAY)
        if op == "saved":
            ss.selected_task = task_config.name
        st.rerun()
    else:
        st.toast(f"Save failed: {status}")


def pre_save() -> None:
    """Validate task config before update/save."""
    for guard in GUARD.save_guards:
        if guard:
            st.toast(guard.msg)
            return

    if GUARD.is_update:
        add_task(task_config=ss.task_config, op="updated")
    else:
        save()


@st.dialog("Save Poll", width="small")
def save() -> None:
    """Save task configuration with user-provided name."""
    name = st.text_input(label="Poll Name", **input_args(key="name"))

    disabled = not name.strip()
    if st.button("Save", disabled=disabled):
        add_task(task_config=ss.task_config, op="saved")


def pre_run() -> None:
    """Validate and and run task."""
    if clause := GUARD.validate(attr="run_guards"):
        st.toast(clause.msg)
        return

    run_task(name=ss.selected_task)


@st.dialog("Evaluation", width="large")
def run_task(name: str) -> None:
    """Execute evaluation task with the selected model."""
    model: ModelCard = ss.selected_model.id
    task_config: PulseConfig = ss.task_config
    task: PulseTask = ss.repo.resolve_task(task_name=ss.selected_task)

    repo: Repository = ss.repo

    with st.spinner(f"Running {name} poll"):
        st.info(task_config.question)

        results = evaluate(
            lm=ss.client.lm,
            task_dict={name: task},
            write_out=True,
            log_samples=True,
            apply_chat_template=True,
            verbosity="INFO",
            confirm_run_unsafe_code=False,
        )

        docs = repo.personas[task_config.docs].to_dict(orient="records") if task_config.docs else None
        completions = repo.completions[task_config.completions].to_dict(orient="list")

        repo.add_results(
            model=model,
            results=results,
            dataset={
                "docs": docs,
                "completions": completions,
            },
        )

    st.success(f"Run completed for Poll '{name}' with model '{model}'")
    time.sleep(2)
    st.rerun()


# endregion


# region RANKING ANALYSIS
def step(p_bar, value: int | float, text: str, delay: float = 0.0) -> None:
    """Update progress bar with value and text."""
    p_bar.progress(value=value, text=text)
    time.sleep(delay)


def pre_rank(container: DeltaGenerator) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Validate and prepare data for ranking completions."""
    if clause := GUARD.validate(attr="rank_guards"):
        ss.rankings = None
        container.warning(clause.msg)
        return None

    chat, completions = get_chat(), ss.get("selected_completions")
    completions = ss.repo.completions[completions].to_dict(orient="list")
    completions = completions["A"] + completions["B"]

    # setup ranker for vllm client and parameters - cached
    ranker = get_ranker(
        client=ss.client,
        chat=chat,
        completions=completions,
        v_pct=st.secrets.V_PCT,
        min_p=st.secrets.MIN_P,
    )

    if rankings := rank(container=container, ranker=ranker):
        ss.rankings = rankings
        # st.rerun()
    else:
        ss.rankings = None


def rank(container: DeltaGenerator, ranker: Ranker, delay: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    with container:
        pbar = st.progress(0, text="")
        step(p_bar=pbar, value=0.0, text="Ranking completions.", delay=delay)

        step(p_bar=pbar, value=0.2, text="Calculating elbows.", delay=delay)
        _ = ranker.elbows  # trigger cached property
        step(p_bar=pbar, value=0.5, text="Calculating completions.", delay=delay)
        _ = ranker.metrics  # trigger cached property
        step(p_bar=pbar, value=0.8, text="Gathering metrics.", delay=delay)
        rankings = ranker.rankings  # trigger cached property

    step(p_bar=pbar, value=1.0, text="Rankings complete.", delay=delay)
    pbar.empty()

    return rankings


# endregion


def delete():
    if not GUARD.has_task:
        reset_config()
        return

    selected_task = ss.get("selected_task")
    ss.repo.task_manager.delete(task_name=selected_task)
    reset_config()
    st.toast(f"Deleted task '{selected_task}'")
    time.sleep(DELAY)
    st.rerun()


# region UI CONTAINERS
def select_container() -> None:
    """Render poll selection and action buttons (save/run/delete)."""
    t_col, btn_col = st.columns((0.4, 0.6), vertical_alignment="bottom")
    t_col.selectbox(
        label="Polls",
        options=ss.repo.task_keys,
        index=None,
        placeholder="Select a Poll",
        key="selected_task",
        on_change=load_selected_task,
    )

    disabled = not ss.get("selected_task")
    save_col, run_col, del_col = btn_col.columns(3)
    if save_col.button(label="Save", use_container_width=True, key="save_task"):
        pre_save()
    if run_col.button(label="Run", use_container_width=True, disabled=disabled, key="run_task"):
        pre_run()
    if del_col.button(label="Delete", use_container_width=True, disabled=disabled, key="delete_task"):
        ss.repo.task_manager.delete(task_name=ss.selected_task)
        st.rerun()


@st.fragment
def prompt_container() -> None:
    """Render prompt configuration UI (persona, question, answer)."""
    st_md(text="Prompts")
    with st.container(border=True):
        st.text_input(
            label="Persona",
            placeholder="You are {{ persona }}.",
            **input_args(key="persona"),
        )
        with st.expander("Batch personas"):
            batch_container()
        st.text_input(
            label="Question",
            placeholder=ph.question,
            **input_args(key="question"),
        )
        st.text_input(
            label="Answer",
            placeholder=ph.answer,
            **input_args(key="answer"),
        )


def batch_container() -> None:
    """Render batch personas selection and management UI."""
    st.selectbox(
        label="Select personas",
        options=ss.repo.personas_keys,
        index=None,
        **input_args(key="docs", session_key="selected_persona"),
    )
    new_col, edit_col, del_col = st.columns(3)
    new_col.button(
        label="Create",
        on_click=new_persona,
        key="new_persona",
        use_container_width=True,
    )

    selected_persona = ss.get("selected_persona")
    edit_col.button(
        label="View/Edit",
        on_click=edit_persona,
        args=(selected_persona,),
        disabled=not selected_persona,
        key="edit_persona",
        use_container_width=True,
    )
    del_col.button(
        label="Delete",
        on_click=delete_persona,
        args=(selected_persona,),
        disabled=not selected_persona,
        key="delete_persona",
        use_container_width=True,
    )


@st.fragment
def completions_container() -> None:
    """Render completions selection and management UI."""
    st.selectbox(
        label="Select completions",
        options=ss.repo.completions_keys,
        index=None,
        **input_args(key="completions", session_key="selected_completions"),
    )

    new_col, edit_col, del_col = st.columns(3)
    new_col.button("Create", on_click=create_completions, use_container_width=True)

    selected_completions = ss.get("selected_completions")
    edit_col.button(
        label="View/Edit",
        on_click=edit_completions,
        args=(selected_completions,),
        disabled=not selected_completions,
        use_container_width=True,
    )
    del_col.button(
        label="Delete",
        on_click=delete_completions,
        args=(selected_completions,),
        disabled=not selected_completions,
        use_container_width=True,
    )


def analysis_container(parent: DeltaGenerator) -> None:
    """Render analysis tables for ranked completions (Side A and Side B)."""
    if rankings := ss.get("rankings"):
        A_df, B_df = rankings
        # Group A
        A_pos = get_position_table(rankings=A_df)
        st_md(text="Side A", container=parent, font_size="18px", **{"text-align": "center"})
        with parent.container(border=False, height=350):
            st.table(apply_html(styler=A_pos, cell_text_color="white"))
        # Group B
        B_pos = get_position_table(rankings=B_df)
        st_md(text="Side B", container=parent, font_size="18px", **{"text-align": "center"})
        with parent.container(border=False, height=350):
            st.table(apply_html(styler=B_pos, cell_text_color="white"))


# endregion


def raise_rank_flag():
    ss.rank_flag = True
    st.rerun()


with st.sidebar:
    sidebar_connection()
    query_params = st.query_params
    if query_params.get("debug") == "True":
        st.dataframe(ss.task_config)

st.header(HEADER)

task_col, comp_col = st.columns(2)
task_col.markdown("#### Create a Poll")
comp_col.markdown("#### Completion Analysis")
task_cont = task_col.container(border=True, height=HEIGHT)
comp_cont = comp_col.container(border=True, height=HEIGHT)

with task_cont:
    select_container()
    prompt_container()  # fragment
with task_cont.container(border=True):
    completions_container()  # fragment

if ss.get("rank_flag"):
    ss.rank_flag = False
    pre_rank(container=comp_cont)

analysis_container(parent=comp_cont)

activate_shortcuts(
    fn_map={
        "save": pre_save,
        "rank": raise_rank_flag,
        "run": pre_run,
    }
)
_, cap = comp_col.columns((2.75, 1))
cap.caption("Press Ctrl+K to open the legend")
