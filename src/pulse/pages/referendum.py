import time

import pandas as pd
import streamlit as st

from lm_eval import evaluate
from streamlit import session_state as ss
from streamlit.logger import get_logger
from streamlit.delta_generator import DeltaGenerator

from pulse.pages.guard import pulse_guard
from pulse.pages.state import (
    st_md,
    get_chat,
    init_session_state,
    sidebar_connection,
    persist_session_state,
)
from pulse.utils.tools import Placeholder as ph
from pulse.utils.tools import apply_html
from pulse.data.pulse_task import PulseTask
from pulse.data.file_manager import FileStatus
from pulse.data.task_manager import TaskStatus
from pulse.connection.sampler import (
    get_elbows,
    get_rankings_df,
    get_position_table,
    get_completions_metrics,
)

init_session_state()
persist_session_state()

logger = get_logger(__name__)


@st.dialog("Create personas", width="large")
def new_persona() -> None:
    name = st.text_input("Name")
    uploaded_files = st.file_uploader("Upload persona file", type=("csv", "json"))

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
    logger.info(changed)

    if st.button("Save"):
        ss.repo.personas.update(name=selected_persona, df=changed)
        st.toast(f"Updated {selected_persona}")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Persona", width="small")
def delete_persona(selected_persona) -> None:
    st.error(f"Confirm: Delete personas '{selected_persona}'?")
    if st.button("Confirm"):
        ss.repo.personas.delete(name=selected_persona)
        ss.selected_persona = None
        st.toast(f"Deleted {selected_persona}")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Create completions", width="large")
def create_completions() -> None:
    name = st.text_input("Name")
    uploaded_files = st.file_uploader("Upload persona file", type=("csv", "json"))

    if uploaded_files:
        # add checks
        _, ext = uploaded_files.name.split(".")
        completions_df = pd.read_json(uploaded_files) if ext == "json" else pd.read_csv(uploaded_files)
    else:
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
        st.toast(f"Updated {selected_completions}")
        time.sleep(0.5)
        st.rerun()


@st.dialog("Delete Completions", width="small")
def delete_completions(selected_completions) -> None:
    st.error(f"Confirm: Delete completions '{selected_completions}'?")
    if st.button("Confirm"):
        ss.repo.completions.delete(name=selected_completions)
        ss.selected_completions = None
        st.toast(f"Deleted {selected_completions}")
        time.sleep(0.5)
        st.rerun()


def update_task_config(key: str) -> None:
    setattr(ss.task_config, key, ss.get(key))


def update_dataset_kwargs_docs() -> None:
    selection = ss.get("selected_persona")
    personas = ss.repo.personas[selection].to_dict(orient="records") if selection else None
    ss.task_config.dataset_kwargs.update({"docs": personas})


def update_dataset_kwargs_completions(child: DeltaGenerator) -> None:
    selection = ss.get("selected_completions")
    completions = ss.repo.completions[selection].to_dict(orient="list") if selection else None
    ss.task_config.dataset_kwargs.update({"completions": completions})

    A_df, B_df = pre_rank(container=child)
    ss.A_df, ss.B_df = A_df, B_df


def completions_container(child: DeltaGenerator) -> None:
    st.selectbox(
        label="Select completions",
        options=ss.repo.all_completions,
        index=None,
        on_change=update_dataset_kwargs_completions,
        args=(child,),
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


def pre_run() -> None:
    task = ss.get("selected_task")

    if not task:
        st.toast("Select a Poll to run.")
        return

    run_task(name=task)


@st.dialog("Evaluation", width="small")
def run_task(name: str) -> None:
    with st.spinner(f"Running {name} poll"):
        # TODO: extract function
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


def pre_save() -> None:
    for guard in pulse_guard.save_guards:
        if guard:
            st.toast(guard.msg)
            return

    save()


@st.dialog("Save Poll", width="small")
def save() -> None:
    name = st.text_input(
        "Poll Name",
        key="task",
        on_change=update_task_config,
        args=("task",),
    )

    disabled = not name.strip()
    if st.button("Save", disabled=disabled):
        # ss.task_config["task"] = name  # or ss.task_config["name"] = name
        status = ss.repo.task_manager.add(task_config=ss.task_config)
        if status == TaskStatus.OK:
            st.toast("Task saved successfully.")
            time.sleep(0.5)
            ss.selected_task = name
            st.rerun()
        else:
            st.toast(f"Save failed: {status}")


@st.dialog("Task Config", width="large")
def show_config() -> None:
    task_config = ss.selected_task
    st.json(ss.repo.task_manager[task_config].to_json())


def select_container(parent: DeltaGenerator) -> None:
    t_col, btn_col = parent.columns((0.4, 0.6), vertical_alignment="bottom")
    t_col.selectbox(
        label="Polls",
        options=ss.repo.all_tasks,
        index=None,
        placeholder="Select a Poll",
        key="selected_task",
    )

    save_col, run_col, del_col = btn_col.columns(3)
    if save_col.button("Save", use_container_width=True, key="save_task"):
        pre_save()
    if run_col.button("Run", use_container_width=True, key="run_task"):
        pre_run()
    if del_col.button("Delete", use_container_width=True, key="delete_task"):
        ss.repo.task_manager.delete(task_name=ss.selected_task)
        st.rerun()


def prompt_container(parent: DeltaGenerator):
    st_md(text="Prompts", container=parent)
    with parent.container(border=True):
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
            placeholder=ph.question,
            on_change=update_task_config,
            args=("doc_to_text",),
            key="doc_to_text",
        )
        st.text_input(
            label="Answer",
            placeholder=ph.answer,
            on_change=update_task_config,
            args=("gen_prefix",),
            key="gen_prefix",
        )


def step(p_bar, value: int | float, text: str, delay: float = 0.0) -> None:
    p_bar.progress(value=value, text=text)
    time.sleep(delay)


def pre_rank(container: DeltaGenerator) -> None:
    for guard in pulse_guard.rank_guards:
        if guard:
            container.warning(guard.msg)
            st.stop()

    chat, completions = get_chat(), ss.get("selected_completions")
    completions = ss.repo.completions[completions].to_dict(orient="list")
    completions = completions["A"] + completions["B"]

    return rank(chat=chat, completions=completions, _container=container)


def rank(chat: list[dict[str, str]], completions: list[str], _container: DeltaGenerator) -> tuple[pd.DataFrame]:
    p_bar = _container.progress(value=0)

    step(p_bar=p_bar, value=0, text="Ranking completions", delay=1)  # 0%
    step(p_bar=p_bar, value=0.25, text="Calculating elbow ranks")  # 25%
    elbows = get_elbows(
        lm=ss.vllm_conn.lm,
        context=chat,
        completions=completions,
        v_size=ss.vllm_conn.max_logprobs,
        v_pct=st.secrets.V_PCT,
        min_p=st.secrets.MIN_P,
    )

    step(p_bar=p_bar, value=0.5, text="Calculating metrics", delay=0.5)  # 50%
    metrics = get_completions_metrics(lm=ss.vllm_conn.lm, context=chat, completions=completions)

    step(p_bar=p_bar, value=0.75, text="Splitting sides", delay=0.5)  # 75%
    A_df, B_df = get_rankings_df(metrics=metrics, elbows=elbows)

    step(p_bar=p_bar, value=1, text="Rankings complete ✔️", delay=0.5)  # 100%
    p_bar.empty()

    return A_df, B_df


def analysis_container(parent: DeltaGenerator) -> None:
    if (A_df := ss.get("A_df")) and (B_df := ss.get("B_df")):  # create guard
        # Group A
        A_pos = get_position_table(rankings=A_df)
        st_md(text="Side A", container=parent, font_size="18px", **{"text-align": "center"})
        with st.container(border=False, height=350):
            st.table(apply_html(styler=A_pos, cell_text_color="white"))
        # Group B
        B_pos = get_position_table(rankings=B_df)
        st_md(text="Side B", container=parent, font_size="18px", **{"text-align": "center"})
        with st.container(border=False, height=350):
            st.table(apply_html(styler=B_pos, cell_text_color="white"))


with st.sidebar:
    sidebar_connection()

HEIGHT = 800

st.header("PULSE - Polling Using LLM-based Sentiment Extraction")

task_col, comp_col = st.columns(2)
task_col.markdown("#### Create a Poll")
comp_col.markdown("#### Completion Analysis")
task_cont = task_col.container(border=True, height=HEIGHT)
comp_cont = comp_col.container(border=True, height=HEIGHT)

select_container(parent=task_cont)
prompt_container(parent=task_cont)
with task_cont.container(border=True):
    completions_container(child=comp_cont)
analysis_container(parent=comp_cont)
