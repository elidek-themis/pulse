import pandas as pd
import streamlit as st

from utils.plot import lineplot
from utils.task import PulseResults
from utils.tools import styler


def select(runs: pd.DataFrame) -> tuple:
    model = st.selectbox(
        "Select model",
        sorted(runs.model.unique()),
    )
    model_runs = runs[runs.model == model]
    task = st.selectbox("Select task", model_runs.task.sort_values())

    if "task" not in st.session_state:
        st.session_state.task = task
    elif task != st.session_state.task:
        st.session_state.task = task
        st.session_state.pop("choices", None)
        st.session_state.pop("columns", None)
        st.session_state.pop("menu_df", None)
        st.session_state.pop("ground_truth", None)
        st.session_state.pop("selected_key", None)

    return model, task


def update_selection() -> None:
    """Updates session state variables"""
    menu_df = st.session_state.menu_df

    selection = st.session_state.menu.selection.rows
    selection = menu_df.index.difference(selection)

    A_choices = menu_df.loc[selection, "Group A"]
    B_choices = menu_df.loc[selection, "Group B"]
    st.session_state.choices = {
        "Group A": A_choices.to_list(),
        "Group B": B_choices.to_list(),
    }
    st.session_state.columns = menu_df.loc[selection, "Alias"].to_list()


def task_summary(results) -> None:
    st.write("Menu")
    choices = results.choices
    if "choices" not in st.session_state:
        st.session_state.choices = {
            "Group A": choices["a"],
            "Group B": choices["b"],
        }

    if "columns" not in st.session_state:
        st.session_state.columns = choices["alias"]

    menu_df = pd.DataFrame(
        {
            "Group A": choices["a"],
            "Group B": choices["b"],
            "Alias": choices["alias"],
        }
    )

    if "menu_df" not in st.session_state:
        st.session_state.menu_df = menu_df

    st.dataframe(
        data=menu_df,
        hide_index=True,
        key="menu",
        on_select=update_selection,
    )
    update_selection()


def get_runs_df() -> pd.DataFrame:
    data = [(e.model, e.task, e) for e in st.session_state.results]
    return pd.DataFrame(data, columns=("model", "task", "output"))


def diff_section(results: PulseResults) -> pd.DataFrame:
    diff = pd.DataFrame(results.metrics)

    key = st.session_state.selected_key
    index = results.values(key) if results.docs else [key]
    diff.index = index

    diff = diff[st.session_state.columns]
    diff["mean"] = diff.mean(axis=1)
    diff["se"] = diff.std(axis=1) / diff.count(axis=1).apply(lambda x: x**0.5)

    with st.expander("Normalized Probability Differences", expanded=True):
        subset = diff.columns.drop(["se"])
        st.dataframe(
            data=styler(
                diff.reset_index(),
                subset=subset,
                a_color=st.session_state.group_a_color,
                b_color=st.session_state.group_b_color,
            ),
            hide_index=True,
        )

    return diff


def set_ground_truth(results: PulseResults) -> None:
    if not st.session_state.get("pct_a", None):
        st.toast("pct_a is not set")
        st.session_state.ground_truth = None
        return

    if not st.session_state.get("pct_b", None):
        st.toast("pct_b is not set")
        st.session_state.ground_truth = None
        return

    pct_a = results.values(st.session_state.pct_a)
    pct_b = results.values(st.session_state.pct_b)

    st.session_state.ground_truth = [a / (a + b) - b / (a + b) for a, b in zip(pct_a, pct_b)]


st.header("PULSE - Polling Using LLM-based Sentiment Extraction")


def setup_sidebar(results: PulseResults) -> None:
    with st.sidebar:
        st.divider()

        samples = results.samples.drop("choices", axis=1)
        string_columns = samples.select_dtypes(include="object").columns
        numerical_columns = samples.select_dtypes(exclude="object").columns

        color_col, a_col, b_col = st.columns([0.4, 0.3, 0.3])
        with color_col:
            st.write("")
            st.write("")
            st.write("Color Picker")
        with a_col:
            st.color_picker("Group A", key="group_a_color", value="#a4c2f4")
        with b_col:
            st.color_picker("Group B", key="group_b_color", value="#ea9999")

        with st.empty().container(border=True):
            st.select_slider(
                label="Figure x",
                options=list(range(5, 21)),
                value=8,
                key="fig_x",
            )

            st.select_slider(
                label="Figure y",
                options=list(range(5, 21)),
                value=6,
                key="fig_y",
            )

        if results.docs:
            st.selectbox("key", options=string_columns, key="selected_key")
        else:
            st.session_state.selected_key = results.system_prompt
            st.write(f"Key: {st.session_state.selected_key}")

        with st.empty().container(border=True):
            st.write("Ground Truth")
            a_col, minus_col, b_col = st.columns([0.5, 0.1, 0.5])

            with a_col:
                st.selectbox(
                    "pct_a",
                    options=numerical_columns,
                    index=None,
                    key="pct_a",
                    on_change=set_ground_truth,
                    args=(results,),
                )
            with minus_col:
                st.write("")
                st.write("")
                st.write("➖")
            with b_col:
                st.selectbox(
                    "pct_b",
                    options=numerical_columns,
                    index=None,
                    key="pct_b",
                    on_change=set_ground_truth,
                    args=(results,),
                )


def lineplot_section(diff: pd.DataFrame) -> None:
    if st.session_state.get("ground_truth", None):
        diff["pct_diff"] = st.session_state.ground_truth
        id_vars = ["index", "pct_diff", "mean"]
    else:
        id_vars = ["index", "mean"]

    diff["mean"] = diff["mean"].map(lambda x: "Group A" if x > 0 else "Group B")
    diff = diff.drop("se", axis=1).reset_index().melt(id_vars=id_vars, value_name="$diff$")

    fig = lineplot(
        diff=diff,
        figsize=(st.session_state.fig_x, st.session_state.fig_y),
        group_a_color=st.session_state.group_a_color,
        group_b_color=st.session_state.group_b_color,
    )

    _, pointplot_col, _ = st.columns([0.2, 0.35, 0.2])
    with pointplot_col:
        st.pyplot(fig)


runs = get_runs_df()
with st.sidebar:
    st.write("Repository")
    model, task = select(runs=runs)

if runs.empty:  # guard
    st.warning("No experiments found.")
    st.stop()

st.subheader(f"{st.session_state.task} results")

run = runs[(runs.model == model) & (runs.task == task)]
results = run.output.item()

task_summary(results=results)  # menu container
setup_sidebar(results=results)  # sidebar options
diff = diff_section(results=results)  # data container
lineplot_section(diff=diff)  # plot container
