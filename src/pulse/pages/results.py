import pandas as pd
import streamlit as st

from streamlit import session_state as ss

from pulse.utils.plot import lineplot
from pulse.utils.tools import styler
from pulse.data.pulse_task import PulseResults


def select(runs: pd.DataFrame) -> tuple:
    model = st.selectbox(
        "Select model",
        sorted(runs.model.unique()),
    )
    model_runs = runs[runs.model == model]
    task = st.selectbox("Select task", model_runs.task.sort_values())

    if "task" not in ss:
        ss.task = task
    elif task != ss.task:
        ss.task = task
        ss.pop("choices", None)
        ss.pop("columns", None)
        ss.pop("menu_df", None)
        ss.pop("ground_truth", None)
        ss.pop("selected_key", None)

    return model, task


def update_selection() -> None:
    """Updates session state variables"""
    menu_df = ss.menu_df

    selection = ss.menu.selection.rows
    selection = menu_df.index.difference(selection)

    A_choices = menu_df.loc[selection, "Group A"]
    B_choices = menu_df.loc[selection, "Group B"]
    ss.choices = {
        "Group A": A_choices.to_list(),
        "Group B": B_choices.to_list(),
    }
    ss.columns = menu_df.loc[selection, "Alias"].to_list()


def task_summary(results) -> None:
    st.write("Menu")
    choices = results.choices.item()

    if "choices" not in ss:
        ss.choices = {
            "Group A": choices["A"],
            "Group B": choices["B"],
        }

    if "columns" not in ss:
        ss.columns = choices["alias"]

    menu_df = pd.DataFrame(
        {
            "Alias": choices["alias"],
            "Group A": choices["A"],
            "Group B": choices["B"],
        }
    )

    if "menu_df" not in ss:
        ss.menu_df = menu_df

    st.multiselect(label="Multi-select example", options=choices["alias"], default=choices["alias"])

    with st.expander("", expanded=True):
        st.dataframe(
            data=menu_df,
            hide_index=True,
            key="menu",
            on_select=update_selection,
        )
    update_selection()


def diff_section(results) -> pd.DataFrame:
    docs = results.docs.item()
    metrics = results.metrics.item()

    key = ss.selected_key
    index = [doc[key] for doc in docs] if results.docs.any() else [key]

    diff = pd.DataFrame(metrics)
    diff.index = index

    diff = diff[ss.columns]
    cols = diff.columns

    diff["mean"] = diff.mean(axis=1)
    diff["se"] = diff.std(axis=1) / diff.count(axis=1).apply(lambda x: x**0.5)

    with st.expander("Normalized Probability Differences", expanded=True):
        diff_col, agg_col = st.columns((0.8, 0.2))

        diff_col.write("Detailed Differences")
        diff_col.dataframe(
            data=styler(
                diff.drop(["mean", "se"], axis=1).reset_index(),
                subset=cols,
                a_color=ss.group_a_color,
                b_color=ss.group_b_color,
            ),
            hide_index=True,
        )

        agg_col.write("Aggregated prediction signal")
        agg_col.dataframe(
            data=styler(
                diff[["mean", "se"]],
                subset=["mean"],
                a_color=ss.group_a_color,
                b_color=ss.group_b_color,
            ),
            hide_index=True,
        )

    return diff


def set_ground_truth(results) -> None:
    docs = results.docs.item()

    if not ss.get("pct_a", None):
        st.toast("pct_a is not set")
        ss.ground_truth = None
        return

    if not ss.get("pct_b", None):
        st.toast("pct_b is not set")
        ss.ground_truth = None
        return

    # pct_a = results.values(ss.pct_a)
    pct_a = [doc[ss.pct_a] for doc in docs]
    pct_b = [doc[ss.pct_b] for doc in docs]

    ss.ground_truth = [a / (a + b) - b / (a + b) for a, b in zip(pct_a, pct_b)]


def setup_sidebar(results) -> None:
    with st.sidebar:
        st.divider()

        string_columns = results.num.item()
        numerical_columns = results.alpha_num.item()
        # string_columns = samples.select_dtypes(include="object").columns
        # numerical_columns = samples.select_dtypes(exclude="object").columns

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

        if results.docs.any():
            st.selectbox("key", options=string_columns, key="selected_key")
        else:
            ss.selected_key = results.system_prompt
            st.write(f"Key: {ss.selected_key}")

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
    if ss.get("ground_truth", None):
        diff["pct_diff"] = ss.ground_truth
        id_vars = ["index", "pct_diff", "mean"]
    else:
        id_vars = ["index", "mean"]

    diff["mean"] = diff["mean"].map(lambda x: "Group A" if x > 0 else "Group B")
    diff = diff.drop("se", axis=1).reset_index().melt(id_vars=id_vars, value_name="$diff$")

    fig = lineplot(
        diff=diff,
        figsize=(ss.fig_x, ss.fig_y),
        group_a_color=ss.group_a_color,
        group_b_color=ss.group_b_color,
    )

    _, pointplot_col, _ = st.columns([0.2, 0.35, 0.2])
    with pointplot_col:
        st.pyplot(fig)


st.header("PULSE - Polling Using LLM-based Sentiment Extraction")
runs = ss.repo.runs

with st.sidebar:
    st.write("Repository")
    model, task = select(runs=runs)

if runs.empty:  # guard
    st.warning("No experiments found.")
    st.stop()

st.subheader(f"{ss.task} results")

run = runs[(runs.model == model) & (runs.task == task)]

task_summary(results=run)  # menu container
setup_sidebar(results=run)  # sidebar options
diff = diff_section(results=run)  # data container
lineplot_section(diff=diff)  # plot container
