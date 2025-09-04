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


def task_summary(results) -> None:
    # with st.expander("Selected completions", expanded=True):
    choices = results.choices.item()

    if "choices" not in ss:
        ss.choices = {
            "Group A": choices["A"],
            "Group B": choices["B"],
        }

    # sel_col, *_ = st.columns(5)
    st.multiselect(label="Selected completions", options=choices["alias"], default=choices["alias"], key="columns")

    menu_df = pd.DataFrame(
        {
            "Alias": choices["alias"],
            "Group A": choices["A"],
            "Group B": choices["B"],
        }
    )

    st.popover("test", use_container_width=True).dataframe(menu_df)

    # with st.expander("", expanded=True):
    #     st.dataframe(
    #         data=menu_df,
    #         hide_index=True,
    #         key="menu",
    #     )


def diff_section(results) -> pd.DataFrame:
    def _escape_dollar(x):
        if isinstance(x, str) and x.count("$") >= 2:
            return x.replace("$", r"\$")

        return x

    docs = results.docs.item()
    metrics = results.metrics.item()

    index = pd.DataFrame(docs).iloc[:, 0]

    diff = pd.DataFrame(metrics)
    diff.index = [_escape_dollar(idx) for idx in index.tolist()]

    diff = diff[ss.columns]
    cols = diff.columns

    diff["mean"] = diff.mean(axis=1)
    diff["SE"] = diff.std(axis=1) / diff.count(axis=1).apply(lambda x: x**0.5)

    agg_tab, diff_tab = st.tabs(("Aggregated Prediction Signal", "Detailed Differences"))

    diff_tab.table(
        data=styler(
            diff.drop(["mean", "SE"], axis=1),
            subset=ss.columns,
            a_color="#a4c2f4",
            b_color="#ea9999",
            index_color="white",
            header_color="white",
            font_size="16px",
            index_font_size="16px",
            header_font_size="16px",
        ),
    )

    with agg_tab:
        df_col, line_col = st.columns((0.3, 0.7))

        agg_df = pd.DataFrame(diff[["mean", "SE"]])
        agg_df.columns = ["$\overline{mean}$", "SE"]

        styled = styler(
            agg_df,
            subset=["$\overline{mean}$"],
            a_color="#a4c2f4",
            b_color="#ea9999",
            index_color="white",
            header_color="white",
            font_size="10px",
            index_font_size="10px",
            header_font_size="10px",
        )
        # styled.columns = [r"$\overline{mean}$", "SE"]
        df_col.table(data=styled)

        with line_col:
            lineplot_section(diff=diff, docs=pd.DataFrame(docs))


def has_gold_cols(docs: pd.DataFrame) -> list | None:
    if {"A pct", "B pct"}.issubset(docs.columns):
        pct_a = docs["A pct"]
        pct_b = docs["B pct"]

        return ((pct_a - pct_b) / (pct_a + pct_b)).tolist()
    # ss.ground_truth = [a / (a + b) - b / (a + b) for a, b in zip(pct_a, pct_b)]


def setup_sidebar() -> None:
    st.sidebar.divider()

    x_col, y_col = st.sidebar.columns(2)

    x_col.select_slider(
        label="Figure x",
        options=[round(x * 0.1, 1) for x in range(50, 201)],
        value=8.0,
        key="fig_x",
    )

    y_col.select_slider(
        label="Figure y",
        options=[round(x * 0.1, 1) for x in range(50, 201)],
        value=6.0,
        key="fig_y",
    )


def lineplot_section(diff: pd.DataFrame, docs: pd.DataFrame) -> None:
    if ground_truth := has_gold_cols(docs=docs):
        diff["pct_diff"] = ground_truth
        id_vars = ["index", "pct_diff", "mean"]
    else:
        id_vars = ["index", "mean"]

    diff["mean"] = diff["mean"].map(lambda x: "Group A" if x > 0 else "Group B")
    diff = diff.drop("SE", axis=1).reset_index().melt(id_vars=id_vars, value_name="$diff$")

    fig = lineplot(
        diff=diff,
        figsize=(ss.fig_x, ss.fig_y),
        group_a_color="blue",
        group_b_color="red",
    )

    pointplot_col, *_ = st.columns([0.5, 0.1, 0.1])
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
setup_sidebar()  # sidebar options
diff = diff_section(results=run)  # data container

st.empty().container(height=100, border=False)  # bottom padding
