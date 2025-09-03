from enum import StrEnum

import pandas as pd


class Placeholder(StrEnum):
    persona = "You are a citizen of the United States of America."
    question = "What will you vote for in the 2024 U.S. presidential election?"
    answer = "I will vote for"
    gen_prompt = (
        "Set if if you want to continue the last message. Else, the assistant's role will be appended to the template."
    )


def styler(df: pd.DataFrame, subset: list, a_color: str, b_color: str) -> pd.DataFrame.style:
    fn = lambda x: f"background-color: {(a_color, b_color)[x < 0]}; color:black"  # noqa: E731
    return df.style.map(func=fn, subset=pd.IndexSlice[slice(None), subset])
