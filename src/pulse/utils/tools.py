from enum import StrEnum

import pandas as pd


class Placeholder(StrEnum):
    persona = "You are a citizen of the United States of America."
    question = "What will you vote for in the 2024 U.S. presidential election?"
    answer = "I will vote for"
    gen_prompt = (
        "Set if if you want to continue the last message. Else, the assistant's role will be appended to the template."
    )


def styler(
    df: pd.DataFrame,
    subset: list,
    a_color: str,
    b_color: str,
    index_color: str = None,
    header_color: str = None,
    font_size: str = "14px",  # cells
    index_font_size: str = "14px",  # index
    header_font_size: str = "14px",  # headers
) -> pd.DataFrame.style:
    def _fn(x):  # style condition for data cells
        color = a_color if x >= 0 else b_color
        return f"background-color: {color}; color: black; font-size: {font_size}"

    styled = df.style.map(_fn, subset=pd.IndexSlice[:, subset])

    # index text color & font size
    if index_color or index_font_size:
        styled = styled.set_table_styles(
            [
                {
                    "selector": "th.row_heading",
                    "props": [
                        ("color", index_color if index_color else "inherit"),
                        ("font-size", index_font_size),
                    ],
                }
            ],
            overwrite=False,
        )

    # header text color & font size
    if header_color or header_font_size:
        styled = styled.set_table_styles(
            [
                {
                    "selector": "th.col_heading",
                    "props": [
                        ("color", header_color if header_color else "inherit"),
                        ("font-size", header_font_size),
                        ("text-align", "center"),
                    ],
                }
            ],
            overwrite=False,
        )

    return styled
