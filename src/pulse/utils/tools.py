from enum import StrEnum
from dataclasses import dataclass
from collections.abc import Callable

import pandas as pd

from pandas.io.formats.style import Styler


class Placeholder(StrEnum):
    persona = "You are a citizen a U.S. citizen."
    question = "What will you vote for in the 2024 U.S. presidential election?"
    answer = "I will vote for"
    completion = " the Republican"


class Latex(StrEnum):
    diff = r"$\overline{\mathrm{diff}}$"


@dataclass
class ModelCard:
    id: str
    root: str

    def __str__(self):
        return self.id

    def __eq__(self, other):
        if isinstance(other, ModelCard):
            return self.root == other.root
        return False

    def __hash__(self):
        return hash(self.id)


def styler(
    df: pd.DataFrame | Styler,
    subset: list[str] | None = None,
    a_color: str = "#a4c2f4",
    b_color: str = "#ea9999",
    cell_text_color: str = "white",
    cond: Callable = lambda x: x >= 0,
) -> Styler:
    def _fn(x):  # style condition for data cells
        color = a_color if cond(x) else b_color
        return f"background-color: {color}; color: black"

    styler = df.style.map(_fn, subset=pd.IndexSlice[:, subset])
    return apply_html(styler, cell_text_color=cell_text_color)


def apply_html(styler: Styler, cell_text_color: str = "white") -> Styler:
    return styler.set_table_styles(
        [
            {  # index name
                "selector": "th.blank",
                "props": [
                    ("color", cell_text_color),
                    ("text-align", "center"),
                ],
            },
            {  # index
                "selector": "th.row_heading",
                "props": [
                    ("color", cell_text_color),
                    ("text-align", "left"),
                ],
            },
            {  # column text color
                "selector": "th.col_heading",
                "props": [
                    ("color", cell_text_color),
                    ("text-align", "center"),
                ],
            },
            {  # cell text color
                "selector": "td > div",
                "props": [
                    ("text-align", "center"),
                ],
            },
        ],
        overwrite=False,
    )
