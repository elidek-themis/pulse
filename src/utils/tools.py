import re

from enum import StrEnum

import pandas as pd


class Placeholder(StrEnum):
    persona = "You are a citizen of the United States of America."
    question = "What will you vote for in the 2024 U.S. presidential election?"
    answer = "I will vote for"
    gen_prompt = (
        "Set if if you want to continue the last message. "
        "Else, the assistant's role will be appended to the template."
    )


def styler(df: pd.DataFrame, subset: list, a_color: str, b_color: str) -> pd.DataFrame.style:
    fn = lambda x: f"background-color: {(a_color, b_color)[x < 0]}; color:black"  # noqa: E731
    return df.style.map(func=fn, subset=pd.IndexSlice[slice(None), subset])


def format_jinja(template: str, indent_width: int = 4) -> str:
    token_re = re.compile(r"({{.*?}}|{%-?.*?-%}|{%.*?%})", re.DOTALL)
    tokens = token_re.findall(template)
    indent = 0
    result = []

    for token in tokens:
        stripped = token.strip()

        # dedent before writing if it's an end tag
        if (
            re.match(r"{%[-\s]*end\w+", stripped)
            or re.match(r"{%[-\s]*else", stripped)
            or re.match(r"{%[-\s]*elif", stripped)
        ):
            indent -= 1

        result.append(" " * (indent * indent_width) + stripped)

        # re-indent after if it's an opening tag (not else/elif/end)
        if re.match(r"{%[-\s]*(for|if|block|macro|filter|with)\b", stripped):
            indent += 1
        elif re.match(r"{%[-\s]*(else|elif)\b", stripped):
            indent += 1

    return "\n".join(result)
