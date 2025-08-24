import re

from enum import StrEnum

import pandas as pd


class Placeholder(StrEnum):
    n = "Number of output sequences to return for the given prompt."
    presence_penalty = (
        "Float that penalizes new tokens based on whether they appear in the generated text so far. "
        "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
    )
    frequency_penalty = (
        "Float that penalizes new tokens based on their frequency in the generated text so far. "
        "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
    )
    repetition_penalty = (
        "Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far. "
        "Values > 1 encourage the model to use new tokens, while values < 1 encourage the model to repeat tokens."
    )
    temperature = (
        "Float that controls the randomness of the sampling. "
        "Lower values make the model more deterministic, while higher values make the model more random. "
        "Zero means greedy sampling."
    )
    top_p = (
        "Float that controls the cumulative probability of the top tokens to consider. "
        "Must be in (0, 1]. Set to 1 to consider all tokens."
    )
    top_k = "Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens."
    min_p = (
        "Float that represents the minimum probability for a token to be considered, "
        "relative to the probability of the most likely token. "
        "Must be in [0, 1]. Set to 0 to disable this."
    )
    max_tokens = "Maximum number of tokens to generate per output sequence."
    min_tokens = (
        "Minimum number of tokens to generate per output sequence before EOS or stop_token_ids can be generated."
    )
    guided_regex = "If specified, the output will follow the regex pattern. Mind the leading whitespace!"
    add_gen_prompt = (
        "If true, the generation prompt will be added to the chat template. "
        "This is a parameter used by chat template in tokenizer config of the model."
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
