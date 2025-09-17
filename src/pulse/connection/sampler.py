from typing import Any

import numpy as np
import pandas as pd

from pulse.connection.vllm_connection import Token, SampleRequest, VLLMCompletions


def reduce_prefixes(completions: list[str]) -> list[str]:
    """Calculate common prefixes between completions."""

    prefixes = []

    for completion in completions:
        # up to (not including) the last token
        continuation = f" {completion}".split(" ")[:-1]
        for i, _ in enumerate(continuation):
            cont = " ".join(continuation[: i + 1])

            if cont not in prefixes:
                prefixes.append(cont)

    return prefixes


def sample_next_tokens(
    lm: VLLMCompletions,
    context: list[dict],
    prefixes: list[str],
    v_size: int,
    v_pct: float = 0.2,
) -> list[list[Token]]:
    """Get next token distributions for each prefix at every position"""

    requests = [SampleRequest(context=context, continuation=cont) for cont in prefixes]
    results = lm.sample(
        requests=requests,
        **{
            "extra_body": {
                "add_generation_prompt": False,
                "logprobs": int(v_size * v_pct),
                "echo": False,
            }
        },
    )

    return [res.next_tokens for res in results]


def find_elbow_rank(logprobs: list[float], min_p: float) -> int:
    """Find the elbow rank where cumulative probability exceeds `min_p`"""

    logprobs = np.array(logprobs)

    # softmax - sub max for numerical stability
    probs = np.exp(logprobs - logprobs.max())
    probs /= probs.sum()

    # sorted_probs = np.sort(probs)[::-1] # already sorted
    elbow_rank = np.searchsorted(np.cumsum(probs), min_p) + 1
    return elbow_rank.item()


def expand_elbows(completions: list[str], elbow_map: dict[str, int]) -> dict[str, list[int]]:
    """Expand elbow ranks to all prefixes of completions"""

    elbows = {}

    for completion in completions:
        elbows[completion] = []
        continuation = f" {completion}".split(" ")[:-1]
        for i, _ in enumerate(continuation):
            cont = " ".join(continuation[: i + 1])

            elbows[completion].append(elbow_map[cont])

    return elbows


def get_elbows(
    lm: VLLMCompletions,
    context: list[dict],
    completions: list[str],
    v_size: int,
    v_pct: float = 0.2,
    min_p: float = 0.95,
) -> dict[str, list[int]]:
    """Get elbow ranks for all prefixes of completions"""

    prefixes = reduce_prefixes(completions=completions)
    next_tokens = sample_next_tokens(lm=lm, context=context, prefixes=prefixes, v_size=v_size, v_pct=v_pct)
    next_logprobs = ([t.logprob for t in nt] for nt in next_tokens)
    elbow_ranks = [find_elbow_rank(logprobs=logprobs, min_p=min_p) for logprobs in next_logprobs]
    elbow_map = dict(zip(prefixes, elbow_ranks))

    return expand_elbows(completions=completions, elbow_map=elbow_map)


def get_completions_metrics(
    lm: VLLMCompletions,
    context: list[dict],
    completions: list[str],
) -> list[dict[str, Any]]:
    """Get completion metrics and ranks for a given context and list of completions"""

    rank_requests = [SampleRequest(context=context, continuation=f" {cont}") for cont in completions]
    results = lm.sample(
        requests=rank_requests,
        parse_continuation=True,
        parse_next_tokens=False,
        **{
            "extra_body": {
                "add_generation_prompt": False,
                "logprobs": 1,
                "echo": False,
            }
        },
    )

    return [res.continuation.data for res in results]


def get_rankings_df(
    metrics: list[dict[str, Any]],
    elbows: dict[str, list[int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine metrics and split into groups"""

    rankings = pd.DataFrame(metrics)
    rankings["elbows"] = rankings["text"].map(elbows)
    rankings.set_index("text", inplace=True)
    rankings.index.name = "completion"
    mid = len(rankings) // 2

    return rankings.iloc[:mid], rankings.iloc[mid:]


def get_position_df(
    rankings: pd.DataFrame,
    yes_bg: str = "lightgreen",
    no_bg: str = "lightcoral",
) -> pd.DataFrame.style:
    assert "ranks" in rankings.columns and "elbows" in rankings.columns

    max_len = max(len(r) for r in rankings["ranks"])

    token_data = {f"token {i}": [] for i in range(max_len)}
    style_data = {f"token {i}": [] for i in range(max_len)}

    for idx, row in rankings.iterrows():
        tokens = idx.split(" ")
        ranks = row["ranks"]
        elbows = row["elbows"]
        for i in range(max_len):
            token = tokens[i] if i < len(tokens) else ""
            token_data[f"token {i}"].append(token)
            if i < len(ranks) and i < len(elbows) and i < len(tokens):
                if ranks[i] <= elbows[i]:
                    style_data[f"token {i}"].append(f"background-color: {yes_bg}; color: black; text-align:center")
                else:
                    style_data[f"token {i}"].append(f"background-color: {no_bg}; color: black; text-align:center")
            else:
                style_data[f"token {i}"].append("")

    token_data["logprob"] = rankings["logprob"].round(4).astype(str).tolist()
    style_data["logprob"] = [""] * len(rankings)

    comp_df = pd.DataFrame(token_data)
    style_df = pd.DataFrame(style_data)

    comp_df = comp_df.reset_index(drop=True)
    style_df = style_df.reset_index(drop=True)

    return comp_df.style.apply(lambda col: style_df[col.name], axis=0)
