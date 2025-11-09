from typing import Any

import numpy as np
import pandas as pd

from pulse.connection.types import Token, SampleRequest
from pulse.connection.vllm_connection import VLLMCompletions


def build_token_prefixes(continuation_tokens: list[Token]) -> list[str]:
    """Build cumulative text at each token position (before each token)."""
    prefixes = []
    cumulative = ""
    for token in continuation_tokens:
        prefixes.append(cumulative)
        cumulative += token.token
    return prefixes


def get_token_based_prefixes(
    lm: VLLMCompletions,
    context: list[dict],
    completions: list[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Sample completions to get actual tokenization and build token-based prefixes."""
    requests = [SampleRequest(context=context, continuation=f" {c}") for c in completions]
    results = lm.sample(
        requests=requests,
        **{
            "extra_body": {
                "add_generation_prompt": False,
                "logprobs": 1,
                "echo": False,
            }
        },
    )

    # Build prefixes for each completion
    all_prefixes = []
    completion_to_prefixes = {}

    for i, result in enumerate(results):
        comp = completions[i]
        prefixes = build_token_prefixes(result.continuation.tokens)
        completion_to_prefixes[comp] = prefixes

        # Collect unique prefixes
        for prefix in prefixes:
            if prefix not in all_prefixes:
                all_prefixes.append(prefix)

    return all_prefixes, completion_to_prefixes


def sample_prefix_distributions(
    lm: VLLMCompletions,
    context: list[dict],
    prefixes: list[str],
    v_size: int,
    v_pct: float = 0.2,
) -> list[list[Token]]:
    """Sample next token distributions at each prefix position."""
    requests = [SampleRequest(context=context, continuation=pref) for pref in prefixes]
    results = lm.sample(
        requests=requests,
        **{
            "extra_body": {"add_generation_prompt": False, "logprobs": int(v_size * v_pct), "echo": False},
        },
    )

    return [res.next_tokens for res in results]


def calculate_prefix_elbows(
    prefix_distributions: list[list[Token]],
    min_p: float = 0.99,
) -> list[int]:
    """Calculate elbow rank for each prefix's distribution."""
    elbow_ranks = []
    for tokens in prefix_distributions:
        logprobs = [t.logprob for t in tokens]
        elbow_rank = find_elbow_rank(logprobs, min_p)
        elbow_ranks.append(elbow_rank)

    return elbow_ranks


def map_elbows_to_completions(
    all_prefixes: list[str],
    elbow_ranks: list[int],
    completion_to_prefixes: dict[str, list[str]],
) -> dict[str, list[int]]:
    """Map elbow ranks from prefixes back to completions."""
    # Create prefix -> elbow mapping
    prefix_to_elbow = dict(zip(all_prefixes, elbow_ranks))

    # Map to each completion's token positions
    completion_elbows = {}
    for comp, prefixes in completion_to_prefixes.items():
        completion_elbows[comp] = [prefix_to_elbow[p] for p in prefixes]

    return completion_elbows


def find_elbow_rank(logprobs: list[float], min_p: float) -> int:
    """Finds rank where cumulative probability exceeds `min_p`"""
    logprobs = np.array(logprobs)

    # softmax - sub max for numerical stability
    probs = np.exp(logprobs - logprobs.max())
    probs /= probs.sum()

    # sorted_probs = np.sort(probs)[::-1] # already sorted
    elbow_rank = np.searchsorted(np.cumsum(probs), min_p) + 1
    return elbow_rank.item()


def get_elbows(
    lm: VLLMCompletions,
    context: list[dict],
    completions: list[str],
    v_size: int,
    v_pct: float = 0.2,
    min_p: float = 0.99,
) -> dict[str, list[int]]:
    """Calculate elbow ranks at each token position for completions."""
    all_prefixes, completion_to_prefixes = get_token_based_prefixes(lm, context, completions)
    prefix_distributions = sample_prefix_distributions(lm, context, all_prefixes, v_size, v_pct)
    elbow_ranks = calculate_prefix_elbows(prefix_distributions, min_p)
    completion_elbows = map_elbows_to_completions(all_prefixes, elbow_ranks, completion_to_prefixes)

    return completion_elbows


def get_completions_metrics(
    lm: VLLMCompletions,
    context: list[dict],
    completions: list[str],
) -> list[dict[str, Any]]:
    """Gets completion metrics and ranks for a given context and list of completions"""
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
    """Combines metrics and splits into groups"""
    rankings = pd.DataFrame(metrics)
    rankings["elbows"] = rankings["text"].map(elbows)
    rankings.set_index("text", inplace=True)
    rankings.index.name = "completion"
    mid = len(rankings) // 2

    return rankings.iloc[:mid], rankings.iloc[mid:]


def get_position_table(
    rankings: pd.DataFrame,
    yes_bg: str = "lightgreen",
    no_bg: str = "lightcoral",
) -> pd.DataFrame.style:
    assert {"ranks", "elbows", "token_strings"}.issubset(rankings.columns)

    def _cell_color(bg: str) -> str:
        color = "color: black"
        text_align = "text-align:center"
        return f"background-color: {bg}; {color}; {text_align}"

    max_len = max(len(r) for r in rankings["ranks"])

    token_data = {f"token {i}": [] for i in range(max_len)}
    style_data = {f"token {i}": [] for i in range(max_len)}

    for _, row in rankings.iterrows():
        tokens = row["token_strings"]
        ranks = row["ranks"]
        elbows = row["elbows"]

        for i in range(max_len):
            col_name = f"token {i}"
            token = tokens[i] if i < len(tokens) else ""
            token_data[col_name].append(token)

            if i < len(ranks) and i < len(elbows):
                if ranks[i] <= elbows[i]:
                    style_data[col_name].append(_cell_color(yes_bg))
                else:
                    style_data[col_name].append(_cell_color(no_bg))
            else:
                style_data[col_name].append("")

    token_data["logprob"] = rankings["logprob"].round(4).astype(str).tolist()
    style_data["logprob"] = [""] * len(rankings)

    comp_df = pd.DataFrame(token_data)
    style_df = pd.DataFrame(style_data)

    return comp_df.style.apply(lambda col: style_df[col.name], axis=0)
