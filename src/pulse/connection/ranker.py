from typing import Any
from functools import cached_property
from dataclasses import dataclass

import pandas as pd

from pulse.connection.sampler import (
    get_elbows,
    get_rankings_df,
    get_position_table,
    get_completions_metrics,
)
from pulse.connection.vllm_connection import VLLMInstance


@dataclass
class Ranker:
    vllm: VLLMInstance
    chat: list[dict[str, str]]
    completions: list[str]

    v_pct: float = 0.2
    min_p: float = 0.99

    def __post_init__(self):
        self.max_logprobs = self.vllm.max_logprobs

    def __bool__(self) -> bool:
        if not getattr(self, "rankings", None):
            return False

        return all(df is not None and not df.empty for df in self.rankings)

    @cached_property
    def elbows(self) -> dict[str, list[int]]:
        return get_elbows(
            lm=self.vllm.lm,
            context=self.chat,
            completions=self.completions,
            v_size=self.max_logprobs,
            v_pct=self.v_pct,
            min_p=self.min_p,
        )

    @cached_property
    def metrics(self) -> list[dict[str, Any]]:
        return get_completions_metrics(
            lm=self.vllm.lm,
            context=self.chat,
            completions=self.completions,
        )

    @cached_property
    def rankings(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return get_rankings_df(
            metrics=self.metrics,
            elbows=self.elbows,
        )

    @property
    def A_df(self) -> pd.DataFrame:
        df, _ = self.rankings
        return df

    @property
    def B_df(self) -> pd.DataFrame:
        _, df = self.rankings
        return df

    def position_tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        A_pos = get_position_table(rankings=self.A_df)
        B_pos = get_position_table(rankings=self.B_df)
        return A_pos, B_pos
