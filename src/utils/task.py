from typing import Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import datasets

from lm_eval.api.task import TaskConfig, ConfigurableTask
from lm_eval.api.instance import Instance


@dataclass
class PulseResults:
    model: str
    task: str
    results: dict

    def __post_init__(self) -> None:
        samples = [x["doc"] for x in self.results["samples"][self.task]]
        self.samples = pd.DataFrame(samples)

    @property
    def docs(self) -> list[dict] | None:
        return self.results["configs"][self.task]["dataset_kwargs"].get("docs", None)

    @property
    def choices(self) -> dict:
        return self.results["configs"][self.task]["dataset_kwargs"]["completions"]

    @property
    def keys(self) -> list[str]:
        sample, *_ = self.docs
        return sample.keys()

    @property
    def system_prompt(self) -> str:
        return self.results["configs"][self.task]["description"]

    @property
    def user_prompt(self) -> str:
        return self.results["configs"][self.task]["doc_to_text"]

    @property
    def assistant_prefix(self) -> str:
        return self.results["configs"][self.task]["gen_prefix"]

    @property
    def metrics(self) -> list[dict]:
        return self.results["results"][self.task]["norm_prob_diff,none"]

    @property
    def configs(self) -> dict[str, Any]:
        task_configs = self.results.get("configs", {})
        cli_configs = self.results.get("config", {})

        return {
            "task_configs": task_configs,
            "cli_configs": cli_configs,
        }

    def values(self, key: str) -> list:
        return [doc[key] for doc in self.docs]

    def __str__(self) -> str:
        return f"Results(model={self.model}, task={self.task})"

    def __hash__(self) -> int:
        return hash((self.model, self.task))


class PulseMultipleChoice(ConfigurableTask):
    def __init__(self, config) -> None:
        self.CONFIG = config
        super().__init__()

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> list[Instance] | Instance:
        kwargs.pop("apply_chat_template", False)
        kwargs.pop("chat_template", None)

        choices = self.doc_to_choice(doc)
        target_delimiter = self.config.target_delimiter
        # if apply_chat_template:
        #     target_delimiter = ""
        arguments = [(ctx, f"{target_delimiter}{cont}") for cont in choices]

        request_list = [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=arg,
                idx=i,
                **kwargs,
            )
            for i, arg in enumerate(arguments)
        ]

        return request_list


def load_dataset(**kwargs) -> dict[str, datasets.Dataset]:
    docs = kwargs.get("docs", None)
    completions = kwargs.get("completions")

    if docs:
        dataset = datasets.Dataset.from_list(docs)
        dataset = dataset.add_column("choices", [completions] * len(dataset))
        dataset.choices = completions
    else:
        dataset = datasets.Dataset.from_list([{"choices": completions}])
        dataset.choices = completions

    return {"test": dataset}


def norm_prob_diff(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, float]]:
    alias = doc["choices"]["alias"]
    no_choices = len(alias)
    results, _ = zip(*results)  # remove is_greedy

    # negative log likelihoods
    lls = np.array(results)
    lls_a, lls_b = lls[:no_choices], lls[no_choices:]

    # exponentiation to get probabilities
    e_a, e_b = np.exp(lls_a), np.exp(lls_b)
    total = e_a + e_b

    # normalize probabilities
    prob_a, prob_b = e_a / total, e_b / total

    # normalized probability differences
    diff = (prob_a - prob_b).tolist()

    return {"norm_prob_diff": dict(zip(alias, diff))}


def aggregation(arr):
    return arr


referendum_task = TaskConfig(
    dataset_name="",
    dataset_path="",
    custom_dataset=load_dataset,
    dataset_kwargs={},
    test_split="test",
    doc_to_target=0,
    doc_to_choice="{{ choices['a'] + choices['b'] }}",
    output_type="multiple_choice",
    process_results=norm_prob_diff,
    metric_list=[{"metric": "norm_prob_diff", "aggregation": aggregation, "higher_is_better": True}],
)
