import json
import hashlib

from typing import Any, Self
from pathlib import Path
from dataclasses import asdict, dataclass
from collections.abc import Callable

import yaml
import numpy as np
import datasets

from lm_eval.utils import sanitize_model_name
from lm_eval.api.task import TaskConfig, ConfigurableTask
from lm_eval.api.instance import Instance

from pulse.utils.paths import TASKS, RESULTS
from pulse.connection.vllm_connection import Prompt


class PulseResults:
    metric: str = "norm_prob_diff,none"

    def __init__(self, model: str, results: dict[str, Any]):
        self.model = model
        results_ = results.get("results")
        self.task = next(iter(results_))

        self.metrics = results_[self.task][self.metric]

    def save(self):
        """@ data/results/{task}-{model}.json"""
        model_sanitized = sanitize_model_name(model_name=self.model)

        file_path = RESULTS / f"{self.task}-{model_sanitized}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "task": self.task,
            "model": self.model,
            "metrics": self.metrics,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return file_path

    def __repr__(self) -> str:
        return f"PulseResults(model={self.model}, task={self.task})"

    def __str__(self) -> str:
        return repr(self)

    @classmethod
    def from_json(cls, file_path: Path | str) -> Self:
        file_path = Path(file_path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        task = data["task"]
        model = data["model"]
        metrics = data["metrics"]

        results = {"results": {task: {cls.metric: metrics}}}
        return cls(model=model, results=results)


@dataclass
class PulseConfig(TaskConfig):
    def __post_init__(self):
        if not self.dataset_kwargs:
            self.dataset_kwargs = {"docs": None}
        self.doc_to_target = -1
        self.output_type = "loglikelihood"
        super().__post_init__()

    @property
    def id(self) -> str:
        data_str = json.dumps(self.dataset_kwargs, sort_keys=True)
        raw_str = f"{self.description}|{self.doc_to_text}|{self.gen_prefix}|{data_str}"

        return hashlib.md5(raw_str.encode()).hexdigest()

    @property
    def data(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "description": self.description,
            "doc_to_text": self.doc_to_text,
            "gen_prefix": self.gen_prefix,
            "dataset_kwargs": self.dataset_kwargs,
        }

    def to_yaml(self):
        return yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True)

    def to_json(self):
        return json.dumps(self.data, indent=2)

    def save(self):
        file_name = f"{self.task}.yaml"
        file_path = TASKS / file_name

        with open(file_path, mode="w", encoding="utf-8") as file:
            file.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, file_path: Path | str) -> Self:
        with open(file_path, encoding="utf-8") as file:
            data = yaml.safe_load(stream=file)

        return cls(**data)

    def to_eval_dict(self) -> dict:
        # ??
        return asdict(self)

    def __hash__(self) -> int:
        return int(self.id, 16)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PulseConfig) and self.id == other.id


class PulseTask(ConfigurableTask):
    TEST_SPLIT: str = "test"

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

    def has_test_docs(self):
        return bool(self.test_docs)

    def test_docs(self):
        return self.dataset[self.TEST_SPLIT]

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> list[Instance] | Instance:
        kwargs.pop("apply_chat_template", False)
        kwargs.pop("chat_template", None)

        choices = self.doc_to_choice(doc)
        target_delimiter = self.config.target_delimiter
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

    def custom_dataset(self, **kwargs) -> dict[str, datasets.Dataset]:
        docs = kwargs.get("docs")
        completions = kwargs.get("completions")

        if docs:
            dataset = datasets.Dataset.from_list(docs)
            dataset = dataset.add_column("choices", [completions] * len(dataset))
            dataset.choices = completions
        else:
            dataset = datasets.Dataset.from_list([{"choices": completions}])
            dataset.choices = completions

        return {"test": dataset}

    def download(self, dataset_kwargs: dict[str, Any] | None = None, **kwargs) -> None:
        self.dataset = self.custom_dataset(**(self.config.metadata or {}), **(self.config.dataset_kwargs or {}))

    def process_results(self, doc: dict[str, Any], results: list[Prompt]):
        alias = doc["choices"]["alias"]
        no_choices = len(alias)

        # negative log likelihoods
        lls = [prompt.continuation.logprob for prompt in results]
        lls_a, lls_b = lls[:no_choices], lls[no_choices:]

        # exponentiation to get probabilities
        e_a, e_b = np.exp(lls_a), np.exp(lls_b)
        total = e_a + e_b

        # normalize probabilities
        prob_a, prob_b = e_a / total, e_b / total

        # normalized probability differences
        diff = (prob_a - prob_b).tolist()

        return {"norm_prob_diff": dict(zip(alias, diff))}

    def doc_to_choice(self, doc: Any, doc_to_choice=None) -> list[str]:
        choices = doc["choices"]
        return choices["A"] + choices["B"]

    def aggregation(self) -> dict[str, Callable[[list], Any]]:
        def _pass(arr: list) -> list:
            return arr

        return {"norm_prob_diff": _pass}

    def higher_is_better(self) -> dict[str, bool]:
        return {"norm_prob_diff": True}
