import json

from typing import Any, Self
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Callable

import yaml
import numpy as np
import datasets

from lm_eval.utils import sanitize_model_name
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance

from pulse.utils.paths import TASKS, RESULTS
from pulse.connection.types import Prompt


@dataclass
class PulseResults:
    model: str
    results: dict[str, Any]
    dataset: dict[str, Any]

    DEFAULT_METRIC: str = field(default="norm_prob_diff,none", repr=False)
    task: str = field(init=False)
    metrics: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        results_ = self.results.get("results")
        self.task = next(iter(results_))
        self.metrics = results_[self.task][self.DEFAULT_METRIC]

        self.has_docs = bool(self.docs)

    def __repr__(self) -> str:
        return f"PulseResults(model={self.model}, task={self.task})"

    def __str__(self) -> str:
        return repr(self)

    @property
    def docs(self) -> dict[str, Any]:
        return self.dataset.get("docs", {})

    @property
    def completions(self) -> dict[str, Any]:
        return self.dataset.get("completions", {})

    @property
    def key(self) -> str:
        model_sanitized = sanitize_model_name(model_name=self.model)
        return f"{self.task}-{model_sanitized}"

    def save(self) -> Path:
        """@ data/results/key.json"""
        file_path = RESULTS / f"{self.key}.json"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "task": self.task,
            "model": self.model,
            "docs": self.docs,
            "completions": self.completions,
            "metrics": self.metrics,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj=payload, fp=f, indent=2, ensure_ascii=False)

        return file_path

    @classmethod
    def from_json(cls, file_path: Path | str) -> Self:
        file_path = Path(file_path)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        task = data["task"]
        model = data["model"]
        metrics = data["metrics"]
        dataset = {
            "docs": data.get("docs", {}),
            "completions": data.get("completions", {}),
        }
        results = {"results": {task: {cls.DEFAULT_METRIC: metrics}}}

        return cls(model=model, results=results, dataset=dataset)


@dataclass
class PulseConfig:
    """Poll configuration, updated by the UI."""

    name: str | None = None
    persona: str | None = None
    docs: str | None = None  # reference to personas file
    question: str | None = None
    answer: str | None = None
    completions: str | None = None  # reference to answers file

    __hash__ = None

    def __eq__(self, other: object) -> bool:
        attrs = ("persona", "question", "answer", "docs", "completions")
        if not isinstance(other, PulseConfig):
            return False

        return all(getattr(self, attr) == getattr(other, attr) for attr in attrs)

    @property
    def data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "persona": self.persona,
            "docs": self.docs,
            "question": self.question,
            "answer": self.answer,
            "completions": self.completions,
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True)

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2)

    def save(self) -> None:
        file_name = f"{self.name}.yaml"
        file_path = TASKS / file_name

        with open(file_path, mode="w", encoding="utf-8") as file:
            file.write(self.to_yaml())

    @classmethod
    def from_yaml(cls, file_path: Path | str) -> Self:
        with open(file_path, encoding="utf-8") as file:
            data = yaml.safe_load(stream=file)

        return cls(**data)


class PulseTask(ConfigurableTask):
    TEST_SPLIT: str = "test"

    def __init__(self, config: dict) -> None:
        super().__init__(config=config)

    def has_test_docs(self):
        return bool(self.test_docs)

    def test_docs(self):
        return self.dataset[self.TEST_SPLIT]

    def construct_requests(self, doc: dict, ctx: str, **kwargs) -> list[Instance] | Instance:
        kwargs.pop("apply_chat_template")
        kwargs.pop("chat_template")

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
        num_choices = len(alias)

        # negative log likelihoods
        lls = [prompt.continuation.logprob for prompt in results]
        lls_a, lls_b = lls[:num_choices], lls[num_choices:]

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
