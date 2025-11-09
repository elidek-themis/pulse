from typing import Any
from pathlib import Path

import pandas as pd

from pulse.utils.paths import (
    TASKS,
    RESULTS,
    PERSONAS,
    COMPLETIONS,
)
from pulse.data.pulse_task import PulseTask, PulseResults
from pulse.data.file_manager import FileManager
from pulse.data.task_manager import TaskManager


class Repository:
    COMPLETIONS_SCHEMA: set[str] = {"A", "B", "alias"}

    def __init__(self):
        self.completions = FileManager(dir=COMPLETIONS, schema=self.COMPLETIONS_SCHEMA)
        self.personas = FileManager(dir=PERSONAS, schema=None)
        self.task_manager = TaskManager(dir=TASKS)
        self.results: dict[str, PulseResults] = self._read_results(dir=RESULTS)

    @property
    def completions_keys(self):
        return self.completions.keys

    @property
    def personas_keys(self):
        return self.personas.keys

    @property
    def task_keys(self):
        return self.task_manager.keys

    @property
    def runs(self) -> pd.DataFrame:
        results: list[PulseResults] = self.results.values()

        cols = ["task", "model", "metrics", "docs", "choices"]

        data = []
        for result in results:
            data.append(
                (
                    result.task,
                    result.model,
                    result.metrics,
                    result.docs,
                    result.completions,
                )
            )

        return pd.DataFrame(data, columns=cols)

    def add_results(self, model: str, results: dict[str, Any], dataset: dict[str, Any]) -> None:
        pulse_results = PulseResults(model=model, results=results, dataset=dataset)
        self.results[pulse_results.key] = pulse_results
        pulse_results.save()

    def _read_results(self, dir: Path) -> dict[str, PulseResults]:
        results = {}
        for json_file in list(dir.glob("*.json")):
            result = PulseResults.from_json(json_file)
            results[json_file.stem] = result
        return results

    def resolve_task(self, task_name: str) -> PulseTask:
        defaults = {
            "doc_to_target": -1,
            "output_type": "loglikelihood",
        }

        task = self.task_manager[task_name]
        docs = self.personas[task.docs].to_dict(orient="records") if task.docs else None
        completions = self.completions[task.completions].to_dict(orient="list")

        task_dict = {
            **defaults,
            "task": task.name,
            "description": task.persona,
            "doc_to_text": task.question,
            "gen_prefix": task.answer,
            "dataset_kwargs": {
                "docs": docs,
                "completions": completions,
            },
        }

        return PulseTask(config=task_dict)
