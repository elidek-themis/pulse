from typing import Any

import pandas as pd

from pulse.utils.paths import RESULTS, PERSONAS, COMPLETIONS
from pulse.data.pulse_task import PulseResults
from pulse.data.file_manager import FileManager
from pulse.data.task_manager import TaskManager


class Repository:
    def __init__(self):
        self.completions = FileManager(dir=COMPLETIONS, schema={"A", "B", "alias"})
        self.personas = FileManager(dir=PERSONAS, schema=None)
        self.task_manager = TaskManager()
        self.results: list[PulseResults] = self._read_results()

    @property
    def all_completions(self):
        return self.completions.data.keys()

    @property
    def all_personas(self):
        return self.personas.data.keys()

    @property
    def all_tasks(self):
        pass
        # return self.task_manager.tasks.keys()

    @property
    def runs(self) -> pd.DataFrame:
        cols = ["task", "model", "metrics", "docs", "choices"]

        data = []
        for result in self.results:
            task = self.task_manager[result.task]
            dataset_kwargs = task.dataset_kwargs
            data.append(
                (
                    result.task,
                    result.model,
                    result.metrics,
                    dataset_kwargs["docs"],
                    dataset_kwargs["completions"],
                )
            )

        return pd.DataFrame(data, columns=cols)

    def add_results(self, model: str, results: dict[str, Any]) -> None:
        pulse_results = PulseResults(model=model, results=results)
        pulse_results.save()
        self.results.append(pulse_results)

    def _read_results(self):
        results = []
        for json_file in list(RESULTS.glob("*.json")):
            result = PulseResults.from_json(json_file)
            results.append(result)
        return results
