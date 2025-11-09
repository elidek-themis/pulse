from copy import copy
from enum import StrEnum
from pathlib import Path

from pulse.utils.paths import TASKS
from pulse.data.pulse_task import PulseConfig


class TaskStatus(StrEnum):
    OK = "ok"
    EXISTS = "Task name already exists"
    DUPLICATE = "Task config already exists"


class TaskManager:
    def __init__(self, dir: Path):
        self.tasks: dict[str, PulseConfig] = self._read_tasks(dir=dir)

    def __contains__(self, task: str) -> bool:
        return task in self.tasks

    def __getitem__(self, key: str) -> PulseConfig:
        return copy(self.tasks[key])

    def add(self, task_config: PulseConfig) -> TaskStatus:
        if self._is_duplicate(task_config=task_config):
            return TaskStatus.DUPLICATE

        task_config.save()
        self.tasks[task_config.name] = copy(task_config)

        return TaskStatus.OK

    def delete(self, task_name: str) -> TaskStatus:
        if task_name not in self:
            return TaskStatus.EXISTS

        del self.tasks[task_name]
        path = TASKS / f"{task_name}.yaml"
        path.unlink(missing_ok=True)

        return TaskStatus.OK

    def _read_tasks(self, dir: Path):
        tasks = {}

        if not dir.exists() or not dir.is_dir():
            return tasks

        for yaml_file in list(dir.glob("*.yaml")):
            task_config = PulseConfig.from_yaml(yaml_file)
            tasks[task_config.name] = task_config

        return tasks

    def _is_duplicate(self, task_config: PulseConfig) -> bool:
        all_tasks = self.tasks.values()
        return any(task_config == task for task in all_tasks)

    @property
    def keys(self) -> list[str]:
        return list(self.tasks.keys())
