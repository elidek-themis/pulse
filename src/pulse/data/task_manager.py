from copy import copy
from enum import StrEnum

from pulse.utils.paths import TASKS
from pulse.data.pulse_task import PulseConfig


class TaskStatus(StrEnum):
    OK = "ok"
    EXISTS = "Task name already exists"
    DUPLICATE = "Task config already exists"


class TaskManager:
    def __init__(self):
        self.tasks: dict[str, PulseConfig] = self._read_tasks()

    def _read_tasks(self):
        tasks = {}

        if not TASKS.exists() or not TASKS.is_dir():
            return tasks

        for yaml_file in list(TASKS.glob("*.yaml")):
            task_config = PulseConfig.from_yaml(yaml_file)
            tasks[task_config.task] = task_config

        return tasks

    def add(self, task_config: PulseConfig) -> TaskStatus:
        # allow task overwrite
        # if self._name_exists(name=task_config.task):
        #     return TaskStatus.EXISTS

        if self._is_duplicate(task_config=task_config):
            return TaskStatus.DUPLICATE

        task_config.save()
        self.tasks[task_config.task] = copy(task_config)

        return TaskStatus.OK

    def delete(self, task_name: str) -> TaskStatus:
        if not self._name_exists(name=task_name):
            return TaskStatus.EXISTS

        self.tasks.pop(task_name)
        path = TASKS / f"{task_name}.yaml"
        path.unlink(missing_ok=True)
        return TaskStatus.OK

    def _name_exists(self, name: str) -> bool:
        return name in self

    def _is_duplicate(self, task_config: PulseConfig) -> bool:
        all_tasks = self.tasks.values()
        return any(task_config == task for task in all_tasks)

    def __contains__(self, task: str) -> bool:
        return task in self.tasks

    def __getitem__(self, key: str) -> PulseConfig:
        return self.tasks[key]
