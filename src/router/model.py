from typing import Any, Literal
from dataclasses import field, dataclass


@dataclass
class Model:
    id: str
    object: str = field(repr=False)
    created: int = field(repr=False)
    root: str
    parent: str | None = field(repr=False)
    max_model_len: int = field(repr=False)
    owned_by: Literal["vllm"] = field(repr=False)
    permission: list[Any] = field(repr=False)

    def __post_init__(self):
        if self.owned_by != "vllm":
            raise ValueError(f"Only `vllm` instances are accepted, got `{self.owned_by}`")


@dataclass
class VLLMModel(Model):
    port: int
    is_sleeping: bool
