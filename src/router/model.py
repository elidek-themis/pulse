from pydantic import Field, field_validator
from vllm.entrypoints.openai.protocol import ModelCard, ModelList


class VLLMModel(ModelCard):
    port: int = Field(..., description="Port of the vLLM instance")

    @field_validator("owned_by")
    @classmethod
    def validate_vllm_owner(cls, v: str) -> str:
        if v != "vllm":
            raise ValueError(f"Only vLLM instances accepted, got '{v}'")
        return v


class VLLMModelList(ModelList):
    data: list[ModelCard | VLLMModel] = Field(default_factory=list)
