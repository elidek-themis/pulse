from typing import Final
from operator import itemgetter

from streamlit import logger
from lm_eval.models.api_models import TemplateAPI

from pulse.connection.types import (
    Token,
    Prompt,
    ModelCard,
    SampleRequest,
)

_LOGGER: Final = logger.get_logger(__name__)


class VLLMCompletions(TemplateAPI):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_card: ModelCard,
        **kwargs,
    ):
        self.model_card = model_card

        super().__init__(
            base_url=base_url,  # /v1/completions
            model=model_card.root,  # tokenizer
            **kwargs,
        )

        self.model = model_card.id
        self.api_key = api_key

    @property
    def model_id(self) -> str:
        return self.model_card.id

    def sample(self, requests: list[SampleRequest], **kwargs) -> list[Prompt]:
        assert self.tokenized_requests
        extra_body = kwargs.get("extra_body", {})
        add_generation_prompt = extra_body.pop("add_generation_prompt", True)

        sample_requests = []
        for chat, continuation in [req.args for req in requests]:
            context = self.apply_chat_template(
                chat_history=chat,
                add_generation_prompt=add_generation_prompt,
            )

            context_enc = self.tok_encode(context)
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)
            sample_requests.append((None, context_enc, continuation_enc))

        inputs, ctxlens, _ = self.batch_loglikelihood_requests([sample_requests])
        outputs = self.model_call(messages=inputs, generate=False, **kwargs)
        parsed = self.parse_logprobs(
            outputs=outputs,
            tokens=inputs,
            ctxlens=ctxlens,
            **kwargs,
        )

        return parsed

    def _create_payload(
        self,
        messages: list[list[int]] | list[dict] | list[str] | str,
        generate=False,
        gen_kwargs: dict | None = None,
        seed: int = 2025,
        eos=None,
        **kwargs,
    ) -> dict:
        if generate:
            raise NotImplementedError

        _LOGGER.info(f"kwargs: {kwargs}")
        extra_body = kwargs.pop("extra_body", {})

        to_ret = {
            "model": self.model_id,
            "prompt": messages,
            "temperature": 1,
            "max_tokens": 1,
            "logprobs": 20,
            "seed": seed,
            "echo": True,
            "prompt_logprobs": 1,
            **extra_body,  # will overwrite
        }

        _LOGGER.info(f"payload: {to_ret}")
        return to_ret

    @staticmethod
    def parse_logprobs(
        outputs: dict | list[dict],
        tokens: list[list[int]] = None,
        ctxlens: list[int] = None,
        **kwargs,
    ) -> list[list[Token]]:
        results = []
        if not isinstance(outputs, list):
            outputs = [outputs]

        for out in outputs:
            choice_ctxlen = zip(sorted(out["choices"], key=itemgetter("index")), ctxlens)
            for choice, ctxlen in choice_ctxlen:
                results.append(Prompt(choice=choice, ctxlen=ctxlen))

        return results

    @staticmethod
    def parse_generations(outputs: dict | list[dict], **kwargs) -> Prompt:
        raise NotImplementedError

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        if any(model in self.model.lower() for model in ["gemma", "mistral"]):
            chat_history = self._combine_system(chat_history=chat_history)
        return super().apply_chat_template(chat_history, add_generation_prompt)

    def _combine_system(self, chat_history: list[dict]) -> list[dict]:
        system_msg, *chat = chat_history
        if system_msg["role"] != "system":
            return chat_history

        system_content = system_msg["content"]
        if chat and chat[0]["role"] == "user":
            chat[0]["content"] = system_content + "\n" + chat[0]["content"]

        return chat
