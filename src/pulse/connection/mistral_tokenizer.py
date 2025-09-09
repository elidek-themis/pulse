from transformers import MistralCommonTokenizer


class MistralTokenizerWrapper(MistralCommonTokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.pop("add_generation_prompt", None)
        return super().apply_chat_template(*args, **kwargs)

    def _is_control_token(self, token_id: int) -> bool:
        return token_id in self.tokenizer.instruct_tokenizer.tokenizer._control_tokens
