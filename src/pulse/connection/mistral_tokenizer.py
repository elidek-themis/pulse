from transformers.tokenization_mistral_common import MistralTokenizerType, MistralCommonTokenizer


class MistralTokenizerWrapper(MistralCommonTokenizer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        kwargs.pop("add_generation_prompt", None)
        return super().apply_chat_template(*args, **kwargs)

    def _is_control_token(self, token_id: int) -> bool:
        """Patch to access _control_tokens as a property and not a method"""
        if self._tokenizer_type == MistralTokenizerType.spm:
            return token_id in self.tokenizer.instruct_tokenizer.tokenizer._control_tokens
        elif self._tokenizer_type == MistralTokenizerType.tekken:
            return token_id < self.tokenizer.instruct_tokenizer.tokenizer.num_special_tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        """vLLM patch to ignore max_loras argument"""
        kwargs.pop("max_loras", None)
        kwargs.pop("_from_auto", None)

        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)
