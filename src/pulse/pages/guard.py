from typing import Self, Final, NamedTuple

from streamlit import session_state as ss

from pulse.utils.tools import contains_placeholder


class Clause(NamedTuple):
    cond: bool
    msg: str

    def __bool__(self) -> bool:
        return self.cond

    def __invert__(self) -> Self:
        return Clause(not self.cond, self.msg)


class Guard:
    @property
    def is_connected(self) -> Clause:
        attr = ss.get("vllm_conn")

        return Clause(
            cond=bool(attr),
            msg="Enter OpenAI compatible server credentials.",
        )

    @property
    def has_selected_model(self) -> Clause:
        attr = ss.get("selected_model")

        return Clause(
            cond=bool(attr),
            msg="Select one of the available models.",
        )

    @property
    def has_chat_template(self) -> Clause:
        msg = "Selected model has no chat template."

        try:
            attr = ss.client.lm.tokenizer.chat_template
            cond = bool(attr)
        except AttributeError:
            cond = False

        return Clause(
            cond=cond,
            msg=msg,
        )

    @property
    def has_prompts(self) -> Clause:
        attrs = ("persona", "question", "answer")

        return Clause(
            cond=all(ss.get(attr) for attr in attrs),
            msg="Incomplete prompt configuration.",
        )

    @property
    def has_personas(self) -> Clause:
        attr = ss.get("selected_persona")

        return Clause(
            cond=bool(attr),
            msg="No persona selected.",
        )

    @property
    def has_valid_personas(self) -> Clause:
        task_config = ss.get("task_config")
        persona_file = task_config.docs

        return Clause(
            cond=bool(persona_file) or persona_file not in ss.repo.personas,
            msg=f"Persona file '{persona_file}' not found.",
        )

    @property
    def has_completions(self) -> Clause:
        attr = ss.get("selected_completions")

        return Clause(
            cond=bool(attr),
            msg="No completions selected.",
        )

    @property
    def has_valid_completions(self) -> Clause:
        task_config = ss.get("task_config")
        completions = task_config.completions

        return Clause(
            cond=completions in ss.repo.completions,
            msg=f"Completions file '{completions}' not found.",
        )

    @property
    def has_task(self) -> Clause:
        attr = ss.get("selected_task")

        return Clause(
            cond=bool(attr),
            msg="No task selected.",
        )

    @property
    def connection_guards(self) -> list[Clause]:
        return [
            ~self.is_connected,
            ~self.has_selected_model,
            ~self.has_chat_template,
        ]

    @property
    def save_guards(self) -> list[Clause]:
        has_placeholder = contains_placeholder(s=ss.persona)
        has_personas = self.has_personas

        ph_yes_batch = Clause(
            cond=has_placeholder and not has_personas,
            msg="Persona contains placeholder but no file selected.",
        )

        ph_no_batch = Clause(
            cond=not has_placeholder and bool(has_personas),
            msg="Batch personas selected but no placeholder in prompt.",
        )

        return [
            ~self.has_prompts,
            ph_yes_batch,
            ph_no_batch,
            ~self.has_completions,
        ]

    @property
    def is_update(self) -> Clause:
        if not (selected := ss.get("selected_task")):
            return Clause(cond=False, msg="")

        return Clause(
            cond=ss.task_config.name == selected,
            msg="Update.",
        )

    @property
    def run_guards(self) -> list[Clause]:
        return [
            *self.connection_guards,
            ~self.has_task,
            *self.save_guards,
        ]

    @property
    def rank_guards(self) -> list[Clause]:
        has_placeholder = Clause(
            cond=contains_placeholder(s=ss.persona),
            msg="Persona prompt contains placeholder.",
        )

        return [
            *self.connection_guards,
            ~self.has_prompts,
            has_placeholder,
            ~self.has_completions,
        ]

    def validate(self, attr: str) -> Clause | None:
        """Returns first failing clause, or None if all pass."""
        guards = getattr(self, f"{attr}")

        if not isinstance(guards, list):
            guards = [guards]

        for guard in guards:
            if guard:
                return guard

        return None


GUARD: Final = Guard()
