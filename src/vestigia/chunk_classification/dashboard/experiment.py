"""Reproducible dashboard experiments: chunks, model settings, prompts, results.

Examples
--------
>>> chunks = normalize_chunks([("intro", "Evidence"), ("", "More evidence")])
>>> [chunk.id for chunk in chunks]
['intro', '2']
"""

from collections.abc import Iterable
import re
from typing import Any, Literal

import yaml
from langchain_core.prompts import PromptTemplate
from pydantic import Field, field_validator, model_validator

from vestigia.chunk_classification import (
    Chunk,
    ChunkStatus,
    EvidenceModel,
    SystemOneModelTemplate,
)
from vestigia.chunk_classification.classifier import ChunkClassifier
from vestigia.chunk_classification.prompts import (
    load_chat_model_prompt,
    load_system_one_prompt,
)

Provider = Literal["openai", "anthropic", "typesafe"]
CHAT_PROVIDERS: tuple[Provider, ...] = ("openai", "anthropic")
CHAT_PROMPT_VARIABLES = {"query", "answer", "chunks"}


def natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


class ModelConfig(EvidenceModel):
    """Provider settings needed to rebuild a classifier, excluding secrets.

    Examples
    --------
    >>> ModelConfig(provider="openai", model="gpt-4o-mini").temperature is None
    True
    """

    provider: Provider = Field(..., description="Backend used for classification.")
    model: str = Field(..., min_length=1, description="Provider model name.")
    base_url: str | None = Field(
        default=None,
        description="Optional API endpoint, such as a local OpenAI-compatible server.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Optional sampling temperature for chat providers.",
    )

    @model_validator(mode="after")
    def _temperature_only_for_chat(self) -> "ModelConfig":
        if self.temperature is not None and self.provider not in CHAT_PROVIDERS:
            raise ValueError("Temperature is only supported by chat providers")
        return self


class ChatPrompt(EvidenceModel):
    """Prompt template used by chat-model classification.

    Examples
    --------
    >>> ChatPrompt(template="{query} {answer} {chunks}").template
    '{query} {answer} {chunks}'
    """

    template: str = Field(
        ...,
        description="Template with query, answer, and chunks placeholders.",
    )


class Evaluation(EvidenceModel):
    """Classification output stored for one chunk.

    Examples
    --------
    >>> Evaluation(status="used").status
    <ChunkStatus.USED: 'used'>
    """

    status: ChunkStatus = Field(..., description="Assigned evidence status.")
    explanation: str | None = Field(
        default=None,
        description="Chat-model explanation, when returned.",
    )
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="System-one confidence in [0, 1].",
    )
    probabilities: dict[ChunkStatus, float] | None = Field(
        default=None,
        description="System-one per-status probabilities.",
    )


class Experiment(EvidenceModel):
    """Everything needed to rerun a classification, plus its latest results.

    Examples
    --------
    >>> experiment = Experiment(
    ...     model=ModelConfig(provider="openai", model="gpt-4o-mini"),
    ...     prompt=ChatPrompt(template="{query} {answer} {chunks}"),
    ...     input="Question?",
    ...     output="Answer.",
    ...     chunks={"1": "Evidence"},
    ... )
    >>> from_yaml(to_yaml(experiment)) == experiment
    True
    """

    version: Literal[1] = Field(default=1, description="Export format version.")
    model: ModelConfig = Field(..., description="Classifier settings.")
    prompt: ChatPrompt | SystemOneModelTemplate = Field(
        ...,
        description="Chat template or system-one state, instruction, and criteria.",
    )
    input: str = Field(..., description="User query given to the generating model.")
    output: str = Field(..., description="Generated answer to attribute.")
    chunks: dict[str, str] = Field(
        ...,
        min_length=1,
        description="Source chunk text keyed by chunk ID, in input order.",
    )
    evaluations: dict[str, Evaluation] = Field(
        default_factory=dict,
        description="Latest results keyed by chunk ID; empty before evaluation.",
    )

    @field_validator("chunks", "evaluations", mode="before")
    @classmethod
    def _string_keys(cls, value: Any) -> Any:
        # Hand-written YAML parses unquoted numeric IDs as integers.
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "Experiment":
        is_chat = self.model.provider in CHAT_PROVIDERS
        if is_chat != isinstance(self.prompt, ChatPrompt):
            raise ValueError(f"Prompt type does not match provider {self.model.provider!r}")
        if self.evaluations and self.evaluations.keys() != self.chunks.keys():
            raise ValueError("Evaluation IDs must match chunk IDs exactly")
        return self


def default_prompt(provider: Provider) -> ChatPrompt | SystemOneModelTemplate:
    """Return the packaged prompt for a provider.

    Examples
    --------
    >>> isinstance(default_prompt("openai"), ChatPrompt)
    True
    """
    if provider in CHAT_PROVIDERS:
        return ChatPrompt(template=load_chat_model_prompt())
    return load_system_one_prompt()


def normalize_chunks(rows: Iterable[tuple[str | None, str | None]]) -> list[Chunk]:
    """Build chunks from ``(id, text)`` rows, numbering blank IDs by position.

    Fully blank rows are ignored. A blank ID becomes the row's 1-based position.

    Raises
    ------
    ValueError
        If a row has no text or two chunks share an ID.

    Examples
    --------
    >>> [c.id for c in normalize_chunks([("", "a"), (None, None), ("x", "b")])]
    ['1', 'x']
    >>> normalize_chunks([("2", "a"), ("", "b")])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Duplicate chunk ID '2'...
    """
    filled = [
        ((chunk_id or "").strip(), text or "")
        for chunk_id, text in rows
        if (chunk_id or "").strip() or (text or "").strip()
    ]
    chunks: dict[str, Chunk] = {}
    for position, (chunk_id, text) in enumerate(filled, start=1):
        chunk_id = chunk_id or str(position)
        if not text.strip():
            raise ValueError(f"Chunk {chunk_id!r} has no text")
        if chunk_id in chunks:
            raise ValueError(
                f"Duplicate chunk ID {chunk_id!r}; give each chunk a unique ID"
            )
        chunks[chunk_id] = Chunk(id=chunk_id, text=text)
    return list(chunks.values())


def parse_chunk_file(content: str) -> list[Chunk]:
    """Read chunks from YAML or JSON in one of three shapes.

    Accepted shapes: a mapping of ID to text, a list of ``{id, text}`` objects
    with optional ``id``, or a list of strings.

    Examples
    --------
    >>> [c.id for c in parse_chunk_file('{"a": "x", "b": "y"}')]
    ['a', 'b']
    >>> [c.id for c in parse_chunk_file("- text: x\\n- id: k\\n  text: y")]
    ['1', 'k']
    >>> [c.text for c in parse_chunk_file('["x", "y"]')]
    ['x', 'y']
    """
    data = yaml.safe_load(content)
    rows: list[tuple[str, str]] = []
    if isinstance(data, dict) and data:
        rows = [(str(key), _text(value)) for key, value in data.items()]
    elif isinstance(data, list) and data:
        for item in data:
            if isinstance(item, str):
                rows.append(("", item))
            elif isinstance(item, dict) and "text" in item and item.keys() <= {"id", "text"}:
                chunk_id = item.get("id")
                rows.append(("" if chunk_id is None else str(chunk_id), _text(item["text"])))
            else:
                raise ValueError("Each list item must be a string or an {id, text} object")
    else:
        raise ValueError(
            "Chunk files must contain a mapping of ID to text or a non-empty list"
        )
    return normalize_chunks(rows)


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Chunk text must be a string")
    return value


def validate_prompt(prompt: ChatPrompt | SystemOneModelTemplate) -> None:
    """Reject templates whose placeholders the classifier cannot fill.

    Examples
    --------
    >>> validate_prompt(ChatPrompt(template="{query} {answer}"))
    Traceback (most recent call last):
    ...
    ValueError: The chat prompt must use exactly {answer}, {chunks}, {query}; missing {chunks}
    """
    if isinstance(prompt, ChatPrompt):
        _check_variables("chat prompt", prompt.template, CHAT_PROMPT_VARIABLES, exact=True)
    else:
        _check_variables("state", prompt.state, {"query", "answer"})
        _check_variables("instruction", prompt.instruction, {"chunk"})


def _check_variables(name: str, template: str, allowed: set[str], exact: bool = False) -> None:
    used = set(PromptTemplate.from_template(template).input_variables)
    if unknown := used - allowed:
        raise ValueError(
            f"The {name} uses unknown placeholders: "
            + ", ".join(f"{{{v}}}" for v in sorted(unknown))
        )
    if exact and (missing := allowed - used):
        raise ValueError(
            f"The {name} must use exactly "
            + ", ".join(f"{{{v}}}" for v in sorted(allowed))
            + "; missing "
            + ", ".join(f"{{{v}}}" for v in sorted(missing))
        )


def build_classifier(config: ModelConfig, api_key: str | None) -> ChunkClassifier:
    """Create the configured classifier; unset options fall back to provider defaults.

    Examples
    --------
    >>> config = ModelConfig(provider="openai", model="m", base_url="http://localhost:1234/v1")
    >>> type(build_classifier(config, api_key="lm-studio")).__name__
    'ChatModelChunkClassifier'
    """
    options = {
        key: value
        for key, value in {
            "model": config.model,
            "base_url": config.base_url,
            "api_key": api_key,
            "temperature": config.temperature,
        }.items()
        if value is not None
    }
    if config.provider == "typesafe":
        from langchain_typesafe import TypeSafeClassifier

        from vestigia.chunk_classification.langchain import SystemOneModelChunkClassifier

        return SystemOneModelChunkClassifier(TypeSafeClassifier(**options))

    from vestigia.chunk_classification.langchain import ChatModelChunkClassifier

    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatModelChunkClassifier(ChatAnthropic(**options))
    from langchain_openai import ChatOpenAI

    return ChatModelChunkClassifier(ChatOpenAI(**options))


async def evaluate(experiment: Experiment, api_key: str | None) -> Experiment:
    """Classify the experiment's chunks and return it with fresh evaluations.

    Provider and validation errors propagate to the caller.

    Examples
    --------
    Within an async function:

        evaluated = await evaluate(experiment, api_key=None)
    """
    validate_prompt(experiment.prompt)
    classifier = build_classifier(experiment.model, api_key)
    prompt = experiment.prompt
    result = await classifier.classify(
        query=experiment.input,
        answer=experiment.output,
        chunks=[Chunk(id=key, text=text) for key, text in experiment.chunks.items()],
        prompt=PromptTemplate.from_template(prompt.template)
        if isinstance(prompt, ChatPrompt)
        else prompt,
    )
    evaluations = {
        item.chunk.id: Evaluation(
            status=item.status,
            explanation=getattr(item, "explanation", None),
            confidence=getattr(item, "confidence", None),
            probabilities=getattr(item, "probabilities", None),
        )
        for item in result.classifications
    }
    return Experiment.model_validate(
        experiment.model_dump() | {"evaluations": evaluations}
    )


class _BlockDumper(yaml.SafeDumper):
    pass


_BlockDumper.add_representer(
    str,
    lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    ),
)


def to_yaml(experiment: Experiment) -> str:
    """Serialize an experiment; multi-line text uses readable block scalars.

    Examples
    --------
    >>> experiment = Experiment(
    ...     model=ModelConfig(provider="openai", model="m"),
    ...     prompt=ChatPrompt(template="{query} {answer} {chunks}"),
    ...     input="Q", output="A", chunks={"1": "Evidence"},
    ... )
    >>> to_yaml(experiment).splitlines()[0]
    'version: 1'
    """
    return yaml.dump(
        experiment.model_dump(mode="json", exclude_none=True),
        Dumper=_BlockDumper,
        sort_keys=False,
        allow_unicode=True,
    )


def from_yaml(content: str) -> Experiment:
    """Parse and validate an exported experiment.

    Raises
    ------
    ValueError
        If the content is not a valid experiment (``pydantic.ValidationError``
        is a ``ValueError``).

    Examples
    --------
    >>> from_yaml("version: 2")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    pydantic_core._pydantic_core.ValidationError: ...
    """
    return Experiment.model_validate(yaml.safe_load(content))
