"""Turn exceptions into dashboard error reports with plain-language guidance.

Each report has a short title, a Markdown message that names the setting to
check, and the original exception text for the "Technical details" section.

Examples
--------
>>> from langchain_core.exceptions import ModelConnectionError
>>> from vestigia.chunk_classification.dashboard.experiment import ModelConfig
>>> config = ModelConfig(provider="openai", model="m", base_url="http://localhost:1234/v1")
>>> describe_evaluation_error(ModelConnectionError("Connection error."), config).title
"Can't reach the model server"
"""

from collections.abc import Callable, Iterable
import traceback
from typing import Any, Literal

import yaml
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from pydantic import Field, ValidationError

from vestigia.chunk_classification import EvidenceModel
from vestigia.chunk_classification.classifier import ModelResponseError
from vestigia.chunk_classification.dashboard.experiment import (
    API_KEY_ENV,
    ChunkFormatError,
    MissingAPIKeyError,
    ModelConfig,
    Provider,
)

PROVIDER_NAMES: dict[Provider, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "typesafe": "TypeSafe",
}
# Union members appear in Pydantic error locations but mean nothing to users.
_HIDDEN_LOCATION_PARTS = {"ChatPrompt", "SystemOneModelTemplate"}
_EXPERIMENT_SECTIONS = {"model", "prompt", "chunks"}
CHUNK_FORMAT_EXAMPLE = """```yaml
intro: The first chunk's text.
details: The second chunk's text.
```"""


class ErrorReport(EvidenceModel):
    """A user-facing description of a failure.

    Examples
    --------
    >>> ErrorReport(title="Oops", message="Try again.", details="RuntimeError: boom").title
    'Oops'
    """

    title: str = Field(..., description="Short plain-language summary.")
    message: str = Field(
        ...,
        description="Markdown explanation naming what to check or change.",
    )
    details: str = Field(
        ...,
        description="Original exception type and message, including causes.",
    )


_Report = Callable[[str, str], ErrorReport]


def technical_details(exc: BaseException) -> str:
    """Format an exception and its causes as ``module.Type: message`` lines.

    Examples
    --------
    >>> try:
    ...     try:
    ...         raise OSError("refused")
    ...     except OSError as cause:
    ...         raise RuntimeError("Connection error.") from cause
    ... except RuntimeError as exc:
    ...     print(technical_details(exc))
    RuntimeError: Connection error.
    Caused by: OSError: refused
    """
    lines: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = "".join(traceback.format_exception_only(current)).strip()
        lines.append(text if not lines else f"Caused by: {text}")
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return "\n".join(lines)


def describe_evaluation_error(
    exc: BaseException, config: ModelConfig | None, api_key_from_field: bool = False
) -> ErrorReport:
    """Describe a failure raised while preparing or running an evaluation.

    ``config`` is ``None`` when the failure happened while reading the setup.
    ``api_key_from_field`` tells whether the key came from the dashboard field
    rather than the environment.

    Examples
    --------
    >>> from vestigia.chunk_classification.dashboard.experiment import MissingAPIKeyError
    >>> config = ModelConfig(provider="typesafe", model="jev-latest")
    >>> report = describe_evaluation_error(MissingAPIKeyError("typesafe"), config)
    >>> report.title
    'No API key'
    >>> "`TYPESAFE_API_KEY`" in report.message
    True
    """
    details = technical_details(exc)

    def report(title: str, message: str) -> ErrorReport:
        return ErrorReport(title=title, message=message, details=details)

    if isinstance(exc, MissingAPIKeyError):
        hint = (
            " Local servers such as LM Studio accept any value."
            if exc.provider == "openai"
            else ""
        )
        return report(
            "No API key",
            f"Enter a key in **API key** or set `{exc.env_var}` before starting "
            f"the dashboard.{hint}",
        )
    if isinstance(exc, ModelResponseError):
        return _response_report(exc, config, report)
    if config is None or not _is_provider_error(exc):
        return _setup_report(exc, report)

    model = f"`{config.model}`"
    name = PROVIDER_NAMES[config.provider]
    endpoint = f"`{config.base_url}`" if config.base_url else f"the {name} API"
    endpoint_title = endpoint[0].upper() + endpoint[1:]
    key_source = (
        "the **API key** field"
        if api_key_from_field
        else f"`{API_KEY_ENV[config.provider]}`"
    )
    if isinstance(exc, ModelTimeoutError):
        return report(
            "The model server timed out",
            f"{endpoint_title} took too long to respond. Try again, or use fewer "
            "or shorter chunks.",
        )
    if isinstance(exc, ModelConnectionError):
        check = (
            "Check that the server is running and the **Base URL** is correct."
            if config.base_url
            else "Check your network connection, or set a **Base URL** for a local "
            "server."
        )
        return report("Can't reach the model server", f"No response from {endpoint}. {check}")
    if isinstance(exc, ModelAuthenticationError):
        return report(
            "API key rejected",
            f"{endpoint_title} did not accept the key from {key_source}. Check that "
            "it is correct and not expired.",
        )
    if isinstance(exc, ModelPermissionDeniedError):
        return report(
            "Access denied",
            f"The key from {key_source} is valid but cannot use {model}. Check the "
            "account's permissions.",
        )
    if isinstance(exc, ModelNotFoundError):
        local = ", or the **Base URL**" if config.base_url else ""
        return report(
            "Model not found",
            f"{endpoint_title} has no model named {model}. Check the **Model** "
            f"name{local}.",
        )
    if isinstance(exc, ModelRateLimitError):
        return report(
            "Rate limit or quota reached",
            f"{endpoint_title} is limiting requests. Wait a moment and try again, "
            "or check your plan's quota.",
        )
    if isinstance(exc, ContextOverflowError):
        return report(
            "Input too long for the model",
            f"The prompt, answer, and chunks exceed what {model} can read at once. "
            "Use fewer or shorter chunks, or a model with a larger context.",
        )
    if isinstance(exc, ModelInvalidRequestError):
        return report(
            "Request rejected",
            f"{endpoint_title} rejected the request for {model}. A setting such as "
            "**Temperature** or the prompt may be unsupported; see the technical "
            "details.",
        )
    return report(
        "Model server error",
        f"{endpoint_title} returned an error. Try again later; the technical "
        "details show the server's message.",
    )


def _is_provider_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            ModelTimeoutError,
            ModelConnectionError,
            ModelAuthenticationError,
            ModelPermissionDeniedError,
            ModelNotFoundError,
            ModelRateLimitError,
            ContextOverflowError,
            ModelInvalidRequestError,
            ModelAPIError,
        ),
    )


def _response_report(
    exc: ModelResponseError, config: ModelConfig | None, report: _Report
) -> ErrorReport:
    model = f"`{config.model}`" if config else "The model"
    advice = "Try again, or use a more capable model."
    groups = [
        f"{label}: {', '.join(f'`{chunk_id}`' for chunk_id in ids)}"
        for label, ids in (
            ("missing", exc.missing),
            ("extra", exc.extra),
            ("repeated", exc.duplicates),
        )
        if ids
    ]
    if groups:
        message = (
            f"{model} returned results that don't match the chunks "
            f"({'; '.join(groups)}). {advice}"
        )
    else:
        message = (
            f"{model} returned a response that could not be read as chunk "
            f"classifications. {advice}"
        )
    return report("Unexpected model response", message)


def _setup_report(exc: BaseException, report: _Report) -> ErrorReport:
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        if all(
            error["type"] == "string_too_short" and error["loc"][-1:] == ("model",)
            for error in errors
        ):
            return report("Model name missing", "Enter a model name in **Model**.")
        return report("Check the setup", _bullets(_field_problems(errors)))
    if isinstance(exc, ValueError):
        return report("Check the setup", _sentence(str(exc)))
    return report(
        "Evaluation failed",
        "Something unexpected went wrong. The technical details below show the "
        "original error.",
    )


def describe_import_error(
    exc: BaseException, kind: Literal["experiment", "chunks"]
) -> ErrorReport:
    """Describe a failure raised while importing an experiment or chunk file.

    Examples
    --------
    >>> import yaml
    >>> try:
    ...     yaml.safe_load("a: [1, 2")
    ... except yaml.YAMLError as exc:
    ...     report = describe_import_error(exc, "experiment")
    >>> report.title, report.message
    ('Not a valid YAML file', 'The file has a syntax error at line 1, column 9.')
    """
    details = technical_details(exc)

    def report(title: str, message: str) -> ErrorReport:
        return ErrorReport(title=title, message=message, details=details)

    if isinstance(exc, UnicodeDecodeError):
        return report("Can't read the file", "Save it as UTF-8 text and try again.")
    if isinstance(exc, yaml.YAMLError):
        title = "Not a valid YAML file" if kind == "experiment" else "Not valid YAML or JSON"
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        return report(title, f"The file has a syntax error{where}.")
    if kind == "chunks":
        if isinstance(exc, ChunkFormatError):
            return report(
                "Unrecognized chunk format",
                f"{_sentence(str(exc))} Use a mapping of ID to text, a list of "
                "`{id, text}` objects, or a list of strings, for example:\n\n"
                + CHUNK_FORMAT_EXAMPLE,
            )
        if isinstance(exc, ValueError):
            return report("Some chunks are invalid", _sentence(str(exc)))
    if kind == "experiment" and isinstance(exc, ValidationError):
        errors = exc.errors()
        if _is_not_an_experiment(errors):
            return report(
                "Not a Vestigia experiment",
                "Export an experiment from the dashboard and import that file.",
            )
        for error in errors:
            if error["loc"] == ("version",):
                return report(
                    "Unsupported file version",
                    f"This file uses version `{error['input']}`; this dashboard reads "
                    "version 1.",
                )
        return report("Some fields are invalid", _bullets(_field_problems(errors)))
    return report(
        "Import failed",
        "Something unexpected went wrong. The technical details below show the "
        "original error.",
    )


def _is_not_an_experiment(errors: Iterable[Any]) -> bool:
    errors = list(errors)
    if any(error["type"] == "model_type" and not error["loc"] for error in errors):
        return True
    missing = {error["loc"][0] for error in errors if error["type"] == "missing"}
    return _EXPERIMENT_SECTIONS <= missing


def _field_problems(errors: Iterable[Any]) -> list[str]:
    problems: list[str] = []
    for error in errors:
        parts = [str(part) for part in error["loc"] if part not in _HIDDEN_LOCATION_PARTS]
        problem = _problem(error)
        item = f"**{' → '.join(parts)}** {problem}" if parts else _sentence(problem)
        if item not in problems:
            problems.append(item)
    return problems


def _problem(error: Any) -> str:
    ctx = error.get("ctx") or {}
    kind = error["type"]
    if kind == "missing":
        return "is required"
    if kind == "extra_forbidden":
        return "is not a recognized field"
    if kind in {"literal_error", "enum"}:
        return f"must be one of {ctx['expected']}"
    if kind == "string_too_short":
        return "must not be empty"
    if kind == "greater_than_equal":
        return f"must be at least {ctx['ge']}"
    if kind == "less_than_equal":
        return f"must be at most {ctx['le']}"
    if kind == "value_error":
        return str(ctx.get("error", error["msg"]))
    return error["msg"][0].lower() + error["msg"][1:]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."
