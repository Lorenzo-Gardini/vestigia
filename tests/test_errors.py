"""User-facing dashboard error reports, built from real exception types."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("yaml")
pytest.importorskip("langchain_core")

import yaml  # noqa: E402
from langchain_core.exceptions import (  # noqa: E402
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

from vestigia.chunk_classification.classifier import ModelResponseError  # noqa: E402
from vestigia.chunk_classification.dashboard import errors  # noqa: E402
from vestigia.chunk_classification.dashboard import experiment as ex  # noqa: E402

LOCAL = ex.ModelConfig(
    provider="openai", model="qwen/qwen3.5-9b", base_url="http://localhost:1234/v1"
)
HOSTED = ex.ModelConfig(provider="anthropic", model="claude-sonnet-4-5")


def _experiment_yaml(**changes: object) -> str:
    experiment = ex.Experiment(
        model=LOCAL,
        prompt=ex.ChatPrompt(template="{query} {answer} {chunks}"),
        input="Q",
        output="A",
        chunks={"a": "Evidence"},
    )
    return ex.to_yaml(experiment.model_copy(update=changes))


def _raised(content: str, parse) -> Exception:
    try:
        parse(content)
    except Exception as exc:
        return exc
    raise AssertionError("expected an error")


class _TimeoutAndConnection(ModelConnectionError, ModelTimeoutError):
    """Mirrors TypeSafe's timeout, which is also a connection error."""


@pytest.mark.parametrize(
    ("exc", "config", "title", "fragment"),
    [
        (
            ModelConnectionError("Connection error."),
            LOCAL,
            "Can't reach the model server",
            "No response from `http://localhost:1234/v1`. Check that the server is "
            "running and the **Base URL** is correct.",
        ),
        (
            ModelConnectionError("Connection error."),
            HOSTED,
            "Can't reach the model server",
            "No response from the Anthropic API. Check your network connection",
        ),
        (
            _TimeoutAndConnection("Timed out."),
            LOCAL,
            "The model server timed out",
            "`http://localhost:1234/v1` took too long to respond.",
        ),
        (
            ModelAuthenticationError("401"),
            HOSTED,
            "API key rejected",
            "The Anthropic API did not accept the key from `ANTHROPIC_API_KEY`.",
        ),
        (
            ModelPermissionDeniedError("403"),
            HOSTED,
            "Access denied",
            "cannot use `claude-sonnet-4-5`",
        ),
        (
            ModelNotFoundError("404"),
            LOCAL,
            "Model not found",
            "has no model named `qwen/qwen3.5-9b`. Check the **Model** name, or the "
            "**Base URL**.",
        ),
        (ModelRateLimitError("429"), HOSTED, "Rate limit or quota reached", "Wait a moment"),
        (
            ContextOverflowError("too long"),
            LOCAL,
            "Input too long for the model",
            "Use fewer or shorter chunks",
        ),
        (
            ModelInvalidRequestError("400"),
            LOCAL,
            "Request rejected",
            "**Temperature**",
        ),
        (ModelAPIError("500"), HOSTED, "Model server error", "Try again later"),
    ],
)
def test_provider_errors_name_the_setting_to_check(
    exc: Exception, config: ex.ModelConfig, title: str, fragment: str
) -> None:
    report = errors.describe_evaluation_error(exc, config)

    assert report.title == title
    assert fragment in report.message
    assert report.details.endswith(str(exc))


def test_rejected_key_points_to_the_field_when_typed_in() -> None:
    report = errors.describe_evaluation_error(
        ModelAuthenticationError("401"), HOSTED, api_key_from_field=True
    )

    assert "the key from the **API key** field" in report.message


def test_missing_key_names_the_environment_variable() -> None:
    report = errors.describe_evaluation_error(ex.MissingAPIKeyError("openai"), LOCAL)

    assert report.title == "No API key"
    assert "`OPENAI_API_KEY`" in report.message
    assert "LM Studio" in report.message


def test_mismatched_response_lists_chunk_ids() -> None:
    exc = ModelResponseError("IDs must match", missing=["a"], extra=["z"])

    report = errors.describe_evaluation_error(exc, LOCAL)

    assert report.title == "Unexpected model response"
    assert "(missing: `a`; extra: `z`)" in report.message


def test_unreadable_response_keeps_the_parser_error_in_details() -> None:
    try:
        try:
            raise ValueError("Invalid JSON output")
        except ValueError as cause:
            raise ModelResponseError("The model response could not be read") from cause
    except ModelResponseError as exc:
        report = errors.describe_evaluation_error(exc, LOCAL)

    assert "could not be read as chunk classifications" in report.message
    assert "Caused by: ValueError: Invalid JSON output" in report.details


@pytest.mark.parametrize(
    ("exc", "title", "message"),
    [
        (
            _raised("", lambda model: ex.ModelConfig(provider="openai", model=model)),
            "Model name missing",
            "Enter a model name in **Model**.",
        ),
        (
            ValueError("Add at least one chunk with text before evaluating"),
            "Check the setup",
            "Add at least one chunk with text before evaluating.",
        ),
        (
            _raised("{query}", lambda t: ex.validate_prompt(ex.ChatPrompt(template=t))),
            "Check the setup",
            "The chat prompt is missing {answer} and {chunks}; it must use {answer}, "
            "{chunks}, and {query}.",
        ),
        (
            RuntimeError("boom"),
            "Evaluation failed",
            "Something unexpected went wrong. The technical details below show the "
            "original error.",
        ),
    ],
)
def test_setup_errors_before_a_call(exc: Exception, title: str, message: str) -> None:
    report = errors.describe_evaluation_error(exc, None)

    assert (report.title, report.message) == (title, message)


@pytest.mark.parametrize(
    ("content", "title", "fragment"),
    [
        ("model: [unclosed", "Not a valid YAML file", "line 1, column 17"),
        ("not: an experiment", "Not a Vestigia experiment", "Export an experiment"),
        ("- a list", "Not a Vestigia experiment", "Export an experiment"),
        (
            _experiment_yaml().replace("version: 1", "version: 2"),
            "Unsupported file version",
            "This file uses version `2`",
        ),
        (
            _experiment_yaml().replace("provider: openai", "provider: other"),
            "Some fields are invalid",
            "- **model → provider** must be one of 'openai', 'anthropic' or 'typesafe'",
        ),
        (
            _experiment_yaml().replace("provider: openai", "provider: typesafe"),
            "Some fields are invalid",
            "Prompt type does not match provider 'typesafe'.",
        ),
    ],
)
def test_experiment_import_errors(content: str, title: str, fragment: str) -> None:
    report = errors.describe_import_error(_raised(content, ex.from_yaml), "experiment")

    assert report.title == title
    assert fragment in report.message


def test_non_utf8_files_are_reported() -> None:
    exc = _raised("", lambda _: b"\xff".decode("utf-8"))

    assert errors.describe_import_error(exc, "experiment").title == "Can't read the file"


@pytest.mark.parametrize(
    ("content", "title", "fragment"),
    [
        ('{"a": "x"', "Not valid YAML or JSON", "syntax error at line 1"),
        ("42", "Unrecognized chunk format", "intro: The first chunk's text."),
        ("- [nested]", "Unrecognized chunk format", "list of strings"),
        (
            "- id: a\n  text: x\n- id: a\n  text: y",
            "Some chunks are invalid",
            "Duplicate chunk ID 'a'",
        ),
    ],
)
def test_chunk_import_errors(content: str, title: str, fragment: str) -> None:
    report = errors.describe_import_error(_raised(content, ex.parse_chunk_file), "chunks")

    assert report.title == title
    assert fragment in report.message


def test_build_classifier_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ex.MissingAPIKeyError, match="OPENAI_API_KEY"):
        ex.build_classifier(LOCAL, api_key=None)

    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio")
    pytest.importorskip("langchain_openai")
    assert type(ex.build_classifier(LOCAL, api_key=None)).__name__ == (
        "ChatModelChunkClassifier"
    )


def test_evaluate_reports_unreadable_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unreadable:
        async def classify(self, **_: object) -> None:
            raise ValueError("Invalid JSON output")

    monkeypatch.setattr(ex, "build_classifier", lambda config, key: Unreadable())
    experiment = ex.from_yaml(_experiment_yaml())

    with pytest.raises(ModelResponseError) as raised:
        asyncio.run(ex.evaluate(experiment, api_key="key"))
    assert isinstance(raised.value.__cause__, ValueError)


def test_dashboard_shows_a_friendly_error_with_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    class Offline:
        async def classify(self, **_: object) -> None:
            raise ModelConnectionError("Connection error.")

    monkeypatch.setattr(ex, "build_classifier", lambda config, key: Offline())
    app = AppTest.from_file(str(Path(ex.__file__).with_name("app.py")), default_timeout=30)
    app.run()
    app.text_area(key="chunk_text_0").input("Evidence")
    app.text_input(key="model").input("qwen/qwen3.5-9b")
    app.text_input(key="base_url").input("http://localhost:1234/v1")
    app.button(key="evaluate").click().run()

    assert not app.exception
    assert app.error[0].value.startswith("**Can't reach the model server**")
    assert "`http://localhost:1234/v1`" in app.error[0].value
    assert app.expander[-1].label == "Technical details"
    assert "ModelConnectionError: Connection error." in app.code[-1].value


def test_dashboard_reports_a_broken_experiment_file() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(ex.__file__).with_name("app.py")), default_timeout=30)
    app.run()
    app.session_state["import_error"] = errors.describe_import_error(
        _raised("model: [unclosed", yaml.safe_load), "experiment"
    )
    app.run()

    assert app.error[0].value.startswith("**Not a valid YAML file**")
