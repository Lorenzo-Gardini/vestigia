"""Execute Python documentation examples without contacting model providers."""

import doctest
import importlib
from importlib.util import find_spec
from pathlib import Path
import re
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize(
    ("module_name", "dependencies"),
    [
        ("vestigia", ()),
        ("vestigia.chunk_classification", ()),
        ("vestigia.chunk_classification.models", ()),
        ("vestigia.chunk_classification.classifier", ()),
        ("vestigia.chunk_classification.prompts", ("yaml",)),
        (
            "vestigia.chunk_classification.langchain",
            ("langchain_core", "langchain_typesafe"),
        ),
        (
            "vestigia.chunk_classification.langchain.chat_model_classifier",
            ("langchain_core",),
        ),
        (
            "vestigia.chunk_classification.langchain.system_one_classifier",
            ("langchain_typesafe",),
        ),
        (
            "vestigia.chunk_classification.dashboard.experiment",
            ("yaml", "langchain_openai"),
        ),
    ],
)
def test_docstring_examples(module_name: str, dependencies: tuple[str, ...]) -> None:
    """Run all executable examples when their optional dependencies are installed."""
    for dependency in dependencies:
        if find_spec(dependency) is None:
            pytest.skip(f"{dependency} is not installed")
    result = doctest.testmod(importlib.import_module(module_name))
    assert result.failed == 0


README = Path(__file__).resolve().parents[1] / "README.md"
PYTHON_EXAMPLES = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)


@pytest.mark.parametrize("source", PYTHON_EXAMPLES)
def test_readme_python_examples(source: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute README snippets with provider constructors and async calls mocked."""
    from vestigia.chunk_classification import (
        ChatModelChunksClassificationResult,
        SystemOneModelChunksClassificationResult,
    )

    classify = None
    if "ChatOpenAI" in source:
        if find_spec("langchain_openai") is None:
            pytest.skip("The OpenAI integration is not installed")
        provider = importlib.import_module("langchain_openai")
        from vestigia.chunk_classification.langchain import ChatModelChunkClassifier

        monkeypatch.setattr(provider, "ChatOpenAI", Mock())
        classify = AsyncMock(
            return_value=ChatModelChunksClassificationResult(classifications=())
        )
        monkeypatch.setattr(ChatModelChunkClassifier, "classify", classify)
    if "TypeSafeClassifier" in source:
        if find_spec("langchain_typesafe") is None:
            pytest.skip("TypeSafe is not installed")
        provider = importlib.import_module("langchain_typesafe")
        from vestigia.chunk_classification.langchain import SystemOneModelChunkClassifier

        monkeypatch.setattr(provider, "TypeSafeClassifier", Mock())
        classify = AsyncMock(
            return_value=SystemOneModelChunksClassificationResult(classifications=())
        )
        monkeypatch.setattr(SystemOneModelChunkClassifier, "classify", classify)
    if "load_system_one_prompt" in source and find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed")

    exec(compile(source, str(README), "exec"), {"__name__": "__main__"})
    if classify is not None:
        classify.assert_awaited_once()
