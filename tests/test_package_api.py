"""Tests for the public package imports and core model validation."""

import importlib
from importlib.metadata import metadata, version
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys

import pytest
from pydantic import ValidationError

import vestigia.chunk_classification as chunk_classification
import vestigia.chunk_classification.classifier as classifier
from vestigia.chunk_classification import (
    ChatModelChunkClassification,
    ChatModelChunksClassificationResult,
    Chunk,
    EvidenceModel,
    SystemOneModelChunkClassification,
    SystemOneModelChunksClassificationResult,
    SystemOneModelTemplate,
)
from vestigia.chunk_classification.classifier import ChunkClassifier
from vestigia.chunk_classification.prompts import load_chat_model_prompt


def test_public_package_exports():
    """Expose the domain models and classifier protocol through package initializers."""
    assert Chunk(id="chunk-1", text="Evidence").text == "Evidence"
    assert chunk_classification.__all__ == [
        "ChatModelChunkClassification",
        "ChatModelChunksClassificationResult",
        "Chunk",
        "ChunkStatus",
        "EvidenceModel",
        "SystemOneModelChunkClassification",
        "SystemOneModelChunksClassificationResult",
        "SystemOneModelTemplate",
    ]
    assert classifier.__all__ == ["ChunkClassifier", "ModelResponseError"]
    assert issubclass(classifier.ModelResponseError, ValueError)
    for name in chunk_classification.__all__:
        assert getattr(chunk_classification, name).__name__ == name
    assert classifier.ChunkClassifier is ChunkClassifier


@pytest.mark.parametrize(
    "model",
    [
        Chunk,
        ChatModelChunkClassification,
        ChatModelChunksClassificationResult,
        SystemOneModelChunkClassification,
        SystemOneModelChunksClassificationResult,
        SystemOneModelTemplate,
    ],
)
def test_pydantic_model_fields_have_descriptions(model: type[EvidenceModel]):
    """Require a description for every exported Pydantic model field."""
    assert all(field.description for field in model.model_fields.values())


def test_evidence_models_freeze_attributes_and_reject_extra_fields():
    """Keep frozen attributes and undeclared-field rejection without claiming deep freezing."""
    chunk = Chunk(id="chunk-1", text="Evidence")

    with pytest.raises(ValidationError, match="frozen_instance"):
        setattr(chunk, "text", "Changed")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Chunk.model_validate({"id": "chunk-1", "text": "Evidence", "extra": "unexpected"})


def test_chat_model_prompt_is_loaded_from_package_resources():
    """Load the default chat-model prompt from the installed package."""
    assert load_chat_model_prompt().strip()


def test_system_one_prompt_is_loaded_and_validated():
    """Load the default system-one prompt from the packaged YAML resource."""
    if find_spec("yaml") is None:
        pytest.skip("PyYAML is not installed")

    from vestigia.chunk_classification.prompts import load_system_one_prompt

    prompt = load_system_one_prompt()

    assert prompt.state
    assert prompt.instruction
    assert len(prompt.criteria) >= 2


@pytest.mark.parametrize("allow_chat", [False, True])
def test_optional_imports_are_isolated(allow_chat: bool) -> None:
    """Block optional imports in a fresh interpreter, not an already-loaded module."""
    if allow_chat and find_spec("langchain_core") is None:
        pytest.skip("LangChain is not installed")
    script = f"""
import importlib.abc
import sys

class BlockBackendImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{"langchain_typesafe", "langchain_core"}} - (
            {{"langchain_core"}} if {allow_chat!r} else set()
        ):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockBackendImports())
from vestigia.chunk_classification import Chunk
import vestigia.chunk_classification.langchain as adapters
assert adapters.__all__ == ["ChatModelChunkClassifier", "SystemOneModelChunkClassifier"]
assert Chunk(id="one", text="Evidence").id == "one"
if {allow_chat!r}:
    from vestigia.chunk_classification.langchain import ChatModelChunkClassifier
    assert ChatModelChunkClassifier.__name__ == "ChatModelChunkClassifier"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_unknown_adapter_attribute_raises() -> None:
    """Preserve normal Python behavior for attributes absent from lazy exports."""
    adapters = importlib.import_module("vestigia.chunk_classification.langchain")
    with pytest.raises(AttributeError, match="UnknownClassifier"):
        getattr(adapters, "UnknownClassifier")


def test_packaged_prompts_have_expected_template_variables() -> None:
    """Validate prompt placeholders without requiring LangChain or YAML."""
    from string import Formatter

    fields = {
        name
        for _, name, _, _ in Formatter().parse(load_chat_model_prompt())
        if name is not None
    }
    assert fields == {"query", "answer", "chunks"}


def test_distribution_metadata_matches_public_package() -> None:
    """Keep installed distribution metadata aligned with the Vestigia project."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    installed = metadata("vestigia")

    assert project["name"] == installed["Name"] == "vestigia"
    assert version("vestigia") == project["version"]
    assert installed["Requires-Python"] == project["requires-python"]
    assert installed["License-Expression"] == project["license"]
    assert installed["Description-Content-Type"] == "text/markdown"
    assert set(installed.get_all("Provides-Extra")) == set(project["optional-dependencies"])
