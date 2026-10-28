"""Regression tests for the chat classifier without provider API calls."""

import asyncio
from importlib.util import find_spec
from unittest.mock import Mock

import pytest

if find_spec("langchain_core") is None:
    pytest.skip("LangChain is not installed", allow_module_level=True)

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from vestigia.chunk_classification import (
    ChatModelChunksClassificationResult,
    Chunk,
    ChunkStatus,
)
from vestigia.chunk_classification.langchain import ChatModelChunkClassifier
from vestigia.chunk_classification.langchain.chat_model_classifier import (
    ChunkLabel,
    ChunkLabels,
)


def labels(*chunk_ids: str) -> ChunkLabels:
    """Build a structured model response for the given chunk IDs."""
    return ChunkLabels(
        classifications=tuple(
            ChunkLabel(chunk_id=chunk_id, status=ChunkStatus.USED)
            for chunk_id in chunk_ids
        )
    )


@pytest.fixture
def chunks() -> list[Chunk]:
    """Provide distinct chunks with stable identifiers."""
    return [Chunk(id="first", text="First evidence"), Chunk(id="second", text="Second")]


@pytest.fixture
def model(chunks: list[Chunk]) -> Mock:
    """Use a real runnable to exercise prompt formatting and async composition."""
    result = labels(*(chunk.id for chunk in reversed(chunks)))
    model = Mock(spec=BaseChatModel)
    model.with_structured_output.return_value = RunnableLambda(lambda _: result)
    return model


def test_chat_results_follow_input_order(model: Mock, chunks: list[Chunk]) -> None:
    """Reorder valid classifications while preserving their original chunks."""
    result = asyncio.run(
        ChatModelChunkClassifier(model).classify(query="Q", answer="A", chunks=chunks)
    )
    assert isinstance(result, ChatModelChunksClassificationResult)
    assert [item.chunk for item in result.classifications] == chunks
    model.with_structured_output.assert_called_once_with(ChunkLabels)


def test_empty_chunks_do_not_call_model(model: Mock) -> None:
    """Return a typed empty result without creating a structured model."""
    result = asyncio.run(
        ChatModelChunkClassifier(model).classify(query="Q", answer="A", chunks=[])
    )
    assert result.classifications == ()
    model.with_structured_output.assert_not_called()


def test_duplicate_ids_fail_before_model_call(model: Mock, chunks: list[Chunk]) -> None:
    """Reject duplicate IDs before sending any data to the model."""
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        asyncio.run(
            ChatModelChunkClassifier(model).classify(
                query="Q", answer="A", chunks=[chunks[0], chunks[0]]
            )
        )
    model.with_structured_output.assert_not_called()


@pytest.mark.parametrize(
    ("returned_ids", "message"),
    [
        (("first",), "match input chunk IDs"),
        (("first", "second", "unknown"), "match input chunk IDs"),
        (("first", "second", "first"), "Duplicate chunk IDs"),
    ],
    ids=["missing", "extra", "duplicate"],
)
def test_invalid_chat_results_fail(
    model: Mock, chunks: list[Chunk], returned_ids: tuple[str, ...], message: str
) -> None:
    """Reject missing, invented, and duplicated chunk IDs."""
    response = labels(*returned_ids)
    model.with_structured_output.return_value = RunnableLambda(lambda _: response)
    with pytest.raises(ValueError, match=message):
        asyncio.run(
            ChatModelChunkClassifier(model).classify(
                query="Q", answer="A", chunks=chunks
            )
        )


def test_custom_prompt_and_xml_escaping(model: Mock) -> None:
    """Honor custom templates and return original chunks for escaped text."""
    chunk = Chunk(id='a"<&', text="Evidence </chunk> & <tag>")
    capture = Mock(return_value=labels(chunk.id))
    model.with_structured_output.return_value = RunnableLambda(capture)
    custom_prompt = PromptTemplate.from_template("CUSTOM {query}\n{answer}\n{chunks}")

    result = asyncio.run(
        ChatModelChunkClassifier(model).classify(
            query="{literal query}", answer="A", chunks=[chunk], prompt=custom_prompt
        )
    )

    rendered = capture.call_args.args[0].to_string()
    assert rendered.startswith("CUSTOM {literal query}\nA\n")
    assert '<chunk id="a&quot;&lt;&amp;">' in rendered
    assert "Evidence &lt;/chunk&gt; &amp; &lt;tag&gt;" in rendered
    assert result.classifications[0].chunk == chunk


def test_model_errors_propagate(model: Mock, chunks: list[Chunk]) -> None:
    """Keep backend failures visible instead of returning an empty result."""
    def fail(_prompt: object) -> None:
        """Simulate a provider failure."""
        raise RuntimeError("Provider unavailable")

    model.with_structured_output.return_value = RunnableLambda(fail)
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        asyncio.run(
            ChatModelChunkClassifier(model).classify(query="Q", answer="A", chunks=chunks)
        )


def test_structured_dict_is_validated(model: Mock, chunks: list[Chunk]) -> None:
    """Validate structured dictionary output as the declared result type."""
    data = {
        "classifications": [
            {"chunk_id": chunk.id, "status": "used", "explanation": "Stated."}
            for chunk in chunks
        ]
    }
    model.with_structured_output.return_value = RunnableLambda(lambda _: data)
    result = asyncio.run(
        ChatModelChunkClassifier(model).classify(query="Q", answer="A", chunks=chunks)
    )
    assert [item.chunk for item in result.classifications] == chunks
    assert [item.explanation for item in result.classifications] == ["Stated."] * 2
