"""Regression tests using real TypeSafe request and response types."""

import asyncio
from importlib.util import find_spec
from unittest.mock import AsyncMock, Mock

import pytest

if find_spec("langchain_typesafe") is None:
    pytest.skip("TypeSafe is not installed", allow_module_level=True)

from langchain_typesafe import Choice, ClassifierResponse, TypeSafeClassifier

from vestigia.chunk_classification import Chunk, ChunkStatus, SystemOneModelTemplate
from vestigia.chunk_classification.classifier import ModelResponseError
from vestigia.chunk_classification.langchain import SystemOneModelChunkClassifier


@pytest.fixture
def chunks() -> list[Chunk]:
    """Provide input chunks independently of backend response order."""
    return [Chunk(id="first", text="First evidence"), Chunk(id="second", text="Second")]


@pytest.fixture
def model() -> Mock:
    """Return a real response with reversed answer order."""
    response = ClassifierResponse.model_validate(
        {
            "model": "test-model",
            "answers": {
                "second": {
                    "type": "choice",
                    "choice": "not_used",
                    "probabilities": {"not_used": 0.9, "used": 0.1},
                    "confidence": 0.9,
                },
                "first": {
                    "type": "choice",
                    "choice": "used",
                    "probabilities": {"used": 0.8, "not_used": 0.2},
                    "confidence": 0.8,
                },
            },
        }
    )
    model = Mock(spec=TypeSafeClassifier)
    model.ainvoke = AsyncMock(return_value=response)
    return model


def test_classifications_match_chunks_by_id(model: Mock, chunks: list[Chunk]) -> None:
    """Associate each response with its ID, retaining input order and scores."""
    result = asyncio.run(
        SystemOneModelChunkClassifier(model).classify(query="Q", answer="A", chunks=chunks)
    )
    assert [item.chunk for item in result.classifications] == chunks
    assert [item.status for item in result.classifications] == [
        ChunkStatus.USED,
        ChunkStatus.NOT_USED,
    ]
    assert result.classifications[0].confidence == 0.8
    assert result.classifications[0].probabilities == {
        ChunkStatus.USED: 0.8,
        ChunkStatus.NOT_USED: 0.2,
    }


def test_custom_prompt_formats_real_choices(model: Mock, chunks: list[Chunk]) -> None:
    """Send formatted context and instructions with the supplied criteria."""
    template = SystemOneModelTemplate(
        state="Question {query}; answer {answer}",
        instruction="Check {chunk}",
        criteria={ChunkStatus.USED: "Used", ChunkStatus.NOT_USED: "Not used"},
    )
    asyncio.run(
        SystemOneModelChunkClassifier(model).classify(
            query="Q", answer="A", chunks=chunks, prompt=template
        )
    )
    request = model.ainvoke.call_args.args[0]
    assert request["state"] == "Question Q; answer A"
    assert list(request["questions"]) == ["first", "second"]
    question = request["questions"]["first"]
    assert isinstance(question, Choice)
    assert question.instructions == "Check First evidence"
    assert question.criteria == template.criteria


def test_empty_chunks_do_not_call_model(model: Mock) -> None:
    """Return no classifications without invoking the backend."""
    result = asyncio.run(
        SystemOneModelChunkClassifier(model).classify(query="Q", answer="A", chunks=[])
    )
    assert result.classifications == ()
    model.ainvoke.assert_not_awaited()


def test_duplicate_ids_fail_before_model_call(model: Mock, chunks: list[Chunk]) -> None:
    """Reject duplicate chunk IDs instead of overwriting request questions."""
    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        asyncio.run(
            SystemOneModelChunkClassifier(model).classify(
                query="Q", answer="A", chunks=[chunks[0], chunks[0]]
            )
        )
    model.ainvoke.assert_not_awaited()


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("missing", "missing: 'first'"),
        ("extra", "extra: 'unknown'"),
        ("wrong_type", "Expected a choice answer"),
        ("unknown_status", "outside the criteria"),
    ],
)
def test_invalid_system_one_answers_fail(
    model: Mock, chunks: list[Chunk], invalid: str, message: str
) -> None:
    """Reject incomplete, unrelated, or incompatible model answers explicitly."""
    response = model.ainvoke.return_value.model_dump()
    if invalid == "missing":
        del response["answers"]["first"]
    elif invalid == "extra":
        response["answers"]["unknown"] = response["answers"]["first"]
    elif invalid == "wrong_type":
        response["answers"]["first"] = {"type": "noul", "noul": 0.8}
    else:
        response["answers"]["first"]["choice"] = "unknown"
    model.ainvoke.return_value = ClassifierResponse.model_validate(response)
    with pytest.raises(ModelResponseError, match=message):
        asyncio.run(
            SystemOneModelChunkClassifier(model).classify(
                query="Q", answer="A", chunks=chunks
            )
        )


def test_backend_errors_propagate(model: Mock, chunks: list[Chunk]) -> None:
    """Preserve provider errors without partial classifications."""
    model.ainvoke.side_effect = RuntimeError("Provider unavailable")
    with pytest.raises(RuntimeError, match="Provider unavailable"):
        asyncio.run(
            SystemOneModelChunkClassifier(model).classify(
                query="Q", answer="A", chunks=chunks
            )
        )


@pytest.mark.parametrize("invalid", ["choice", "probabilities"])
def test_answer_labels_must_belong_to_custom_criteria(
    model: Mock, chunks: list[Chunk], invalid: str
) -> None:
    """Reject valid enum statuses that were not offered in the custom prompt."""
    response = model.ainvoke.return_value.model_dump()
    if invalid == "choice":
        response["answers"]["first"]["choice"] = "contradicted"
    else:
        response["answers"]["first"]["probabilities"]["contradicted"] = 0.1
    model.ainvoke.return_value = ClassifierResponse.model_validate(response)
    template = SystemOneModelTemplate(
        state="{query} {answer}",
        instruction="{chunk}",
        criteria={ChunkStatus.USED: "Used", ChunkStatus.NOT_USED: "Not used"},
    )
    with pytest.raises(ValueError, match="outside the criteria"):
        asyncio.run(
            SystemOneModelChunkClassifier(model).classify(
                query="Q", answer="A", chunks=chunks, prompt=template
            )
        )
