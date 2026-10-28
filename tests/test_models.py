"""Tests for required fields, defaults, and numeric model boundaries."""

import pytest
from pydantic import ValidationError

from vestigia.chunk_classification import (
    ChatModelChunkClassification,
    Chunk,
    ChunkStatus,
    SystemOneModelChunkClassification,
    SystemOneModelTemplate,
)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["probabilities", "confidence"])
def test_invalid_numeric_scores_are_rejected(value: float, field: str) -> None:
    """Reject out-of-range and non-finite probability and confidence values."""
    data = {
        "chunk": Chunk(id="one", text="Evidence"),
        "status": ChunkStatus.USED,
        "probabilities": {ChunkStatus.USED: 0.9, ChunkStatus.NOT_USED: 0.1},
        "confidence": 0.9,
    }
    data[field] = {ChunkStatus.USED: value} if field == "probabilities" else value
    with pytest.raises(ValidationError):
        SystemOneModelChunkClassification.model_validate(data)


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_numeric_score_boundaries_are_valid(value: float) -> None:
    """Accept inclusive boundaries without normalizing the distribution."""
    result = SystemOneModelChunkClassification(
        chunk=Chunk(id="one", text="Evidence"),
        status=ChunkStatus.USED,
        probabilities={ChunkStatus.USED: value},
        confidence=value,
    )
    assert result.confidence == value
    assert result.probabilities[ChunkStatus.USED] == value


def test_nullable_explanation_keeps_its_default() -> None:
    """Keep explanation optional while source chunk fields remain required."""
    chunk = Chunk(id="one", text="Evidence")
    item = ChatModelChunkClassification(chunk=chunk, status=ChunkStatus.USED)
    assert item.explanation is None
    assert all(field.is_required() for field in Chunk.model_fields.values())
    with pytest.raises(ValidationError, match="Field required"):
        Chunk.model_validate({"id": "one"})


def test_template_requires_at_least_two_criteria() -> None:
    """Preserve the minimum number of alternatives required by the template."""
    with pytest.raises(ValidationError):
        SystemOneModelTemplate(
            state="{query} {answer}",
            instruction="{chunk}",
            criteria={ChunkStatus.USED: "Used"},
        )
