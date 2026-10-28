"""Data models shared by Vestigia's chunk classification implementations.

Examples
--------
>>> chunk = Chunk(id="source-1", text="The plan lasts 60 months.")
>>> classification = ChatModelChunkClassification(
...     chunk=chunk,
...     status=ChunkStatus.USED,
...     explanation="The evidence states the plan duration.",
... )
>>> classification.status
<ChunkStatus.USED: 'used'>
"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ChunkStatus(StrEnum):
    """Observable relationships between a source chunk and a generated answer.

    Examples
    --------
    >>> ChunkStatus("used") is ChunkStatus.USED
    True
    """

    USED = "used"
    PARTIALLY_USED = "partially_used"
    CONTRADICTED = "contradicted"
    NOT_USED = "not_used"
    UNCERTAIN = "uncertain"


class EvidenceModel(BaseModel):
    """Base model with frozen attributes and no undeclared fields.

    Freezing prevents attribute reassignment, not mutation of nested dictionaries.
    Pydantic's normal type coercion still applies.

    Examples
    --------
    >>> chunk = Chunk(id="source-1", text="Evidence")
    >>> chunk.model_config["frozen"]
    True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class Chunk(EvidenceModel):
    """A source text segment identified by a stable ID.

    Examples
    --------
    >>> chunk = Chunk(id="source-1", text="The plan lasts 60 months.")
    >>> chunk.id
    'source-1'
    """

    id: str = Field(
        ...,
        description="Stable identifier used to associate results with this chunk.",
    )
    text: str = Field(
        ...,
        description="Source text that the classifier evaluates.",
    )


################## CHAT MODEL MODELS ##################


class ChatModelChunkClassification(EvidenceModel):
    """A chat model's classification and explanation for one chunk.

    Examples
    --------
    >>> item = ChatModelChunkClassification(
    ...     chunk=Chunk(id="source-1", text="Evidence"),
    ...     status=ChunkStatus.USED,
    ...     explanation="The evidence supports the answer.",
    ... )
    >>> item.status
    <ChunkStatus.USED: 'used'>
    """

    chunk: Chunk = Field(
        ...,
        description="Source chunk associated with this classification.",
    )
    status: ChunkStatus = Field(
        ...,
        description="Evidence attribution status assigned to the chunk.",
    )
    explanation: str | None = Field(
        default=None,
        description="Optional explanation for the assigned status.",
    )


class ChatModelChunksClassificationResult(EvidenceModel):
    """Classifications returned by a chat model.

    Examples
    --------
    >>> item = ChatModelChunkClassification(
    ...     chunk=Chunk(id="source-1", text="Evidence"),
    ...     status=ChunkStatus.USED,
    ... )
    >>> result = ChatModelChunksClassificationResult(classifications=(item,))
    >>> len(result.classifications)
    1
    """

    classifications: tuple[ChatModelChunkClassification, ...] = Field(
        ...,
        description="Per-chunk classifications produced by the chat model.",
    )


################## SYSTEM ONE MODEL MODELS ##################


class SystemOneModelChunkClassification(EvidenceModel):
    """A classification with per-status probabilities and confidence.

    Examples
    --------
    >>> item = SystemOneModelChunkClassification(
    ...     chunk=Chunk(id="source-1", text="Evidence"),
    ...     status=ChunkStatus.USED,
    ...     probabilities={ChunkStatus.USED: 0.9, ChunkStatus.NOT_USED: 0.1},
    ...     confidence=0.9,
    ... )
    >>> item.confidence
    0.9
    """

    chunk: Chunk = Field(
        ...,
        description="Source chunk associated with this classification.",
    )
    status: ChunkStatus = Field(
        ...,
        description="Most likely evidence attribution status for the chunk.",
    )
    probabilities: dict[
        ChunkStatus, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    ] = Field(
        ...,
        description="Finite probabilities in [0, 1]; values are not normalized.",
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        allow_inf_nan=False,
        description="Finite model confidence in the inclusive interval [0, 1].",
    )


class SystemOneModelChunksClassificationResult(EvidenceModel):
    """Classifications returned by a system-one model.

    Examples
    --------
    >>> item = SystemOneModelChunkClassification(
    ...     chunk=Chunk(id="source-1", text="Evidence"),
    ...     status=ChunkStatus.USED,
    ...     probabilities={ChunkStatus.USED: 0.9, ChunkStatus.NOT_USED: 0.1},
    ...     confidence=0.9,
    ... )
    >>> result = SystemOneModelChunksClassificationResult(classifications=(item,))
    >>> result.classifications[0].status
    <ChunkStatus.USED: 'used'>
    """

    classifications: tuple[SystemOneModelChunkClassification, ...] = Field(
        ...,
        description="Per-chunk classifications produced by the system-one model.",
    )


class SystemOneModelTemplate(EvidenceModel):
    """Prompt state, instruction, and criteria for system-one classification.

    Examples
    --------
    >>> prompt = SystemOneModelTemplate(
    ...     state="Question: {query} Answer: {answer}",
    ...     instruction="Assess: {chunk}",
    ...     criteria={
    ...         ChunkStatus.USED: "Supports the answer",
    ...         ChunkStatus.NOT_USED: "Does not support the answer",
    ...     },
    ... )
    >>> len(prompt.criteria)
    2
    """

    state: str = Field(
        ...,
        description="Template for the overall query-and-answer context.",
    )
    instruction: str = Field(
        ...,
        description="Template for the instruction applied to each chunk.",
    )
    criteria: dict[ChunkStatus, str] = Field(
        ...,
        min_length=2,
        description="Text criterion for each status considered by the model.",
    )
