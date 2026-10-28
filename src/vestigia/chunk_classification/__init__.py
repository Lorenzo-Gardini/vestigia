"""Models and types for classifying evidence chunks.

Examples
--------
Create a chunk and represent a chat-model classification:

>>> from vestigia.chunk_classification import (
...     ChatModelChunkClassification,
...     ChatModelChunksClassificationResult,
...     Chunk,
...     ChunkStatus,
... )
>>> chunk = Chunk(id="source-1", text="Evidence")
>>> chat_result = ChatModelChunksClassificationResult(
...     classifications=(
...         ChatModelChunkClassification(
...             chunk=chunk,
...             status=ChunkStatus.USED,
...             explanation="The evidence supports the answer.",
...         ),
...     )
... )
>>> chat_result.classifications[0].status is ChunkStatus.USED
True

Define a system-one prompt and represent its classification result:

>>> from vestigia.chunk_classification import (
...     SystemOneModelChunkClassification,
...     SystemOneModelChunksClassificationResult,
...     SystemOneModelTemplate,
... )
>>> prompt = SystemOneModelTemplate(
...     state="Question: {query} Answer: {answer}",
...     instruction="Assess this evidence: {chunk}",
...     criteria={
...         ChunkStatus.USED: "Supports the answer",
...         ChunkStatus.NOT_USED: "Does not support the answer",
...     },
... )
>>> system_result = SystemOneModelChunksClassificationResult(
...     classifications=(
...         SystemOneModelChunkClassification(
...             chunk=chunk,
...             status=ChunkStatus.USED,
...             probabilities={ChunkStatus.USED: 0.95, ChunkStatus.NOT_USED: 0.05},
...             confidence=0.95,
...         ),
...     )
... )
>>> system_result.classifications[0].confidence
0.95
"""

from .models import (
    ChatModelChunkClassification,
    ChatModelChunksClassificationResult,
    Chunk,
    ChunkStatus,
    EvidenceModel,
    SystemOneModelChunkClassification,
    SystemOneModelChunksClassificationResult,
    SystemOneModelTemplate,
)

__all__ = [
    "ChatModelChunkClassification",
    "ChatModelChunksClassificationResult",
    "Chunk",
    "ChunkStatus",
    "EvidenceModel",
    "SystemOneModelChunkClassification",
    "SystemOneModelChunksClassificationResult",
    "SystemOneModelTemplate",
]
