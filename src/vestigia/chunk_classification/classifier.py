"""Protocol for implementations that classify evidence chunks."""

from collections.abc import Sequence
from typing import Protocol

from vestigia.chunk_classification import Chunk

__all__ = ["ChunkClassifier"]


class ChunkClassifier[PromptType, OutputType](Protocol):
    """Asynchronous interface for classifying a sequence of chunks.

    Examples
    --------
    An implementation can be used with any sequence of chunks:

        chunks = [Chunk(id="source-1", text="Evidence")]
        result = await classifier.classify(
            query="What supports the answer?",
            answer="The evidence supports it.",
            chunks=chunks,
        )
    """

    async def classify(
        self,
        *,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
        prompt: PromptType | None = None,
    ) -> OutputType:
        """Classify chunks using the query, answer, and optional prompt.

        Chunk IDs must be unique within a call. Empty input returns an empty
        result without invoking a backend. Results retain input order.

        Raises
        ------
        ValueError
            If IDs are duplicated or the backend response does not correspond
            exactly to the requested chunks.

        Examples
        --------
        Within an async function:

            result = await classifier.classify(
                query="What supports the answer?",
                answer="The evidence supports it.",
                chunks=[Chunk(id="source-1", text="Evidence")],
            )
        """
        ...


def _index_chunks(chunks: Sequence[Chunk]) -> dict[str, Chunk]:
    """Index chunks in input order, rejecting duplicate identifiers.

    Examples
    --------
    >>> list(_index_chunks([Chunk(id="one", text="Evidence")]))
    ['one']
    """
    indexed: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.id in indexed:
            raise ValueError(f"Duplicate chunk ID: {chunk.id!r}")
        indexed[chunk.id] = chunk
    return indexed
