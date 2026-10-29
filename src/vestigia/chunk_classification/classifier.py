"""Protocol for implementations that classify evidence chunks."""

from collections.abc import Iterable, Sequence
from typing import Protocol

from vestigia.chunk_classification import Chunk

__all__ = ["ChunkClassifier", "ModelResponseError"]


class ModelResponseError(ValueError):
    """A backend response that does not correspond to the requested chunks.

    ``missing``, ``extra``, and ``duplicates`` list the offending chunk IDs
    when the mismatch is about IDs; they are empty otherwise.

    Examples
    --------
    >>> error = ModelResponseError("Answer IDs must match", missing=["a"], extra=["z"])
    >>> error.missing, error.extra
    (('a',), ('z',))
    >>> str(error)
    "Answer IDs must match (missing: 'a'; extra: 'z')"
    """

    def __init__(
        self,
        message: str,
        *,
        missing: Iterable[str] = (),
        extra: Iterable[str] = (),
        duplicates: Iterable[str] = (),
    ) -> None:
        self.missing = tuple(missing)
        self.extra = tuple(extra)
        self.duplicates = tuple(duplicates)
        parts = [
            f"{label}: {', '.join(map(repr, ids))}"
            for label, ids in (
                ("missing", self.missing),
                ("extra", self.extra),
                ("duplicate", self.duplicates),
            )
            if ids
        ]
        super().__init__(f"{message} ({'; '.join(parts)})" if parts else message)


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
            If input chunk IDs are duplicated.
        ModelResponseError
            If the backend response does not correspond exactly to the
            requested chunks. It is a ``ValueError`` subclass.

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


def _check_response_ids(
    message: str, expected: Iterable[str], returned: Sequence[str]
) -> None:
    """Reject response IDs that are missing, extra, or repeated.

    Examples
    --------
    >>> _check_response_ids("IDs must match", ["a", "b"], ["b", "c", "c"])
    Traceback (most recent call last):
    ...
    vestigia.chunk_classification.classifier.ModelResponseError: IDs must match (missing: 'a'; extra: 'c'; duplicate: 'c')
    """
    expected = list(expected)
    duplicates = list(dict.fromkeys(i for i in returned if returned.count(i) > 1))
    missing = [i for i in expected if i not in returned]
    extra = list(dict.fromkeys(i for i in returned if i not in expected))
    if duplicates or missing or extra:
        raise ModelResponseError(
            message, missing=missing, extra=extra, duplicates=duplicates
        )
