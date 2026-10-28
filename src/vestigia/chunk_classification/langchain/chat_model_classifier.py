"""Chat-model implementation of the chunk classifier protocol.

Example:

    classifier = ChatModelChunkClassifier(configured_chat_model)
    result = await classifier.classify(
        query="What supports the answer?",
        answer="The evidence supports it.",
        chunks=[Chunk(id="source-1", text="Evidence")],
    )

"""

from collections.abc import Sequence
from html import escape

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import Field

from vestigia.chunk_classification import (
    ChatModelChunkClassification,
    ChatModelChunksClassificationResult,
    Chunk,
    ChunkStatus,
    EvidenceModel,
)
from vestigia.chunk_classification.classifier import ChunkClassifier, _index_chunks
from vestigia.chunk_classification.prompts import load_chat_model_prompt


class ChunkLabel(EvidenceModel):
    """Structured-output label for one chunk, identified only by its ID."""

    chunk_id: str = Field(..., description="Identifier of the classified input chunk.")
    status: ChunkStatus = Field(
        ...,
        description="Evidence attribution status assigned to the chunk.",
    )
    explanation: str | None = Field(
        default=None,
        description="Optional explanation for the assigned status.",
    )


class ChunkLabels(EvidenceModel):
    """Structured-output schema requested from the chat model."""

    classifications: tuple[ChunkLabel, ...] = Field(
        ...,
        description="Exactly one label per input chunk.",
    )


class ChatModelChunkClassifier(
    ChunkClassifier[PromptTemplate, ChatModelChunksClassificationResult]
):
    """Classify chunks with a LangChain chat model and structured output.

    Examples
    --------
    Given a configured LangChain chat model:

        classifier = ChatModelChunkClassifier(model)
        result = await classifier.classify(
            query="What supports the answer?",
            answer="The evidence supports it.",
            chunks=[Chunk(id="source-1", text="Evidence")],
        )
    """

    def __init__(self, model: BaseChatModel) -> None:
        """Initialize the classifier with a LangChain chat model.

        Examples
        --------
        Given an application-configured model:

            classifier = ChatModelChunkClassifier(model)
        """
        self._model = model

    async def classify(
        self,
        *,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
        prompt: PromptTemplate | None = None,
    ) -> ChatModelChunksClassificationResult:
        """Classify each chunk, using the supplied or default prompt.

        The prompt receives ``query``, ``answer``, and XML-escaped ``chunks``.
        The model returns only chunk IDs with labels; results reference the
        original ``Chunk`` objects. An empty sequence returns an empty result
        without calling the model. Classifications are returned in input order.

        Parameters
        ----------
        query
            Original user question, used as context for attribution.
        answer
            Generated answer to compare with the evidence.
        chunks
            Source chunks with unique IDs within this call.
        prompt
            Optional LangChain template; the packaged prompt is used when omitted.

        Returns
        -------
        ChatModelChunksClassificationResult
            Validated classifications corresponding exactly to the source chunks.

        Raises
        ------
        ValueError
            If input IDs are duplicated or result IDs are missing, extra, or
            duplicated.

        Backend and Pydantic validation errors propagate to the caller.

        Examples
        --------
        Within an async function:

            result = await classifier.classify(
                query="What supports the answer?",
                answer="The evidence supports it.",
                chunks=[Chunk(id="source-1", text="Evidence")],
            )
        """
        indexed_chunks = _index_chunks(chunks)
        if not indexed_chunks:
            return ChatModelChunksClassificationResult(classifications=())

        if prompt is None:
            prompt = PromptTemplate.from_template(load_chat_model_prompt())
        chain = prompt | self._model.with_structured_output(ChunkLabels)

        result = ChunkLabels.model_validate(
            await chain.ainvoke(
                {
                    "query": query,
                    "answer": answer,
                    "chunks": "\n\n---\n\n".join(
                        f'<chunk id="{escape(chunk.id, quote=True)}">\n'
                        f'{escape(chunk.text, quote=False)}\n</chunk>'
                        for chunk in indexed_chunks.values()
                    ),
                }
            )
        )
        labels = {label.chunk_id: label for label in result.classifications}
        if len(labels) != len(result.classifications):
            raise ValueError("Duplicate chunk IDs in model classifications")
        if labels.keys() != indexed_chunks.keys():
            raise ValueError("Model classifications must match input chunk IDs exactly")
        return ChatModelChunksClassificationResult(
            classifications=tuple(
                ChatModelChunkClassification(
                    chunk=chunk,
                    status=labels[chunk_id].status,
                    explanation=labels[chunk_id].explanation,
                )
                for chunk_id, chunk in indexed_chunks.items()
            )
        )
