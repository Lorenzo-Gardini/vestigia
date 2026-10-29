"""System-one implementation of the chunk classifier protocol.

Example:

    classifier = SystemOneModelChunkClassifier(configured_typesafe_model)
    result = await classifier.classify(
        query="What supports the answer?",
        answer="The evidence supports it.",
        chunks=[Chunk(id="source-1", text="Evidence")],
    )

"""

from collections.abc import Sequence

from langchain_core.prompts import PromptTemplate
from langchain_typesafe import Choice, ChoiceAnswer, TypeSafeClassifier

from vestigia.chunk_classification import (
    Chunk,
    ChunkStatus,
    SystemOneModelChunkClassification,
    SystemOneModelChunksClassificationResult,
    SystemOneModelTemplate,
)
from vestigia.chunk_classification.classifier import (
    ChunkClassifier,
    ModelResponseError,
    _check_response_ids,
    _index_chunks,
)
from vestigia.chunk_classification.prompts import load_system_one_prompt


class SystemOneModelChunkClassifier(
    ChunkClassifier[SystemOneModelTemplate, SystemOneModelChunksClassificationResult]
):
    """Classify chunks with a TypeSafe system-one model.

    Examples
    --------
    Given a configured TypeSafe model:

        classifier = SystemOneModelChunkClassifier(model)
        result = await classifier.classify(
            query="What supports the answer?",
            answer="The evidence supports it.",
            chunks=[Chunk(id="source-1", text="Evidence")],
        )
    """

    def __init__(self, model: TypeSafeClassifier) -> None:
        """Initialize the classifier with a TypeSafe model.

        Examples
        --------
        Given an application-configured model:

            classifier = SystemOneModelChunkClassifier(model)
        """
        self._model = model

    async def classify(
        self,
        *,
        query: str,
        answer: str,
        chunks: Sequence[Chunk],
        prompt: SystemOneModelTemplate | None = None,
    ) -> SystemOneModelChunksClassificationResult:
        """Classify chunks and preserve results by their stable chunk IDs.

        An empty sequence returns an empty result without calling the model.
        The template state receives ``query`` and ``answer``; its instruction
        receives each chunk's ``text`` as ``chunk``.

        Parameters
        ----------
        query
            Original user question, used as attribution context.
        answer
            Generated answer to compare with each source chunk.
        chunks
            Source chunks with unique IDs within this call.
        prompt
            Optional state, instruction, and status criteria. When omitted,
            the packaged YAML template is loaded.

        Returns
        -------
        SystemOneModelChunksClassificationResult
            Classifications in input order, including probabilities and confidence.

        Raises
        ------
        ValueError
            If input IDs are duplicated.
        ModelResponseError
            If answer IDs differ from input IDs, an answer is not a choice, or
            its labels are absent from the criteria.

        Backend, template-formatting, and Pydantic validation errors propagate.

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
            return SystemOneModelChunksClassificationResult(classifications=())

        if prompt is None:
            prompt = load_system_one_prompt()

        filled_state: str = PromptTemplate.from_template(prompt.state).format(
            query=query, answer=answer
        )
        filled_instructions: dict[str, str] = {
            chunk.id: (
                PromptTemplate.from_template(prompt.instruction).format(
                    chunk=chunk.text
                )
            )
            for chunk in indexed_chunks.values()
        }

        response = await self._model.ainvoke(
            {
                "state": filled_state,
                "questions": {
                    chunk_id: Choice(instructions=chunk_text, criteria=prompt.criteria)
                    for chunk_id, chunk_text in filled_instructions.items()
                },
            }
        )
        _check_response_ids(
            "Model answer IDs must match input chunk IDs exactly",
            indexed_chunks,
            list(response.answers),
        )
        classifications: list[SystemOneModelChunkClassification] = []
        for chunk in indexed_chunks.values():
            chunk_answer = response.answers[chunk.id]
            if not isinstance(chunk_answer, ChoiceAnswer):
                raise ModelResponseError(f"Expected a choice answer for chunk {chunk.id!r}")
            if (
                chunk_answer.choice not in prompt.criteria
                or not chunk_answer.probabilities.keys() <= prompt.criteria.keys()
            ):
                raise ModelResponseError(
                    f"Model answer uses a status outside the criteria: {chunk.id!r}"
                )
            classifications.append(
                SystemOneModelChunkClassification(
                    chunk=chunk,
                    status=ChunkStatus(chunk_answer.choice),
                    probabilities={
                        ChunkStatus(status): probability
                        for status, probability in chunk_answer.probabilities.items()
                    },
                    confidence=chunk_answer.confidence,
                )
            )

        return SystemOneModelChunksClassificationResult(
            classifications=tuple(classifications)
        )
