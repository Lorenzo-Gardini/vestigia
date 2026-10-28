"""LangChain-backed implementations for classifying evidence chunks.

Classifiers:
    - ChatModelChunkClassifier: structured-output chat-model backend
    - SystemOneModelChunkClassifier: TypeSafe system-one backend

Each classifier is imported lazily so callers need only install the optional
dependencies for the backend they use.

Examples
--------
>>> from vestigia.chunk_classification.langchain import ChatModelChunkClassifier
>>> from vestigia.chunk_classification.langchain import SystemOneModelChunkClassifier
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat_model_classifier import ChatModelChunkClassifier
    from .system_one_classifier import SystemOneModelChunkClassifier

__all__ = [
    "ChatModelChunkClassifier",
    "SystemOneModelChunkClassifier",
]


def __getattr__(name: str) -> Any:
    """Load a classifier only when its optional backend is requested.

    Examples
    --------
    >>> "ChatModelChunkClassifier" in __all__
    True
    """
    if name == "ChatModelChunkClassifier":
        from .chat_model_classifier import ChatModelChunkClassifier

        return ChatModelChunkClassifier
    if name == "SystemOneModelChunkClassifier":
        from .system_one_classifier import SystemOneModelChunkClassifier

        return SystemOneModelChunkClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
