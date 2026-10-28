"""Vestigia: trace answers back to evidence.

Public APIs are exposed from their domain subpackages rather than re-exported
from this root package.

Examples
--------
>>> from vestigia.chunk_classification import Chunk, ChunkStatus
>>> Chunk(id="source-1", text="Evidence").id, ChunkStatus.USED.value
('source-1', 'used')
"""
