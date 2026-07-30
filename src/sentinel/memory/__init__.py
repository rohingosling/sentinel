#-----------------------------------------------------------------------------------------------------------------------
# Module:  __init__.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Package initialiser; not executable directly.
#
# Description:
#
#   Three-tier memory (architecture 3.2.4).
#
#     working.py    Tier 1  diskcache      session-scoped, TTL-evicted, volatile by policy
#     episodic.py   Tier 2  SQLite         what happened and when -- the recency arm
#     semantic.py   Tier 3  sqlite-vec     what things mean -- the similarity arm
#     retriever.py          MemorySystem   the two arms blended into one ranked context
#     embeddings.py         fastembed      local ONNX vectors, never a cloud provider
#
#   Import MemorySystem from here. The tier modules are importable directly for tests and for the rare caller that
#   genuinely wants one tier, but application code should go through the facade so the three stay in step.
#-----------------------------------------------------------------------------------------------------------------------

from sentinel.memory.embeddings import (
    DEFAULT_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    FastEmbedEmbedder,
    cosine_similarity,
    embed_one,
)
from sentinel.memory.episodic import Episode, EpisodicMemory
from sentinel.memory.retriever import (
    MemorySystem,
    RetrievedMemory,
    blend_score,
    open_memory_system,
    recency_decay,
)
from sentinel.memory.semantic import SemanticMatch, SemanticMemory
from sentinel.memory.working  import WorkingMemory

__all__ = [
    "DEFAULT_DIMENSIONS",
    "DEFAULT_EMBEDDING_MODEL",
    "Embedder",
    "Episode",
    "EpisodicMemory",
    "FastEmbedEmbedder",
    "MemorySystem",
    "RetrievedMemory",
    "SemanticMatch",
    "SemanticMemory",
    "WorkingMemory",
    "blend_score",
    "cosine_similarity",
    "embed_one",
    "open_memory_system",
    "recency_decay",
]
