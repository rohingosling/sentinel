#-----------------------------------------------------------------------------------------------------------------------
# Module:  embeddings.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Local ONNX text embeddings (Decision 2).
#
#   Embeddings are computed on this machine, by `fastembed` running `all-MiniLM-L6-v2` through `onnxruntime`, and never
#   by a cloud provider. Three reasons, in the order they mattered:
#
#     1. Memory is the most privacy-sensitive data Sentinel holds -- a durable record of everything the user has said.
#     2. Retrieval sits in the critical path of every turn, so a cloud round trip would tax every interaction.
#     3. Anthropic has no embeddings endpoint at all, so the primary LLM cannot supply embed() even in principle.
#
#   Two properties of the model are relied on elsewhere and are asserted here rather than assumed:
#
#     * The output is 384-dimensional. sqlite-vec stores whatever width it is handed, so a disagreement between the
#       model, the configured width, and the schema corrupts search silently instead of failing.
#     * The output is already L2-normalised. That is what lets sqlite-vec's cosine distance be read directly as
#       1 - similarity, with no renormalisation step on the query path.
#
#   The model loads lazily. Constructing an Embedder is free; the ~90 MB of weights are read on the first embed() call,
#   so `sentinel version` and `sentinel key` never pay for a subsystem they do not use.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import struct

from pathlib import Path
from typing  import Any, Protocol, runtime_checkable

from sentinel.errors import EmbeddingError

logger = logging.getLogger ( __name__ )

# The bundled model, and its width. Both are Decision 2 and are matched -- changing one without the other is the exact
# failure the startup guards exist to catch.

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS      = 384

# Sentinel records the short model name in every vector's embedding_model column, because that is the identity a user
# recognises and what agent.yaml asks for. fastembed wants the fully qualified repository name. This maps between them
# so neither surface has to know about the other.

MODEL_IDENTIFIERS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


#-----------------------------------------------------------------------------------------------------------------------
# Vector encoding
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: pack_vector
#
# Description:
#
#   Encode a vector as the little-endian float32 blob sqlite-vec expects.
#
# Arguments:
#
#   vector : The vector components.
#
# Returns:
#
#   The packed bytes. The cast to float32 is deliberate and lossy: fastembed returns float64, sqlite-vec stores
#   FLOAT[n] as float32, and packing at the model's width would double every stored vector for precision the cosine
#   comparison cannot use.
#
#-----------------------------------------------------------------------------------------------------------------------

def pack_vector ( vector: list [ float ] ) -> bytes:

    # Return data to caller.

    return struct.pack ( f"<{len ( vector )}f", *vector )


#-----------------------------------------------------------------------------------------------------------------------
# Function: unpack_vector
#
# Description:
#
#   Decode a sqlite-vec blob back into a vector.
#
# Arguments:
#
#   blob : The stored bytes.
#
# Returns:
#
#   The vector components.
#
#   Raises EmbeddingError if the blob length is not a whole number of float32 values.
#
#-----------------------------------------------------------------------------------------------------------------------

def unpack_vector ( blob: bytes ) -> list [ float ]:

    # A blob of the wrong length means the column holds something that is not a vector at all.

    if len ( blob ) % 4 != 0:
        raise EmbeddingError (
            f"Stored vector is {len ( blob )} bytes, which is not a whole number of 32-bit floats."
        )

    # Return data to caller.

    return list ( struct.unpack ( f"<{len ( blob ) // 4}f", blob ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: cosine_similarity
#
# Description:
#
#   Cosine similarity between two vectors.
#
#   Used only where a stored vector has to be compared outside a sqlite-vec KNN query -- scoring an episodic candidate
#   that the semantic search did not return. The KNN path takes its similarity from the extension's own distance.
#
# Arguments:
#
#   left  : First vector.
#   right : Second vector.
#
# Returns:
#
#   The similarity, -1.0 to 1.0. Zero when either vector has no magnitude or the widths differ, since neither has a
#   meaningful angle and raising would make an unscoreable candidate fail a whole retrieval.
#
#-----------------------------------------------------------------------------------------------------------------------

def cosine_similarity ( left: list [ float ], right: list [ float ] ) -> float:

    # Mismatched widths have no defined angle.

    if len ( left ) != len ( right ) or not left:
        return 0.0

    dot         = sum ( a * b for a, b in zip ( left, right, strict = True ) )
    left_scale  = sum ( a * a for a in left ) ** 0.5
    right_scale = sum ( b * b for b in right ) ** 0.5

    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0

    # Return data to caller.

    return float ( dot / ( left_scale * right_scale ) )


#-----------------------------------------------------------------------------------------------------------------------
# Embedder interface
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: Embedder
#
# Description:
#
#   What the memory system needs from an embedding model.
#
#   A Protocol rather than a base class, so the suite can supply a deterministic stand-in without inheriting the ONNX
#   runtime -- most memory tests are about storage and scoring, and loading 90 MB of weights to assert that a top-k
#   query returns ten rows would make the suite slow for no added confidence.
#
# Attributes:
#
#   model_name : The identity recorded against every vector this embedder produces.
#   dimensions : Output width.
#-----------------------------------------------------------------------------------------------------------------------

@runtime_checkable
class Embedder ( Protocol ):

    model_name: str
    dimensions: int

    #-------------------------------------------------------------------------------------------------------------------
    # Function: embed
    #
    # Description:
    #
    #   Embed a batch of texts.
    #
    # Arguments:
    #
    #   texts : The texts to embed.
    #
    # Returns:
    #
    #   One vector per input, in input order.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def embed ( self, texts: list [ str ] ) -> list [ list [ float ] ]:

        ...


#-----------------------------------------------------------------------------------------------------------------------
# Function: embed_one
#
# Description:
#
#   Embed a single text.
#
#   A free function rather than a second Protocol method, so every stand-in in the suite has one method to implement
#   and the batching path is the only one that can drift.
#
# Arguments:
#
#   embedder : The embedder to use.
#   text     : The text to embed.
#
# Returns:
#
#   The vector.
#
#   Raises EmbeddingError if the embedder returned nothing.
#
#-----------------------------------------------------------------------------------------------------------------------

def embed_one ( embedder: Embedder, text: str ) -> list [ float ]:

    vectors = embedder.embed ( [ text ] )

    if not vectors:
        raise EmbeddingError ( f"The embedding model {embedder.model_name!r} returned no vector for the query." )

    # Return data to caller.

    return vectors [ 0 ]


#-----------------------------------------------------------------------------------------------------------------------
# fastembed implementation
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Class: FastEmbedEmbedder
#
# Description:
#
#   The shipped embedder: fastembed over a local ONNX model.
#
#   The weights are read from cache_directory, which is ${SENTINEL_DATA}/models rather than fastembed's default
#   temporary directory. A temp directory means the ~90 MB download is repeated after every reboot on some machines,
#   and it puts user-visible state outside the one tree that gets backed up.
#
# Attributes:
#
#   model_name      : Short model identity, recorded against every vector.
#   dimensions      : Output width, asserted against the model on first use.
#   cache_directory : Where the ONNX weights live.
#-----------------------------------------------------------------------------------------------------------------------

class FastEmbedEmbedder:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the embedder without loading the model.
    #
    # Arguments:
    #
    #   model_name      : Short model identity, e.g. "all-MiniLM-L6-v2".
    #   cache_directory : Directory holding the ONNX weights. fastembed's default when omitted.
    #   dimensions      : Expected output width.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   model_name: str                    = DEFAULT_EMBEDDING_MODEL,
                   cache_directory: str | Path | None = None,
                   dimensions: int = DEFAULT_DIMENSIONS ) -> None:

        self.model_name      = model_name
        self.dimensions      = dimensions
        self.cache_directory = Path ( cache_directory ) if cache_directory is not None else None

        # Loaded on first use. Constructing an embedder must stay free, because the orchestrator builds one during
        # startup and a 6-second model load there would sit between the user's double-click and their window.

        self._model: Any = None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: repository_name
    #
    # Description:
    #
    #   Map the short model name to the identifier fastembed expects.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The fully qualified repository name. An unrecognised short name is passed through unchanged, so a model
    #   fastembed knows about but this table does not is still usable by writing its full name in agent.yaml.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def repository_name ( self ) -> str:

        # Return data to caller.

        return MODEL_IDENTIFIERS.get ( self.model_name, self.model_name )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: load
    #
    # Description:
    #
    #   Load the ONNX model, once.
    #
    #   The first call may download the weights, which needs the network. Every later call, and every call on a machine
    #   where the installer already placed them, reads from cache_directory and makes no outbound request at all
    #   (T3.17).
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The loaded fastembed model.
    #
    #   Raises EmbeddingError if fastembed is not installed, or the model could not be loaded.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def load ( self ) -> Any:

        if self._model is not None:
            return self._model

        # Imported here rather than at module scope: fastembed pulls in onnxruntime, which is the single slowest import
        # in the dependency set, and the CLI's credential and version commands must not pay for it.

        try:
            from fastembed import TextEmbedding
        except ImportError as error:
            raise EmbeddingError (
                f"The fastembed package is not installed, so Sentinel cannot embed anything and memory retrieval "
                f"is unavailable. Install it with `pip install fastembed`. ({error})"
            ) from error

        try:
            self._model = TextEmbedding (
                model_name = self.repository_name (),
                cache_dir  = str ( self.cache_directory ) if self.cache_directory is not None else None,
            )
        except Exception as error:
            raise EmbeddingError (
                f"Could not load the embedding model {self.repository_name ()!r} from "
                f"{self.cache_directory or 'the fastembed default cache'}: {error}. The first run needs network "
                f"access to download roughly 90 MB of model weights; later runs do not."
            ) from error

        logger.info (
            "Embedding model %s loaded from %s.",
            self.repository_name (), self.cache_directory or "the fastembed default cache",
        )

        # Return data to caller.

        return self._model

    #-------------------------------------------------------------------------------------------------------------------
    # Function: embed
    #
    # Description:
    #
    #   Embed a batch of texts.
    #
    # Arguments:
    #
    #   texts : The texts to embed.
    #
    # Returns:
    #
    #   One vector per input, in input order. An empty input list returns an empty list without loading the model.
    #
    #   Raises EmbeddingError if the model could not be loaded, if embedding failed, or if the model's output width is
    #   not the width this embedder was told to expect.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def embed ( self, texts: list [ str ] ) -> list [ list [ float ] ]:

        # Nothing to embed is not a reason to read 90 MB from disk.

        if not texts:
            return []

        model = self.load ()

        try:
            vectors = [ [ float ( component ) for component in vector ] for vector in model.embed ( texts ) ]
        except Exception as error:
            raise EmbeddingError ( f"Embedding failed for {len ( texts )} text(s): {error}" ) from error

        # The width is checked on every batch rather than once at load, because it costs a comparison and because a
        # wrong width reaching sqlite-vec is stored rather than rejected.

        for vector in vectors:
            if len ( vector ) != self.dimensions:
                raise EmbeddingError (
                    f"The embedding model {self.model_name!r} produced a {len ( vector )}-dimensional vector, but "
                    f"{self.dimensions} was expected. Set database.vector_dimensions to {len ( vector )} in "
                    f"agent.yaml, or configure the model that matches the schema."
                )

        # Return data to caller.

        return vectors
