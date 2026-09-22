#-----------------------------------------------------------------------------------------------------------------------
# Module:  semantic.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Tier 3: semantic memory (architecture 3.2.4).
#
#   Meaning-addressed storage over sqlite-vec. The vec0 virtual table carries the 384-dimensional embedding plus every
#   field needed to render a hit, so one KNN statement answers a search rather than a KNN followed by a join.
#
#   The table is declared distance_metric=cosine, which makes the reported distance exactly 1 - cosine similarity. That
#   identity is used throughout and holds because the model returns L2-normalised vectors (see embeddings.py); it is
#   not a general property of the extension.
#
#   Two guards run before this tier will answer anything, and both exist because their failure mode is silence rather
#   than an error:
#
#     * Width. sqlite-vec stores whatever width it is handed. If the configured database.vector_dimensions, the
#       embedding model's actual output, and the schema's FLOAT[n] do not all agree, search returns plausible garbage.
#       All three are compared, and a disagreement aborts.
#     * Model identity. Vectors from two different models share a coordinate space only by coincidence. Changing
#       embedding_model in agent.yaml against a populated store does not error -- recall simply gets worse. Every
#       vector therefore records the model that produced it, and a mismatch blocks search until the store is
#       re-embedded.
#
#   Blocking is the point. A memory system that quietly forgets is worse than one that says it cannot currently
#   remember, because only the second gets fixed.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import re
import uuid

from dataclasses import dataclass
from datetime    import datetime

import aiosqlite

from sentinel.database import DATABASE_FAULTS
from sentinel.errors import (
    DatabaseError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    VectorDimensionError,
)
from sentinel.memory.embeddings import pack_vector, unpack_vector
from sentinel.memory.episodic   import to_iso_timestamp

logger = logging.getLogger ( __name__ )

# The vec0 table, and the auxiliary columns a hit is rendered from.

VECTOR_TABLE   = "semantic_vectors"
VECTOR_COLUMNS = "id, content, category, timestamp, embedding_model"

# Pulls the declared width out of the stored CREATE statement. vec0 exposes no pragma for it, and its shadow tables
# describe the storage rather than the declaration, so the schema text is the authoritative source.

DECLARED_WIDTH_PATTERN = re.compile ( r"FLOAT\s*\[\s*(\d+)\s*\]", re.IGNORECASE )

# Default category for a stored fact when the caller does not classify it.

CATEGORY_KNOWLEDGE = "knowledge"


#-----------------------------------------------------------------------------------------------------------------------
# Class: SemanticMatch
#
# Description:
#
#   One hit from a similarity search.
#
# Attributes:
#
#   id              : Record identifier. Shares the episode's id when the record was written alongside an episode.
#   content         : The original text.
#   category        : Memory type.
#   timestamp       : ISO 8601 creation time.
#   embedding_model : Model that produced the vector.
#   similarity      : Cosine similarity to the query, 0.0 to 1.0 for normalised vectors.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class SemanticMatch:

    id:              str
    content:         str
    category:        str
    timestamp:       str
    embedding_model: str
    similarity:      float


#-----------------------------------------------------------------------------------------------------------------------
# Class: SemanticMemory
#
# Description:
#
#   Vector storage and nearest-neighbour search.
#
# Attributes:
#
#   connection      : The shared aiosqlite connection.
#   model_name      : Configured embedding model identity, recorded on every write.
#   dimensions      : Configured vector width.
#   blocked_reason  : Why search is refused, or None when the tier is healthy.
#-----------------------------------------------------------------------------------------------------------------------

class SemanticMemory:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Bind to an open connection.
    #
    # Arguments:
    #
    #   connection : An open aiosqlite connection with sqlite-vec loaded and the schema migrated.
    #   model_name : Configured embedding model identity.
    #   dimensions : Configured vector width.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   connection: aiosqlite.Connection,
                   model_name: str,
                   dimensions: int ) -> None:

        self.connection = connection
        self.model_name = model_name
        self.dimensions = dimensions
        self.blocked_reason: str | None = None

    #-------------------------------------------------------------------------------------------------------------------
    # Startup guards
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: declared_dimensions
    #
    # Description:
    #
    #   The vector width the schema declares.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The declared width, or None when the table is absent or its declaration cannot be read.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def declared_dimensions ( self ) -> int | None:

        async with self.connection.execute (
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", ( VECTOR_TABLE, )
        ) as cursor:
            row = await cursor.fetchone ()

        if not row or row [ 0 ] is None:
            return None

        match = DECLARED_WIDTH_PATTERN.search ( str ( row [ 0 ] ) )

        # Return data to caller.

        return int ( match.group ( 1 ) ) if match else None

    #-------------------------------------------------------------------------------------------------------------------
    # Function: stored_models
    #
    # Description:
    #
    #   Every distinct embedding model represented in the store.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The model identities, sorted. Empty for an empty store, which is why a fresh install never trips the identity
    #   guard.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def stored_models ( self ) -> list [ str ]:

        async with self.connection.execute (
            f"SELECT DISTINCT embedding_model FROM {VECTOR_TABLE}"
        ) as cursor:
            rows = await cursor.fetchall ()

        # Return data to caller.

        return sorted ( str ( row [ 0 ] ) for row in rows if row [ 0 ] is not None )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: assert_dimensions
    #
    # Description:
    #
    #   Confirm the configured width, the model's width, and the schema's width all agree.
    #
    #   Run at startup, before anything is stored. The failure this prevents is not a crash but a store full of
    #   truncated or padded vectors whose search results look ordinary and are wrong (T3.18).
    #
    # Arguments:
    #
    #   model_dimensions : The embedding model's actual output width. Skipped when None, which is how a caller checks
    #                      the schema against configuration without paying to load the model.
    #
    # Returns:
    #
    #   The agreed width.
    #
    #   Raises VectorDimensionError on any disagreement.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def assert_dimensions ( self, model_dimensions: int | None = None ) -> int:

        # The model is the authority on what it produces, so it is checked first and named first.

        if model_dimensions is not None and model_dimensions != self.dimensions:
            raise VectorDimensionError (
                f"The embedding model {self.model_name!r} produces {model_dimensions}-dimensional vectors, but "
                f"database.vector_dimensions is {self.dimensions}. sqlite-vec stores whatever width it is given, so "
                f"leaving this mismatched would corrupt every search silently rather than failing. Set "
                f"database.vector_dimensions to {model_dimensions}, or configure the model that matches."
            )

        declared = await self.declared_dimensions ()

        if declared is not None and declared != self.dimensions:
            raise VectorDimensionError (
                f"The {VECTOR_TABLE} table was created as FLOAT[{declared}], but database.vector_dimensions is "
                f"{self.dimensions}. The schema width is fixed when the table is created and cannot be changed in "
                f"place. Set database.vector_dimensions to {declared}, or delete the database and re-initialise to "
                f"rebuild it at the new width -- which discards every stored vector."
            )

        # Return data to caller.

        return self.dimensions

    #-------------------------------------------------------------------------------------------------------------------
    # Function: assert_model_identity
    #
    # Description:
    #
    #   Confirm every stored vector came from the configured model.
    #
    #   On a mismatch the tier is blocked rather than left to degrade: search raises until the store is re-embedded or
    #   the configuration is put back (T3.19).
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   True when the store is consistent with the configured model, including when it is empty.
    #
    #   Raises EmbeddingModelMismatchError when stored vectors came from a different model.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def assert_model_identity ( self ) -> bool:

        models = await self.stored_models ()

        # An empty store agrees with every model. This is the fresh-install path.

        if not models:
            self.blocked_reason = None

            return True

        foreign = [ name for name in models if name != self.model_name ]

        if not foreign:
            self.blocked_reason = None

            return True

        reason = (
            f"Semantic memory holds vectors produced by {', '.join ( repr ( name ) for name in foreign )}, but the "
            f"configured embedding model is {self.model_name!r}. Vectors from different models are not comparable, so "
            f"searching across them would quietly return the wrong memories instead of failing. Re-embed the store "
            f"with the configured model, or set database.embedding_model back to the model that wrote them."
        )

        self.blocked_reason = reason

        raise EmbeddingModelMismatchError ( reason )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: assert_ready
    #
    # Description:
    #
    #   Run both startup guards.
    #
    # Arguments:
    #
    #   model_dimensions : The embedding model's actual output width, when it is known.
    #
    # Returns:
    #
    #   None.
    #
    #   Raises VectorDimensionError or EmbeddingModelMismatchError, whichever guard fails first.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def assert_ready ( self, model_dimensions: int | None = None ) -> None:

        await self.assert_dimensions ( model_dimensions )
        await self.assert_model_identity ()

        logger.debug (
            "Semantic memory ready: %d-dimensional vectors from %s.", self.dimensions, self.model_name
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: raise_if_blocked
    #
    # Description:
    #
    #   Refuse an operation when a guard has failed.
    #
    #   Called on the read path rather than only at startup, because the config can be reloaded and the store can be
    #   written to by a second process between one turn and the next.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   None.
    #
    #   Raises EmbeddingModelMismatchError when the tier is blocked.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def raise_if_blocked ( self ) -> None:

        if self.blocked_reason is not None:
            raise EmbeddingModelMismatchError ( self.blocked_reason )

    #-------------------------------------------------------------------------------------------------------------------
    # Writes
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: store
    #
    # Description:
    #
    #   Store one vector and its auxiliary fields.
    #
    #   Replaces any existing record with the same identifier, because vec0 rejects a duplicate primary key outright
    #   and "store this memory again after editing it" is an ordinary thing for the agent to do.
    #
    # Arguments:
    #
    #   vector    : The embedding.
    #   content   : The original text.
    #   category  : Memory type.
    #   moment    : Creation time. Now when omitted.
    #   record_id : Explicit identifier. A fresh UUID when omitted; pass the episode's id to pair the two tiers.
    #
    # Returns:
    #
    #   The record's identifier.
    #
    #   Raises VectorDimensionError if the vector is not the configured width, or DatabaseError if the write failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def store ( self,
                      vector: list [ float ],
                      content: str,
                      category: str           = CATEGORY_KNOWLEDGE,
                      moment: datetime | None = None,
                      record_id: str | None = None ) -> str:

        # Checked here as well as at startup: a caller can hand over a vector from anywhere, and the whole point of the
        # width guard is that sqlite-vec would accept a wrong one.

        if len ( vector ) != self.dimensions:
            raise VectorDimensionError (
                f"Refusing to store a {len ( vector )}-dimensional vector in a {self.dimensions}-dimensional store."
            )

        # A vector with no magnitude has no angle to anything, so cosine search cannot rank it. sqlite-vec accepts it
        # and then reports a NULL distance for it on every query, where it sorts ahead of genuine matches. Refusing it
        # at the door is the only place this stays cheap to diagnose.

        if not any ( component != 0.0 for component in vector ):
            raise EmbeddingError (
                f"Refusing to store a zero-magnitude vector for {content [ :60 ]!r}. Cosine similarity is undefined "
                f"against the origin, so the record would be unsearchable and would sort ahead of real matches."
            )

        identifier = record_id if record_id is not None else str ( uuid.uuid4 () )

        try:
            await self.connection.execute ( f"DELETE FROM {VECTOR_TABLE} WHERE id = ?", ( identifier, ) )

            await self.connection.execute (
                f"INSERT INTO {VECTOR_TABLE} ( id, embedding, content, category, timestamp, embedding_model ) "
                f"VALUES ( ?, ?, ?, ?, ?, ? )",
                (
                    identifier,
                    pack_vector ( vector ),
                    content,
                    category,
                    to_iso_timestamp ( moment ),
                    self.model_name,
                ),
            )

            await self.connection.commit ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Could not store a semantic vector: {error}" ) from error

        # Return data to caller.

        return identifier

    #-------------------------------------------------------------------------------------------------------------------
    # Function: delete
    #
    # Description:
    #
    #   Remove one record.
    #
    # Arguments:
    #
    #   record_id : The identifier to remove.
    #
    # Returns:
    #
    #   True when a record was removed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def delete ( self, record_id: str ) -> bool:

        cursor = await self.connection.execute ( f"DELETE FROM {VECTOR_TABLE} WHERE id = ?", ( record_id, ) )

        await self.connection.commit ()

        # Return data to caller.

        return bool ( cursor.rowcount )

    #-------------------------------------------------------------------------------------------------------------------
    # Reads
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: search
    #
    # Description:
    #
    #   Nearest neighbours of a query vector.
    #
    # Arguments:
    #
    #   vector    : The query embedding.
    #   limit     : Maximum neighbours to consider.
    #   threshold : Minimum cosine similarity a hit must reach. None applies no floor.
    #
    # Returns:
    #
    #   Matches ordered by similarity, most similar first, with everything below the threshold removed. An empty store
    #   returns an empty list rather than raising (T3.16).
    #
    #   Raises EmbeddingModelMismatchError if the tier is blocked, VectorDimensionError if the query is the wrong
    #   width, or DatabaseError if the query failed.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def search ( self,
                       vector: list [ float ],
                       limit: int = 10,
                       threshold: float | None = None ) -> list [ SemanticMatch ]:

        self.raise_if_blocked ()

        if len ( vector ) != self.dimensions:
            raise VectorDimensionError (
                f"Refusing to search a {self.dimensions}-dimensional store with a {len ( vector )}-dimensional query."
            )

        # A non-positive k is a caller asking for nothing, not an error. sqlite-vec would reject it.

        neighbours = max ( int ( limit ), 0 )

        if neighbours == 0:
            return []

        try:
            async with self.connection.execute (
                f"SELECT {VECTOR_COLUMNS}, distance FROM {VECTOR_TABLE} "
                f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                ( pack_vector ( vector ), neighbours ),
            ) as cursor:
                rows = await cursor.fetchall ()

        except DATABASE_FAULTS as error:
            raise DatabaseError ( f"Semantic search failed: {error}" ) from error

        matches: list [ SemanticMatch ] = []

        for row in rows:

            # sqlite-vec reports a NULL distance for a stored vector with no magnitude -- there is no angle between the
            # query and the origin. store() refuses such vectors, so this can only be a row written before that guard
            # existed or by something other than Sentinel. Scoring it as entirely dissimilar keeps one bad row from
            # taking down a whole retrieval, and the threshold then discards it.

            if row [ 5 ] is None:
                logger.warning ( "Semantic record %s has no usable vector and was skipped.", row [ 0 ] )

                continue

            # The table is declared distance_metric=cosine, so distance is exactly 1 - similarity.

            similarity = 1.0 - float ( row [ 5 ] )

            if threshold is not None and similarity < threshold:
                continue

            matches.append (
                SemanticMatch (
                    id              = str ( row [ 0 ] ),
                    content         = str ( row [ 1 ] ) if row [ 1 ] is not None else "",
                    category        = str ( row [ 2 ] ) if row [ 2 ] is not None else "",
                    timestamp       = str ( row [ 3 ] ) if row [ 3 ] is not None else "",
                    embedding_model = str ( row [ 4 ] ) if row [ 4 ] is not None else "",
                    similarity      = similarity,
                )
            )

        # Return data to caller.

        return matches

    #-------------------------------------------------------------------------------------------------------------------
    # Function: vectors_for
    #
    # Description:
    #
    #   Read stored embeddings by identifier.
    #
    #   Used by the unified retriever to score an episodic candidate the KNN did not return: without its vector, a
    #   recent episode has no similarity term at all and would be ranked on recency alone.
    #
    # Arguments:
    #
    #   record_ids : Identifiers to look up.
    #
    # Returns:
    #
    #   Identifiers mapped to their vectors. Identifiers with no stored vector are simply absent from the mapping.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def vectors_for ( self, record_ids: list [ str ] ) -> dict [ str, list [ float ] ]:

        if not record_ids:
            return {}

        # One placeholder per identifier, so the values stay bound rather than interpolated.

        placeholders = ", ".join ( "?" for _ in record_ids )

        async with self.connection.execute (
            f"SELECT id, embedding FROM {VECTOR_TABLE} WHERE id IN ( {placeholders} )",
            tuple ( record_ids ),
        ) as cursor:
            rows = await cursor.fetchall ()

        vectors: dict [ str, list [ float ] ] = {}

        for row in rows:
            if row [ 1 ] is not None:
                vectors [ str ( row [ 0 ] ) ] = unpack_vector ( bytes ( row [ 1 ] ) )

        # Return data to caller.

        return vectors

    #-------------------------------------------------------------------------------------------------------------------
    # Function: get
    #
    # Description:
    #
    #   Read one record's auxiliary fields by identifier.
    #
    # Arguments:
    #
    #   record_id : The identifier to look up.
    #
    # Returns:
    #
    #   The record with a similarity of 0.0, since no query was involved, or None when no such record exists.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def get ( self, record_id: str ) -> SemanticMatch | None:

        async with self.connection.execute (
            f"SELECT {VECTOR_COLUMNS} FROM {VECTOR_TABLE} WHERE id = ?", ( record_id, )
        ) as cursor:
            row = await cursor.fetchone ()

        if not row:
            return None

        # Return data to caller.

        return SemanticMatch (
            id              = str ( row [ 0 ] ),
            content         = str ( row [ 1 ] ) if row [ 1 ] is not None else "",
            category        = str ( row [ 2 ] ) if row [ 2 ] is not None else "",
            timestamp       = str ( row [ 3 ] ) if row [ 3 ] is not None else "",
            embedding_model = str ( row [ 4 ] ) if row [ 4 ] is not None else "",
            similarity      = 0.0,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: count
    #
    # Description:
    #
    #   How many vectors are stored.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   The record count.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def count ( self ) -> int:

        async with self.connection.execute ( f"SELECT count(*) FROM {VECTOR_TABLE}" ) as cursor:
            row = await cursor.fetchone ()

        # Return data to caller.

        return int ( row [ 0 ] ) if row else 0
