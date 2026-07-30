#-----------------------------------------------------------------------------------------------------------------------
# Module:  retriever.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   Unified retrieval across the three memory tiers (architecture 3.2.4).
#
#   The algorithm:
#
#       1. q     <- embed(t)                            Query embedding.
#       2. S_sem <- semantic.search(q, k, theta)        Meaning: what is this about?
#       3. S_rec <- episodic.recent(tau, k)             Recency:  what were we just doing?
#       4. score <- (1 - lambda) * cosine(q, m)
#                 + lambda       * recency_decay(tau - m.timestamp)
#       5. C     <- top_k(S_sem union S_rec, by=score, k)
#
#   with recency_decay(dt) = exp(-dt / tau_half) and tau_half = 7 days by default.
#
#   ONE DELIBERATE DEVIATION from the algorithm as written in the architecture document, and it is not cosmetic. The
#   document applies theta to the blended score at step 5. Doing that makes the semantic tier unable to do its job: at
#   theta = 0.72 and lambda = 0.3, a memory with an excellent cosine of 0.75 falls below the threshold once it is a
#   month old (0.7*0.75 + 0.3*0.05 = 0.54), so the tier that exists specifically to recall old, relevant things would
#   discard exactly those. Symmetrically, a recent episode with no vector at all can never clear 0.72 on recency alone,
#   so the recency arm would return nothing.
#
#   Theta is therefore applied to the COSINE, where it is a relevance floor and behaves as T3.12 describes, and the
#   blended score is used for RANKING the union. Recency candidates enter on recency merit and are ranked, not gated.
#   Architecture 3.2.4 has been amended to match.
#
#   Both arms are retrieved with a widened k before scoring. Taking exactly k from each and then blending would let a
#   candidate that ranks eleventh on similarity but first on the blended score be dropped before it was ever scored.
#-----------------------------------------------------------------------------------------------------------------------

import logging
import math

from dataclasses import dataclass
from datetime    import UTC, datetime
from typing      import Any

import aiosqlite
import diskcache

from sentinel.config import SentinelConfig
from sentinel.memory.embeddings import (
    Embedder,
    FastEmbedEmbedder,
    cosine_similarity,
    embed_one,
)
from sentinel.memory.episodic import (
    CATEGORY_CONVERSATION,
    Episode,
    EpisodicMemory,
    from_iso_timestamp,
    to_iso_timestamp,
)
from sentinel.errors          import MemorySystemError
from sentinel.memory.semantic import SemanticMatch, SemanticMemory
from sentinel.memory.working  import DEFAULT_SESSION, WorkingMemory

logger = logging.getLogger ( __name__ )

# Seconds in a day, for turning the configured half-life into the units the decay works in.

SECONDS_PER_DAY = 86400.0

# How much wider than k each arm is retrieved before scoring. Three is a judgement, not a measurement: it is enough
# that a candidate ranked outside the top k on one axis can still win on the blend, and small enough that the KNN cost
# is unchanged at this scale.

CANDIDATE_WIDENING = 3

# Where a retrieved memory came from. Carried through to the caller so the prompt can say "from a past conversation"
# rather than presenting recalled facts and recent events identically.

SOURCE_SEMANTIC = "semantic"
SOURCE_EPISODIC = "episodic"


#-----------------------------------------------------------------------------------------------------------------------
# Scoring
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: recency_decay
#
# Description:
#
#   Exponential decay of a memory's recency contribution.
#
#   A SECOND DELIBERATE DEVIATION from the architecture document, of the same kind as the threshold one above. The
#   document writes the decay as exp(-dt / tau_half) while calling tau_half a half-life. Those two statements
#   disagree: exp(-1) is 0.368, so under that formula a memory at "one half-life" keeps 37% of its recency, not 50%.
#   That is a time constant, not a half-life. The parameter is named half-life in the algorithm, in agent.yaml, and in
#   every place a user will read it, so the implementation is a true half-life -- 2^(-dt / tau_half) -- and
#   architecture 3.2.4 has been amended to match.
#
#   The difference at the 7-day default is small but not negligible: a fortnight-old memory scores 0.25 rather than
#   0.135, roughly twice the recency contribution.
#
# Arguments:
#
#   age_seconds       : How long ago the memory was recorded. Negative ages are treated as zero, since a memory dated
#                       in the future is a clock problem and must not score above a memory from this instant.
#   half_life_seconds : Age at which the contribution has halved.
#
# Returns:
#
#   A value in (0.0, 1.0]: 1.0 for something recorded now, exactly 0.5 at one half-life, 0.25 at two, approaching
#   zero thereafter.
#
#-----------------------------------------------------------------------------------------------------------------------

def recency_decay ( age_seconds: float, half_life_seconds: float ) -> float:

    # A non-positive half-life would divide by zero or invert the curve. Treat it as "no decay".

    if half_life_seconds <= 0.0:
        return 1.0

    age = max ( age_seconds, 0.0 )

    # Return data to caller.

    return float ( math.exp ( -math.log ( 2.0 ) * age / half_life_seconds ) )


#-----------------------------------------------------------------------------------------------------------------------
# Function: blend_score
#
# Description:
#
#   Combine similarity and recency into one ranking score.
#
# Arguments:
#
#   similarity     : Cosine similarity to the query, 0.0 to 1.0.
#   recency        : Recency decay, 0.0 to 1.0.
#   recency_weight : Share given to recency. 0.0 ranks on meaning alone, 1.0 on age alone.
#
# Returns:
#
#   The blended score.
#
#-----------------------------------------------------------------------------------------------------------------------

def blend_score ( similarity: float, recency: float, recency_weight: float ) -> float:

    weight = min ( max ( recency_weight, 0.0 ), 1.0 )

    # Return data to caller.

    return ( 1.0 - weight ) * similarity + weight * recency


#-----------------------------------------------------------------------------------------------------------------------
# Class: RetrievedMemory
#
# Description:
#
#   One memory selected by unified retrieval.
#
# Attributes:
#
#   id         : Record identifier. Episodic and semantic records written together share one.
#   summary    : The text to put in front of the model.
#   category   : Memory type.
#   timestamp  : ISO 8601 instant the memory describes.
#   source     : "semantic" or "episodic" -- which arm surfaced it.
#   similarity : Cosine similarity to the query. Zero when the record has no stored vector.
#   recency    : Recency decay at retrieval time.
#   score      : The blended ranking score.
#-----------------------------------------------------------------------------------------------------------------------

@dataclass ( frozen = True )
class RetrievedMemory:

    id:         str
    summary:    str
    category:   str
    timestamp:  str
    source:     str
    similarity: float
    recency:    float
    score:      float

    #-------------------------------------------------------------------------------------------------------------------
    # Function: as_prompt_entry
    #
    # Description:
    #
    #   The shape prompt.build_memory_section expects.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   A mapping with "summary", "timestamp", "category", "source", and "score".
    #
    #-------------------------------------------------------------------------------------------------------------------

    def as_prompt_entry ( self ) -> dict [ str, Any ]:

        # Return data to caller.

        return {
            "summary": self.summary,
            "timestamp": self.timestamp,
            "category": self.category,
            "source": self.source,
            "score": self.score,
        }


#-----------------------------------------------------------------------------------------------------------------------
# Class: MemorySystem
#
# Description:
#
#   The three tiers behind one interface.
#
#   Owns nothing it did not open: the connection and the cache are handed in, because the whole agent shares one SQLite
#   connection (WAL is single-writer) and one diskcache store.
#
# Attributes:
#
#   working       : Tier 1.
#   episodic      : Tier 2.
#   semantic      : Tier 3.
#   embedder      : The embedding model.
#   configuration : The loaded configuration, for the retrieval parameters.
#-----------------------------------------------------------------------------------------------------------------------

class MemorySystem:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Assemble the three tiers.
    #
    # Arguments:
    #
    #   configuration : The loaded configuration.
    #   working       : Tier 1.
    #   episodic      : Tier 2.
    #   semantic      : Tier 3.
    #   embedder      : The embedding model.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self,
                   configuration: SentinelConfig,
                   working: WorkingMemory,
                   episodic: EpisodicMemory,
                   semantic: SemanticMemory,
                   embedder: Embedder ) -> None:

        self.configuration = configuration
        self.working       = working
        self.episodic      = episodic
        self.semantic      = semantic
        self.embedder      = embedder

    #-------------------------------------------------------------------------------------------------------------------
    # Function: build
    #
    # Description:
    #
    #   Assemble a memory system from an open connection and cache.
    #
    #   Does not run the startup guards -- assert_ready does, and it is separate so a caller can construct the system
    #   without loading the embedding model, which is what most of the suite wants.
    #
    # Arguments:
    #
    #   configuration : The loaded configuration.
    #   connection    : An open aiosqlite connection with sqlite-vec loaded and the schema migrated.
    #   cache         : An open diskcache store.
    #   session_id    : Session to scope working memory to.
    #   embedder      : The embedding model. A FastEmbedEmbedder reading ${SENTINEL_DATA}/models when omitted.
    #
    # Returns:
    #
    #   The assembled MemorySystem.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @classmethod
    def build ( cls,
                configuration: SentinelConfig,
                connection: aiosqlite.Connection,
                cache: diskcache.Cache,
                session_id: str = DEFAULT_SESSION,
                embedder: Embedder | None = None ) -> "MemorySystem":

        model = embedder if embedder is not None else FastEmbedEmbedder (
            model_name      = configuration.database.embedding_model,
            cache_directory = configuration.models_directory,
            dimensions      = configuration.database.vector_dimensions,
        )

        # Return data to caller.

        return cls (
            configuration = configuration,
            working = WorkingMemory (
                cache       = cache,
                session_id  = session_id,
                default_ttl = float ( configuration.memory.working_ttl ),
            ),
            episodic = EpisodicMemory ( connection ),
            semantic = SemanticMemory (
                connection = connection,
                model_name = configuration.database.embedding_model,
                dimensions = configuration.database.vector_dimensions,
            ),
            embedder = model,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: assert_ready
    #
    # Description:
    #
    #   Run the semantic tier's startup guards.
    #
    #   Called during startup, before the first turn. Loading the embedding model here is the point: the width guard is
    #   worthless unless it compares against what the model actually produces.
    #
    # Arguments:
    #
    #   check_model : Load the embedding model to obtain its true output width. Passing False checks the schema against
    #                 configuration only, which is what a caller does when it wants the cheap half of the check.
    #
    # Returns:
    #
    #   None.
    #
    #   Raises VectorDimensionError or EmbeddingModelMismatchError when a guard fails.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def assert_ready ( self, check_model: bool = True ) -> None:

        await self.semantic.assert_ready (
            model_dimensions = self.embedder.dimensions if check_model else None
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Writes
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: store_episode
    #
    # Description:
    #
    #   Record an episode, and by default its embedding alongside it.
    #
    #   Both tiers get the same identifier. That is what lets retrieval recognise the episodic and semantic views of
    #   one memory as the same memory and score it once, rather than presenting the model with a duplicate.
    #
    # Arguments:
    #
    #   summary    : What happened.
    #   category   : Episode kind.
    #   tags       : Free-form labels.
    #   session_id : Session this belongs to.
    #   moment     : The instant the episode describes. Now when omitted.
    #   embed      : Also store a semantic vector. False for episodes with no retrievable meaning, such as a heartbeat
    #                tick that did nothing.
    #
    # Returns:
    #
    #   The shared identifier.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def store_episode ( self,
                              summary: str,
                              category: str             = CATEGORY_CONVERSATION,
                              tags: list [ str ] | None = None,
                              session_id: str | None    = None,
                              moment: datetime | None   = None,
                              embed: bool = True ) -> str:

        identifier = await self.episodic.store (
            summary    = summary,
            category   = category,
            tags       = tags,
            session_id = session_id if session_id is not None else self.working.session_id,
            moment     = moment,
        )

        if embed:
            await self.store_fact (
                content   = summary,
                category  = category,
                moment    = moment,
                record_id = identifier,
            )

        # Return data to caller.

        return identifier

    #-------------------------------------------------------------------------------------------------------------------
    # Function: store_fact
    #
    # Description:
    #
    #   Embed and store a piece of text in semantic memory.
    #
    # Arguments:
    #
    #   content   : The text to remember.
    #   category  : Memory type.
    #   moment    : Creation time. Now when omitted.
    #   record_id : Explicit identifier. A fresh UUID when omitted.
    #
    # Returns:
    #
    #   The record's identifier.
    #
    #   Raises EmbeddingError if the text could not be embedded.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def store_fact ( self,
                           content: str,
                           category: str           = CATEGORY_CONVERSATION,
                           moment: datetime | None = None,
                           record_id: str | None = None ) -> str:

        vector = embed_one ( self.embedder, content )

        # Return data to caller.

        return await self.semantic.store (
            vector    = vector,
            content   = content,
            category  = category,
            moment    = moment,
            record_id = record_id,
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Retrieval
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: retrieve_relevant
    #
    # Description:
    #
    #   Retrieve the memories most worth putting in front of the model for this input.
    #
    # Arguments:
    #
    #   text           : The triggering text, embedded to form the query.
    #   moment         : The instant to measure recency from. Now when omitted.
    #   limit          : Maximum memories to return. memory.retrieval_limit when omitted.
    #   threshold      : Minimum cosine for a semantic candidate. memory.relevance_threshold when omitted.
    #   recency_weight : Share of the score given to recency. memory.recency_weight when omitted.
    #   half_life_days : Recency half-life. memory.recency_half_life when omitted.
    #
    # Returns:
    #
    #   Memories ordered by blended score, highest first, at most `limit` of them. Empty when nothing is stored, when
    #   nothing clears the threshold, or when the query text is blank.
    #
    #   Raises EmbeddingError if the query could not be embedded, or EmbeddingModelMismatchError if the semantic tier
    #   is blocked.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def retrieve_relevant ( self,
                                  text: str,
                                  moment: datetime | None      = None,
                                  limit: int | None            = None,
                                  threshold: float | None      = None,
                                  recency_weight: float | None = None,
                                  half_life_days: float | None = None ) -> list [ RetrievedMemory ]:

        settings = self.configuration.memory

        top_k          = limit if limit is not None else settings.retrieval_limit
        floor          = threshold if threshold is not None else settings.relevance_threshold
        weight         = recency_weight if recency_weight is not None else settings.recency_weight
        half_life      = half_life_days if half_life_days is not None else settings.recency_half_life
        half_life_secs = half_life * SECONDS_PER_DAY
        now            = moment if moment is not None else datetime.now ( UTC )

        # Nothing to embed means nothing to compare against. Returning early keeps a blank heartbeat tick from paying
        # for a model load and a KNN scan.

        if not text.strip () or top_k <= 0:
            return []

        query_vector = embed_one ( self.embedder, text )
        widened      = max ( top_k * CANDIDATE_WIDENING, top_k )

        # Arm 1: meaning. Theta is applied here, as a relevance floor on the cosine.

        matches = await self.semantic.search ( vector = query_vector, limit = widened, threshold = floor )

        # Arm 2: recency. Not gated by theta -- see the module banner.

        episodes = await self.episodic.recent ( limit = widened, before = None )

        candidates = self.merge_candidates (
            query_vector      = query_vector,
            matches           = matches,
            episodes          = episodes,
            now               = now,
            recency_weight    = weight,
            half_life_seconds = half_life_secs,
            episode_vectors = await self.semantic.vectors_for (
                [ episode.id for episode in episodes ]
            ),
        )

        candidates.sort ( key = lambda memory: memory.score, reverse = True )

        logger.debug (
            "Retrieved %d candidate(s) from %d semantic and %d episodic; returning %d.",
            len ( candidates ), len ( matches ), len ( episodes ), min ( top_k, len ( candidates ) ),
        )

        # Return data to caller.

        return candidates [ : top_k ]

    #-------------------------------------------------------------------------------------------------------------------
    # Function: merge_candidates
    #
    # Description:
    #
    #   Score and de-duplicate the two arms into one candidate list.
    #
    #   The semantic arm is walked first so that a record present in both keeps its exact KNN similarity rather than
    #   the value recomputed from the stored float32 vector. They agree to several decimal places; preferring the
    #   extension's own number keeps the ranking reproducible.
    #
    # Arguments:
    #
    #   query_vector      : The query embedding.
    #   matches           : Semantic hits.
    #   episodes          : Recent episodes.
    #   now               : The instant to measure recency from.
    #   recency_weight    : Share of the score given to recency.
    #   half_life_seconds : Recency half-life, in seconds.
    #   episode_vectors   : Stored vectors for the episodes, where they exist.
    #
    # Returns:
    #
    #   One RetrievedMemory per distinct identifier, unsorted.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def merge_candidates ( self,
                           query_vector: list [ float ],
                           matches: list [ SemanticMatch ],
                           episodes: list [ Episode ],
                           now: datetime,
                           recency_weight: float,
                           half_life_seconds: float,
                           episode_vectors: dict [ str, list [ float ] ] ) -> list [ RetrievedMemory ]:

        merged: dict [ str, RetrievedMemory ] = {}

        for match in matches:

            recency = self.recency_of ( match.timestamp, now, half_life_seconds )

            merged [ match.id ] = RetrievedMemory (
                id         = match.id,
                summary    = match.content,
                category   = match.category,
                timestamp  = match.timestamp,
                source     = SOURCE_SEMANTIC,
                similarity = match.similarity,
                recency    = recency,
                score      = blend_score ( match.similarity, recency, recency_weight ),
            )

        for episode in episodes:

            # Already surfaced by the meaning arm, and with a better similarity number.

            if episode.id in merged:
                continue

            vector     = episode_vectors.get ( episode.id )
            similarity = cosine_similarity ( query_vector, vector ) if vector else 0.0
            recency    = self.recency_of ( episode.timestamp, now, half_life_seconds )

            merged [ episode.id ] = RetrievedMemory (
                id         = episode.id,
                summary    = episode.summary,
                category   = episode.category,
                timestamp  = episode.timestamp,
                source     = SOURCE_EPISODIC,
                similarity = similarity,
                recency    = recency,
                score      = blend_score ( similarity, recency, recency_weight ),
            )

        # Return data to caller.

        return list ( merged.values () )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: recency_of
    #
    # Description:
    #
    #   Recency decay of a stored timestamp.
    #
    # Arguments:
    #
    #   timestamp         : The stored ISO 8601 instant.
    #   now               : The instant to measure from.
    #   half_life_seconds : Recency half-life, in seconds.
    #
    # Returns:
    #
    #   The decay value. Zero for a timestamp that cannot be parsed, which ranks an undated memory as infinitely old
    #   rather than as brand new -- the safe direction when the alternative is promoting corrupt rows to the top.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def recency_of ( self, timestamp: str, now: datetime, half_life_seconds: float ) -> float:

        instant = from_iso_timestamp ( timestamp )

        if instant is None:
            return 0.0

        # Return data to caller.

        return recency_decay ( ( now - instant ).total_seconds (), half_life_seconds )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: retrieve_for_prompt
    #
    # Description:
    #
    #   Retrieve memories already shaped for prompt assembly.
    #
    #   The bridge to core/prompt.py's section 5. A retrieval failure returns nothing rather than propagating: memory
    #   is an enrichment of the turn, and an agent that refuses to answer because its recall is degraded is worse than
    #   one that answers without it. The failure is logged, and the semantic tier's own guards still block silently
    #   wrong results.
    #
    # Arguments:
    #
    #   text   : The triggering text.
    #   moment : The instant to measure recency from. Now when omitted.
    #
    # Returns:
    #
    #   Entries for build_memory_section, or an empty list.
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def retrieve_for_prompt ( self,
                                    text: str,
                                    moment: datetime | None = None ) -> list [ dict [ str, Any ] ]:

        try:
            memories = await self.retrieve_relevant ( text, moment = moment )
        except MemorySystemError as error:
            logger.warning ( "Memory retrieval is unavailable for this turn: %s", error )

            return []

        # Return data to caller.

        return [ memory.as_prompt_entry () for memory in memories ]

    #-------------------------------------------------------------------------------------------------------------------
    # Housekeeping
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: prune
    #
    # Description:
    #
    #   Apply the retention policy to episodic memory and drop expired working-memory entries.
    #
    #   Semantic memory is not pruned by age. A fact the user stated two years ago is not less true, and the tier is
    #   sized by vector count rather than by time.
    #
    # Arguments:
    #
    #   moment : The instant to measure from. Now when omitted.
    #
    # Returns:
    #
    #   A report with keys "episodes_pruned" and "working_expired".
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def prune ( self, moment: datetime | None = None ) -> dict [ str, int ]:

        pruned  = await self.episodic.prune ( self.configuration.memory.episodic_retention, moment )
        expired = self.working.expire ()

        # Return data to caller.

        return { "episodes_pruned": pruned, "working_expired": expired }

    #-------------------------------------------------------------------------------------------------------------------
    # Function: describe
    #
    # Description:
    #
    #   A summary of what the memory system currently holds.
    #
    #   What `sentinel init` and any future status command report, and the cheapest way to confirm that memory survived
    #   a restart.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   A report with keys "episodes", "vectors", "working_bytes", "embedding_model", "vector_dimensions", and
    #   "blocked".
    #
    #-------------------------------------------------------------------------------------------------------------------

    async def describe ( self ) -> dict [ str, str ]:

        # Return data to caller.

        return {
            "episodes": str ( await self.episodic.count () ),
            "vectors": str ( await self.semantic.count () ),
            "working_bytes": str ( self.working.volume () ),
            "embedding_model": self.semantic.model_name,
            "vector_dimensions": str ( self.semantic.dimensions ),
            "blocked": self.semantic.blocked_reason or "",
        }


#-----------------------------------------------------------------------------------------------------------------------
# Function: open_memory_system
#
# Description:
#
#   Open every resource the memory system needs and assemble it.
#
#   The convenience path for a caller that has a configuration and nothing else -- a script, a test, or a future
#   `sentinel memory` command. The long-lived agent does not use it: the orchestrator already holds the shared
#   connection and cache and passes them to MemorySystem.build.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   session_id    : Session to scope working memory to.
#   embedder      : The embedding model. The configured local one when omitted.
#
# Returns:
#
#   The assembled MemorySystem. The caller owns closing the connection and the cache, reachable as
#   system.episodic.connection and system.working.cache.
#
#-----------------------------------------------------------------------------------------------------------------------

async def open_memory_system ( configuration: SentinelConfig,
                               session_id: str = DEFAULT_SESSION,
                               embedder: Embedder | None = None ) -> MemorySystem:

    from sentinel.cache    import open_cache
    from sentinel.database import connect

    connection = await connect (
        database_path      = configuration.database_path,
        wal_mode           = configuration.database.wal_mode,
        busy_timeout       = configuration.database.busy_timeout,
        journal_size_limit = configuration.database.journal_size_limit,
        load_vectors       = True,
    )

    cache = open_cache ( configuration.cache_directory, configuration.memory.working_size_limit )

    # Return data to caller.

    return MemorySystem.build (
        configuration = configuration,
        connection    = connection,
        cache         = cache,
        session_id    = session_id,
        embedder      = embedder,
    )


#-----------------------------------------------------------------------------------------------------------------------
# Re-exported for callers that want to stamp their own timestamps in the stored format.
#-----------------------------------------------------------------------------------------------------------------------

__all__ = [
    "CANDIDATE_WIDENING",
    "SOURCE_EPISODIC",
    "SOURCE_SEMANTIC",
    "MemorySystem",
    "RetrievedMemory",
    "blend_score",
    "open_memory_system",
    "recency_decay",
    "to_iso_timestamp",
]
