#-----------------------------------------------------------------------------------------------------------------------
# Module:  gateway.py
# Project: Sentinel
# Version: 0.1.0
# Date:    2025
# Author:  Rohin Gosling
# Note:    Import-only module; not executable directly.
#
# Description:
#
#   FastAPI gateway (architecture 4.2).
#
#   Endpoints:
#
#       GET  /health                 service status, version, uptime
#       GET  /v1/models              the single model Sentinel advertises
#       POST /v1/chat/completions    a turn, streaming or not
#       GET  /api/logs               the append-only event log, filtered
#       GET  /metrics                Prometheus counters, when enabled
#
#   Four cross-cutting concerns, in the order a request meets them:
#
#     1. Rate limiting. Fixed window, per client. A local agent cannot be flooded from the network, but a runaway client
#        loop will happily bill a real API, so the limit is on by default.
#     2. Authentication. Bearer token compared in constant time against the value in the OS keyring. Loopback binding is
#        not a trust boundary -- every local process, including any browser page, can reach 127.0.0.1.
#     3. Request logging, with the Authorization header never recorded.
#     4. Error handling, which returns structured JSON and never a traceback.
#
#   The 401 path is why auth exists at all rather than being deferred: Open WebUI is a separate process that must
#   authenticate, so the token has to exist before Phase 2 can talk to this gateway.
#-----------------------------------------------------------------------------------------------------------------------

import asyncio
import json
import logging
import secrets
import time
import uuid

from collections     import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing          import Any

from fastapi                 import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses       import JSONResponse, StreamingResponse
from pydantic                import ValidationError

from sentinel                   import __version__
from sentinel.api               import openai_compat
from sentinel.config            import SentinelConfig
from sentinel.core.heartbeat    import HeartbeatLoop
from sentinel.core.loop         import Trigger
from sentinel.core.orchestrator import AgentOrchestrator
from sentinel.errors            import SentinelError
from sentinel.llm.adapter       import LlmAdapter
from sentinel.logging.exporters import PrometheusExporter
from sentinel.logging.logger    import DEFAULT_QUERY_LIMIT, EventLogger
from sentinel.security.secrets  import SECRET_API_TOKEN, read_secret

logger = logging.getLogger ( __name__ )

# What create_application will accept for its heartbeat: the loop itself, or something that produces it later.

HeartbeatSource = HeartbeatLoop | Callable [ [], HeartbeatLoop | None ]

# The same arrangement for the event logger, and for the same reason: the runtime opens its database connection on the
# event loop that does not exist when the application is built.

EventSource = EventLogger | Callable [ [], EventLogger | None ]

# And for the orchestrator, which holds the event logger and therefore inherits the same constraint.

OrchestratorSource = AgentOrchestrator | Callable [ [], AgentOrchestrator | None ]

# Endpoints reachable without a token. /health must stay open: a readiness probe that
# needs a credential cannot report that the credential is the thing that is broken.

UNAUTHENTICATED_PATHS = frozenset ( { "/health", "/docs", "/openapi.json", "/redoc" } )

# Sentinel binds to loopback and is called by a local Open WebUI, so the browser origin
# is a localhost port. A wildcard would let any page a browser has open call the agent.

ALLOWED_ORIGIN_PATTERN = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

# Terminator the OpenAI streaming protocol expects after the final chunk.

SSE_TERMINATOR = "data: [DONE]\n\n"


#-----------------------------------------------------------------------------------------------------------------------
# Class: RateLimiter
#
# Description:
#
#   Fixed-window request limiter, keyed per client.
#
#   In-process and in-memory on purpose: Sentinel is a single desktop process, so a shared store would add a dependency
#   to solve a problem that does not exist here.
#
# Attributes:
#
#   requests : Requests permitted per window.
#   window   : Window length, in seconds.
#-----------------------------------------------------------------------------------------------------------------------

class RateLimiter:

    #-------------------------------------------------------------------------------------------------------------------
    # Function: __init__
    #
    # Description:
    #
    #   Construct the limiter.
    #
    # Arguments:
    #
    #   requests : Requests permitted per window.
    #   window   : Window length, in seconds.
    #
    # Returns:
    #
    #   None.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def __init__ ( self, requests: int, window: float ) -> None:

        # Record the limit and prepare per-client hit records.

        self.requests = requests
        self.window   = window

        self._hits: dict [ str, deque [ float ] ] = defaultdict ( deque )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: allow
    #
    # Description:
    #
    #   Record a request and report whether it is within the limit.
    #
    # Arguments:
    #
    #   client : Client identifier, normally the peer address.
    #   now    : Current monotonic time. Read from the clock when omitted.
    #
    # Returns:
    #
    #   True when the request is permitted, False when the client has exhausted its window. A rejected request is not
    #   recorded, so a client held at the limit is not permanently locked out by its own retries.
    #
    #-------------------------------------------------------------------------------------------------------------------

    def allow ( self, client: str, now: float | None = None ) -> bool:

        # Discard hits that have aged out of the window.

        moment = now if now is not None else time.monotonic ()
        hits   = self._hits [ client ]

        while hits and moment - hits [ 0 ] > self.window:
            hits.popleft ()

        if len ( hits ) >= self.requests:
            return False

        hits.append ( moment )

        # Return data to caller.

        return True


#-----------------------------------------------------------------------------------------------------------------------
# Application factory
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: create_application
#
# Description:
#
#   Build the FastAPI application.
#
#   A factory rather than a module-level app: tests construct one per case with their own configuration and a fake
#   adapter, and a module-level singleton would leak state between them.
#
# Arguments:
#
#   configuration : The loaded configuration.
#   adapter       : The LLM adapter the loop should call.
#   api_token     : Expected bearer token. Read from the keyring when omitted.
#   on_shutdown   : Called when an authenticated client asks the agent to stop. Omitting it leaves the endpoint
#                   unregistered rather than registering one that does nothing, so an install with no process manager
#                   returns 404 instead of pretending to have complied.
#   heartbeat     : The autonomous tick, for the pause, resume, and status endpoints. Omitting it leaves those three
#                   unregistered, on the same reasoning as on_shutdown.
#
#                   Accepts either the loop itself or a callable returning it, because the two callers need different
#                   things. A test has a loop in hand and passes it. `sentinel start` does not -- the loop's database
#                   connection must be opened on the runtime's event loop, which does not exist when the application
#                   is built -- so it passes a callable that resolves once the runtime is up.
#   events        : The event logger, for the log query and metrics endpoints. Accepts a logger or a callable
#                   returning one, for the same reason as the heartbeat. Omitting it leaves GET /api/logs unregistered.
#   orchestrator  : What interactive turns go through. A source, again for the same reason -- the runtime's
#                   orchestrator holds the event logger, which cannot exist before the event loop does. A source that
#                   answers None falls back to the stateless orchestrator built here, so an application serving before
#                   its runtime has finished starting still answers, just without an audit trail.
#
# Returns:
#
#   A configured FastAPI application.
#
#-----------------------------------------------------------------------------------------------------------------------

def create_application ( configuration: SentinelConfig,
                         adapter: LlmAdapter,
                         api_token: str | None                     = None,
                         on_shutdown: Callable [ [], None ] | None = None,
                         heartbeat: HeartbeatSource | None         = None,
                         events: EventSource | None                = None,
                         orchestrator: OrchestratorSource | None = None ) -> FastAPI:

    application = FastAPI (
        title       = "Sentinel",
        version     = __version__,
        description = "OpenAI-compatible API for the Sentinel agent.",
    )

    started_at = time.monotonic ()

    # Interactive turns go through the orchestrator, not straight to the loop. Until Phase 5 they went straight to it,
    # because only the orchestrator's completed-result form existed and this endpoint streams -- with the consequence
    # that interactive turns silently skipped the session handling autonomous ones got. process_stream() closes that,
    # and one path into the loop is worth more than the small indirection.

    fallback_orchestrator = AgentOrchestrator ( configuration = configuration, adapter = adapter )

    # Resolve the expected token once, at construction. Re-reading the keyring per request
    # would put a synchronous OS call on the hot path for a value that does not change.

    expected_token = api_token if api_token is not None else read_secret ( SECRET_API_TOKEN )

    if configuration.api.auth_required and not expected_token:
        logger.warning (
            "api.auth_required is true but no API token is configured. Every request will be "
            "refused. Run `sentinel key set api-token --generate` to create one."
        )

    limiter = RateLimiter (
        requests = configuration.api.rate_limit.requests,
        window   = float ( configuration.api.rate_limit.window ),
    )

    application.add_middleware (
        CORSMiddleware,
        allow_origin_regex = ALLOWED_ORIGIN_PATTERN,
        allow_credentials  = True,
        allow_methods      = [ "GET", "POST", "OPTIONS" ],
        allow_headers      = [ "Authorization", "Content-Type" ],
    )

    #-------------------------------------------------------------------------------------------------------------------
    # Middleware
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: guard_request
    #
    # Description:
    #
    #   Apply rate limiting, authentication, and request logging to every request.
    #
    # Arguments:
    #
    #   request      : The incoming request.
    #   call_next    : The next handler in the chain.
    #
    # Returns:
    #
    #   The handler's response, or a 429 or 401 when a gate refuses. The Authorization header is never logged.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @application.middleware ( "http" )
    async def guard_request ( request: Request,
                              call_next: Callable [ [ Request ], Awaitable [ Response ] ] ) -> Response:

        started = time.monotonic ()
        client  = request.client.host if request.client else "unknown"

        # Gate 1: rate limit, before auth so a flood cannot be used to probe tokens.

        if configuration.api.rate_limit.enabled and not limiter.allow ( client ):
            logger.warning ( "Rate limit exceeded for %s on %s.", client, request.url.path )

            return JSONResponse (
                status_code = 429,
                content = openai_compat.error_body (
                    "Too many requests. Slow down and retry.",
                    error_type = "rate_limit_error",
                    code       = "rate_limit_exceeded",
                ),
                headers = { "Retry-After": str ( configuration.api.rate_limit.window ) },
            )

        # Gate 2: authentication.

        if configuration.api.auth_required and request.url.path not in UNAUTHENTICATED_PATHS:

            if not is_authorised ( request, expected_token ):
                logger.warning ( "Unauthorised request from %s to %s.", client, request.url.path )

                return JSONResponse (
                    status_code = 401,
                    content = openai_compat.error_body (
                        "Invalid or missing API token.",
                        error_type = "authentication_error",
                        code       = "invalid_api_key",
                    ),
                    headers = { "WWW-Authenticate": "Bearer" },
                )

        # Gate 3: run the handler, then log the outcome without the credential.

        response = await call_next ( request )

        logger.info (
            "%s %s -> %d (%.0f ms)",
            request.method, request.url.path, response.status_code,
            ( time.monotonic () - started ) * 1000.0,
        )

        # Return data to caller.

        return response

    #-------------------------------------------------------------------------------------------------------------------
    # Error handling
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: handle_sentinel_error
    #
    # Description:
    #
    #   Convert a SentinelError into a structured 500 response.
    #
    # Arguments:
    #
    #   request : The request that failed.
    #   error   : The raised error.
    #
    # Returns:
    #
    #   A JSON error body. The message is the exception's own text, which the hierarchy is written to keep free of
    #   credentials; the traceback goes to the log, never to the client.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @application.exception_handler ( SentinelError )
    async def handle_sentinel_error ( request: Request, error: Exception ) -> JSONResponse:

        logger.exception ( "Request to %s failed: %s", request.url.path, error )

        # Return data to caller.

        return JSONResponse (
            status_code = 500,
            content     = openai_compat.error_body ( str ( error ), error_type = "server_error" ),
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Endpoints
    #-------------------------------------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------------------------------------
    # Function: health
    #
    # Description:
    #
    #   Report service status.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   Status, version, and uptime. Unauthenticated by design; see UNAUTHENTICATED_PATHS.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @application.get ( "/health" )
    async def health () -> dict [ str, Any ]:

        # Return data to caller.

        return {
            "status": "healthy",
            "version": __version__,
            "uptime": round ( time.monotonic () - started_at, 3 ),
            "model": configuration.llm.model,
        }

    #-------------------------------------------------------------------------------------------------------------------
    # Function: list_models
    #
    # Description:
    #
    #   List the models this API exposes.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   An OpenAI-shaped list carrying exactly one entry: Sentinel itself.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @application.get ( "/v1/models" )
    async def list_models () -> dict [ str, Any ]:

        # Return data to caller.

        return openai_compat.model_list ()

    #-------------------------------------------------------------------------------------------------------------------
    # Function: chat_completions
    #
    # Description:
    #
    #   Run one turn through the agentic loop.
    #
    # Arguments:
    #
    #   request : The incoming request, parsed by hand so a malformed body can be reported as 422 with an OpenAI-shaped
    #             body rather than FastAPI's own error shape.
    #
    # Returns:
    #
    #   A chat.completion object, or a stream of chat.completion.chunk events when the request asked for streaming.
    #
    #-------------------------------------------------------------------------------------------------------------------

    @application.post ( "/v1/chat/completions" )
    async def chat_completions ( request: Request ) -> Response:

        # Parse the body ourselves so both malformed JSON and schema violations produce
        # one consistent 422 shape.

        try:
            payload = await request.json ()
        except Exception:
            return JSONResponse (
                status_code = 422,
                content     = openai_compat.error_body ( "Request body is not valid JSON." ),
            )

        try:
            parsed = openai_compat.ChatCompletionRequest.model_validate ( payload )
        except ValidationError as error:
            return JSONResponse (
                status_code = 422,
                content = openai_compat.error_body (
                    f"Request does not match the chat-completions schema: {error.error_count ()} "
                    f"problem(s). First: {error.errors () [ 0 ] [ 'msg' ]}"
                ),
            )

        history, content = parsed.split_history ()

        trigger = Trigger ( kind = "UserMessage", content = content )

        completion_id = f"chatcmpl-{uuid.uuid4 ().hex}"

        turns = resolve_orchestrator ( orchestrator, fallback_orchestrator )

        # Streaming and non-streaming diverge only in presentation; the turn is identical.

        if parsed.stream:
            return StreamingResponse (
                stream_completion ( turns, trigger, history, completion_id ),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        result = await turns.process ( trigger, history = history )

        # Return data to caller.

        return JSONResponse (
            content = openai_compat.completion_response (
                completion_id = completion_id,
                text          = result.text,
                stop_reason   = result.stop_reason,
                prompt_tokens = result.usage.total_prompt_tokens,
                output_tokens = result.usage.output_tokens,
            )
        )

    #-------------------------------------------------------------------------------------------------------------------
    # Function: control_shutdown
    #
    # Description:
    #
    #   Stop the agent at an authenticated client's request.
    #
    #   This is how `sentinel stop` reaches a running instance. Two alternatives were rejected: a PID file plus a
    #   signal, which on Windows has no portable graceful equivalent and would skip the FSM's teardown entirely; and an
    #   unauthenticated endpoint, which would let any local page a browser has open stop the agent.
    #
    #   Authentication is the middleware's, not this handler's -- the path is absent from UNAUTHENTICATED_PATHS, so an
    #   unauthenticated caller is refused before routing.
    #
    # Arguments:
    #
    #   None.
    #
    # Returns:
    #
    #   An acknowledgement, sent before the teardown begins. The hook only requests the shutdown; it does not block, so
    #   this response reaches the client rather than dying with the server that was writing it.
    #
    #-------------------------------------------------------------------------------------------------------------------

    if on_shutdown is not None:

        @application.post ( "/control/shutdown" )
        async def control_shutdown () -> dict [ str, Any ]:

            logger.info ( "Shutdown requested through the control endpoint." )

            on_shutdown ()

            # Return data to caller.

            return { "status": "stopping" }

    #-------------------------------------------------------------------------------------------------------------------
    # Heartbeat control
    #
    #   Registered only when a heartbeat was supplied, so an install running without one answers 404 rather than
    #   pretending to have paused something that does not exist -- the same reasoning as the shutdown hook above.
    #
    #   Each handler resolves the source per request rather than closing over a loop captured at construction. When
    #   the source is a callable it may legitimately answer None -- the runtime has not finished starting, or the
    #   heartbeat is disabled -- and 503 is the honest reply to "pause something that is not there yet".
    #-------------------------------------------------------------------------------------------------------------------

    if heartbeat is not None:

        #---------------------------------------------------------------------------------------------------------------
        # Function: heartbeat_status
        #
        # Description:
        #
        #   Report what the autonomous tick is currently doing.
        #
        # Arguments:
        #
        #   None.
        #
        # Returns:
        #
        #   The heartbeat status.
        #
        #---------------------------------------------------------------------------------------------------------------

        @application.get ( "/api/heartbeat/status" )
        async def heartbeat_status () -> dict [ str, Any ]:

            # Return data to caller.

            return require_heartbeat ( heartbeat ).status ().as_dict ()

        #---------------------------------------------------------------------------------------------------------------
        # Function: heartbeat_pause
        #
        # Description:
        #
        #   Suspend the autonomous tick.
        #
        # Arguments:
        #
        #   None.
        #
        # Returns:
        #
        #   The status after pausing. Idempotent: pausing an already-paused heartbeat succeeds rather than erroring,
        #   because the caller's intent is a state, not a transition.
        #
        #---------------------------------------------------------------------------------------------------------------

        @application.post ( "/api/heartbeat/pause" )
        async def heartbeat_pause () -> dict [ str, Any ]:

            loop = require_heartbeat ( heartbeat )

            logger.info ( "Heartbeat paused through the control endpoint." )

            loop.pause ()

            # Return data to caller.

            return loop.status ().as_dict ()

        #---------------------------------------------------------------------------------------------------------------
        # Function: heartbeat_resume
        #
        # Description:
        #
        #   Resume the autonomous tick.
        #
        # Arguments:
        #
        #   None.
        #
        # Returns:
        #
        #   The status after resuming. Idempotent, for the same reason as pause.
        #
        #---------------------------------------------------------------------------------------------------------------

        @application.post ( "/api/heartbeat/resume" )
        async def heartbeat_resume () -> dict [ str, Any ]:

            loop = require_heartbeat ( heartbeat )

            logger.info ( "Heartbeat resumed through the control endpoint." )

            loop.resume ()

            # Return data to caller.

            return loop.status ().as_dict ()

    #-------------------------------------------------------------------------------------------------------------------
    # Event log
    #
    #   Registered only when an event logger was supplied, on the same reasoning as the two blocks above: an install
    #   running without one answers 404 rather than pretending to have an empty audit trail, which is a materially
    #   different claim.
    #-------------------------------------------------------------------------------------------------------------------

    if events is not None:

        #---------------------------------------------------------------------------------------------------------------
        # Function: read_logs
        #
        # Description:
        #
        #   Read the append-only event log, filtered.
        #
        # Arguments:
        #
        #   category       : Restrict to one of the eight categories.
        #   event          : Restrict to one event name.
        #   correlation_id : Restrict to one turn -- the query that reconstructs "what happened when I asked X".
        #   after          : ISO 8601 lower bound, exclusive.
        #   before         : ISO 8601 upper bound, exclusive.
        #   limit          : Maximum rows. Clamped to logging.query_limit.
        #   offset         : Rows to skip, for paging.
        #
        # Returns:
        #
        #   The matching events, newest first, with the total that matched the same filters. The total is what lets a
        #   caller tell "that is all of them" from "that is the first page".
        #
        #---------------------------------------------------------------------------------------------------------------

        @application.get ( "/api/logs" )
        async def read_logs ( category: str | None = None,
                              event: str | None          = None,
                              correlation_id: str | None = None,
                              after: str | None          = None,
                              before: str | None         = None,
                              limit: int                 = DEFAULT_QUERY_LIMIT,
                              offset: int = 0 ) -> dict [ str, Any ]:

            journal = require_events ( events )

            # Clamped rather than refused. A client asking for more than the ceiling wants as much as it can have, and
            # a 422 there would make paging code harder to write for no protection the clamp does not already give.

            bounded = max ( 1, min ( limit, configuration.logging.query_limit ) )

            found = await journal.query (
                category       = category,
                event          = event,
                correlation_id = correlation_id,
                after          = after,
                before         = before,
                limit          = bounded,
                offset         = max ( 0, offset ),
            )

            total = await journal.count (
                category       = category,
                event          = event,
                correlation_id = correlation_id,
                after          = after,
                before         = before,
            )

            # Return data to caller.

            return {
                "events": [ record.as_dict () for record in found ],
                "count": len ( found ),
                "total": total,
                "limit": bounded,
                "offset": max ( 0, offset ),
            }

        #---------------------------------------------------------------------------------------------------------------
        # Function: read_metrics
        #
        # Description:
        #
        #   Serve the event counters in the Prometheus text exposition format.
        #
        #   Registered only when logging.prometheus_enabled is set, because without it no PrometheusExporter is
        #   attached and the endpoint would have nothing but zeros to report. Authenticated like everything else: the
        #   counters say how much the agent has been used and when, which is not public information on a shared
        #   machine.
        #
        # Arguments:
        #
        #   None.
        #
        # Returns:
        #
        #   The exposition text, or 503 when the logger is not up yet.
        #
        #---------------------------------------------------------------------------------------------------------------

        if configuration.logging.prometheus_enabled:

            @application.get ( "/metrics" )
            async def read_metrics () -> Response:

                journal = require_events ( events )

                rendered = "".join (
                    exporter.render ()
                    for exporter in journal.exporters
                    if isinstance ( exporter, PrometheusExporter )
                )

                # Return data to caller.

                return Response ( content = rendered, media_type = "text/plain; version=0.0.4" )

    # Return data to caller.

    return application


#-----------------------------------------------------------------------------------------------------------------------
# Helpers
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
# Function: require_heartbeat
#
# Description:
#
#   Resolve a heartbeat source to the loop itself.
#
# Arguments:
#
#   source : The loop, or a callable returning it.
#
# Returns:
#
#   The heartbeat loop.
#
#   Raises HTTPException 503 when a callable source answers None. That happens while the runtime is still starting,
#   or when the heartbeat is disabled in the configuration -- neither is a client error, and both are conditions the
#   caller can sensibly retry, which 503 says and 404 does not.
#
#-----------------------------------------------------------------------------------------------------------------------

def require_heartbeat ( source: HeartbeatSource ) -> HeartbeatLoop:

    loop = source () if callable ( source ) else source

    if loop is None:
        raise HTTPException (
            status_code = 503,
            detail      = "The heartbeat is not running. It is disabled, or the agent is still starting.",
        )

    # Return data to caller.

    return loop


#-----------------------------------------------------------------------------------------------------------------------
# Function: require_events
#
# Description:
#
#   Resolve an event source to the logger itself.
#
# Arguments:
#
#   source : The logger, or a callable returning it.
#
# Returns:
#
#   The event logger.
#
#   Raises HTTPException 503 when a callable source answers None -- the runtime is still starting, or the log could not
#   be opened. Retryable in the first case and diagnosable in the second, which is what 503 says and 404 does not.
#
#-----------------------------------------------------------------------------------------------------------------------

def require_events ( source: EventSource ) -> EventLogger:

    journal = source () if callable ( source ) else source

    if journal is None:
        raise HTTPException (
            status_code = 503,
            detail      = "The event log is not available. The agent is still starting, or it could not be opened.",
        )

    # Return data to caller.

    return journal


#-----------------------------------------------------------------------------------------------------------------------
# Function: resolve_orchestrator
#
# Description:
#
#   Resolve an orchestrator source, falling back when it has nothing to give.
#
# Arguments:
#
#   source   : The orchestrator, a callable returning it, or None.
#   fallback : The stateless orchestrator to use when the source has none.
#
# Returns:
#
#   The orchestrator a turn should run through. Unlike the heartbeat and the event log, an unresolvable source here
#   falls back rather than answering 503: a turn can genuinely be run without a session store or an audit trail, and
#   refusing to answer the user because the log is not open yet would be the wrong way round.
#
#-----------------------------------------------------------------------------------------------------------------------

def resolve_orchestrator ( source: OrchestratorSource | None,
                           fallback: AgentOrchestrator ) -> AgentOrchestrator:

    if source is None:
        return fallback

    resolved = source () if callable ( source ) else source

    # Return data to caller.

    return resolved if resolved is not None else fallback


#-----------------------------------------------------------------------------------------------------------------------
# Function: is_authorised
#
# Description:
#
#   Check a request's bearer token.
#
# Arguments:
#
#   request        : The incoming request.
#   expected_token : The token the gateway expects, or None when none is configured.
#
# Returns:
#
#   True when the header carries the expected token. False when the header is absent, malformed, or wrong -- and also
#   when no token is configured at all, because an unconfigured gateway that accepts everything is worse than one that
#   refuses everything and says so in the log.
#
#   The comparison is constant-time. A local attacker can time this endpoint precisely, and a short-circuiting compare
#   leaks the token one byte at a time.
#
#-----------------------------------------------------------------------------------------------------------------------

def is_authorised ( request: Request, expected_token: str | None ) -> bool:

    # No configured token means nothing can match.

    if not expected_token:
        return False

    header = request.headers.get ( "Authorization", "" )

    scheme, _, presented = header.partition ( " " )

    if scheme.lower () != "bearer" or not presented:
        return False

    # Return data to caller.

    return secrets.compare_digest ( presented.strip (), expected_token )


#-----------------------------------------------------------------------------------------------------------------------
# Function: stream_completion
#
# Description:
#
#   Render a turn as an OpenAI-compatible server-sent event stream (architecture 3.3.8).
#
# Arguments:
#
#   turns         : The orchestrator.
#   trigger       : What started the turn.
#   history       : Prior conversation history, as the client sent it.
#   completion_id : Identifier carried by every chunk of this completion.
#
# Returns:
#
#   SSE frames: an opening chunk carrying the assistant role, a status frame per tool step, a chunk per text delta, a
#   terminal chunk carrying the finish reason, and finally [DONE].
#
#-----------------------------------------------------------------------------------------------------------------------

async def stream_completion ( turns: AgentOrchestrator,
                              trigger: Trigger,
                              history: list [ dict [ str, Any ] ],
                              completion_id: str ) -> AsyncIterator [ str ]:

    # Open with a role-only chunk, as the OpenAI streaming schema expects.

    yield frame ( openai_compat.completion_chunk ( completion_id, role = True ) )

    finish_reason = openai_compat.FINISH_STOP

    try:
        async for event in turns.process_stream ( trigger, history = history ):

            if event.type == "status":

                # Tool progress renders as a spinner caption, not as assistant text.

                yield frame ( openai_compat.status_event ( event.status, done = event.done ) )

            elif event.type == "text":
                yield frame ( openai_compat.completion_chunk ( completion_id, text = event.text ) )

            elif event.type == "result" and event.result is not None:
                finish_reason = openai_compat.finish_reason_for ( event.result.stop_reason )

    except asyncio.CancelledError:

        # The client disconnected. Stop cleanly rather than logging a failure.

        logger.info ( "Client disconnected mid-stream; abandoning the turn." )

        raise

    except SentinelError as error:

        # Report the failure inside the stream: the status line is already 200, so there
        # is no way to turn this into an HTTP error the client would notice.

        logger.exception ( "Streaming turn failed: %s", error )

        yield frame (
            openai_compat.completion_chunk (
                completion_id, text = f"\n\n[error] {error}"
            )
        )

    yield frame (
        openai_compat.completion_chunk ( completion_id, finish_reason = finish_reason )
    )

    yield SSE_TERMINATOR


#-----------------------------------------------------------------------------------------------------------------------
# Function: frame
#
# Description:
#
#   Render one object as a server-sent event frame.
#
# Arguments:
#
#   payload : The object to send.
#
# Returns:
#
#   The SSE frame, terminated by the blank line the protocol requires.
#
#-----------------------------------------------------------------------------------------------------------------------

def frame ( payload: dict [ str, Any ] ) -> str:

    # Return data to caller.

    return f"data: {json.dumps ( payload )}\n\n"
