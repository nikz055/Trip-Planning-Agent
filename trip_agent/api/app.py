"""FastAPI routes. The browser talks only to this API, never to the database.

Every mutating route accepts an `Idempotency-Key` header; repeating a request
with the same key returns the first result (BR-16).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from trip_agent.config import Settings
from trip_agent.db.memory import MemoryStore
from trip_agent.llm.factory import make_llm
from trip_agent.orchestrator.orchestrator import NotFound, Orchestrator, TripError
from trip_agent.tools.mock import MockToolProvider

UI_DIR = Path(__file__).resolve().parent.parent / "ui"
Key = Header(default=None, alias="Idempotency-Key")


class NewTrip(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class FormAnswers(BaseModel):
    answers: dict[str, Any]


class Revision(BaseModel):
    instruction: str = Field(min_length=1, max_length=2000)


class Lock(BaseModel):
    item_id: str
    locked: bool = True


def share_view(view: dict[str, Any]) -> dict[str, Any]:
    """The read-only view behind a share link: the plan, nothing that can be acted on."""
    request = view["request"]
    return {
        "trip_id": view["trip_id"],
        "state": view["state"],
        "planner": view["planner"],
        "version": view["version"],
        "request": {k: request[k] for k in ("origin", "destination_city", "start_date", "end_date", "adults", "children", "pace")},
        "plan": view["plan"],
        "read_only": True,
    }


def create_app(orchestrator: Orchestrator | None = None, settings: Settings | None = None) -> FastAPI:
    """Pass an orchestrator to use it as is (tests); otherwise one is built at startup:
    Postgres when DATABASE_URL is set, in-memory otherwise."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = None
        if orchestrator is not None:
            app.state.orchestrator, app.state.storage = orchestrator, "provided"
        else:
            chosen = settings or Settings.from_env()
            try:
                from trip_agent.db.postgres import PostgresStore

                store = await PostgresStore.open()
                app.state.storage = "postgres"
            except RuntimeError:  # DATABASE_URL not set
                store, app.state.storage = MemoryStore(), "memory (nothing is kept after a restart)"
            app.state.orchestrator = Orchestrator(store, make_llm(chosen), MockToolProvider(), chosen)
        yield
        if store is not None and hasattr(store, "close"):
            await store.close()

    app = FastAPI(title="Trip-planning agent", lifespan=lifespan)

    @app.exception_handler(TripError)
    async def trip_error(_: Request, exc: TripError) -> JSONResponse:
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    def orch(request: Request) -> Orchestrator:
        return request.app.state.orchestrator

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "planner": orch(request).llm.name, "storage": request.app.state.storage}

    @app.post("/api/trips")
    async def create_trip(body: NewTrip, request: Request, key: str | None = Key) -> dict[str, Any]:
        return await orch(request).create_trip(body.text, key)

    @app.get("/api/trips/{trip_id}")
    async def get_trip(trip_id: str, request: Request) -> dict[str, Any]:
        return await orch(request).get(trip_id)

    @app.post("/api/trips/{trip_id}/form")
    async def submit_form(trip_id: str, body: FormAnswers, request: Request, key: str | None = Key) -> dict[str, Any]:
        return await orch(request).submit_form(trip_id, body.answers, key)

    @app.post("/api/trips/{trip_id}/revise")
    async def revise(trip_id: str, body: Revision, request: Request, key: str | None = Key) -> dict[str, Any]:
        return await orch(request).revise(trip_id, body.instruction, key)

    @app.post("/api/trips/{trip_id}/lock")
    async def lock(trip_id: str, body: Lock, request: Request, key: str | None = Key) -> dict[str, Any]:
        return await orch(request).set_lock(trip_id, body.item_id, body.locked, key)

    @app.post("/api/trips/{trip_id}/accept")
    async def accept(trip_id: str, request: Request, key: str | None = Key) -> dict[str, Any]:
        return await orch(request).accept(trip_id, key)

    @app.post("/api/trips/{trip_id}/cancel")
    async def cancel(trip_id: str, request: Request, key: str | None = Key) -> dict[str, Any]:
        return await orch(request).cancel(trip_id, key)

    @app.get("/api/share/{trip_id}")
    async def share(trip_id: str, request: Request) -> dict[str, Any]:
        """Read-only: no resume, no re-fetch, nothing is written."""
        o = orch(request)
        session = await o.store.load(trip_id)
        if session is None or session.plan is None:
            raise NotFound("There is no plan to share at this link.")
        return share_view(o.view(session))

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/share/{trip_id}")
    async def share_page(trip_id: str) -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    return app
