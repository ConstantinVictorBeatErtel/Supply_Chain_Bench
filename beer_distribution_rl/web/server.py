"""FastAPI app: static UI, WebSocket stream, playable game REST API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from beer_distribution_rl.agents.ippo.policy_agent import PolicyLoadError
from beer_distribution_rl.web.runner import GameError, GameRunner

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StartRequest(BaseModel):
    role: str
    ai_mode: Literal["sterman", "ippo"] = "sterman"
    seed: int | None = None


class OrderRequest(BaseModel):
    quantity: int = Field(..., ge=0, le=10_000)


class ControlRequest(BaseModel):
    action: Literal["reset"]
    seed: int | None = None


class ConnectionManager:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)

    def broadcast_threadsafe(self, message: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)


def create_app(runner: GameRunner | None = None) -> FastAPI:
    manager = ConnectionManager()
    episode = runner or GameRunner()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.bind_loop(asyncio.get_running_loop())
        episode.add_listener(manager.broadcast_threadsafe)
        app.state.runner = episode
        app.state.manager = manager
        yield
        episode.remove_listener(manager.broadcast_threadsafe)
        episode.shutdown()

    app = FastAPI(title="Beer Distribution Game", lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return episode.snapshot()

    @app.post("/api/start")
    async def start_game(body: StartRequest) -> dict[str, Any]:
        try:
            return episode.start(body.role, body.ai_mode, seed=body.seed)
        except PolicyLoadError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except GameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/order")
    async def place_order(body: OrderRequest) -> dict[str, Any]:
        try:
            return episode.submit_order(body.quantity)
        except GameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/control")
    async def control(body: ControlRequest) -> dict[str, Any]:
        if body.action == "reset":
            return episode.reset(seed=body.seed)
        raise HTTPException(status_code=400, detail="Unsupported action")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            await ws.send_json(episode.snapshot())
            while True:
                data = await ws.receive_json()
                if isinstance(data, dict) and data.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await manager.disconnect(ws)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
