from __future__ import annotations

import os
import uuid
from datetime import timedelta

from aiohttp import web
from dotenv import load_dotenv
from livekit.api import AccessToken, RoomAgentDispatch, RoomConfiguration, VideoGrants

load_dotenv(".env.local")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "priya")
HOST = os.getenv("TOKEN_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("TOKEN_SERVER_PORT", "8000"))

if not LIVEKIT_URL:
    raise RuntimeError("LIVEKIT_URL is missing")

if not LIVEKIT_API_KEY:
    raise RuntimeError("LIVEKIT_API_KEY is missing")

if not LIVEKIT_API_SECRET:
    raise RuntimeError("LIVEKIT_API_SECRET is missing")


def cors_headers() -> dict[str, str]:
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    }


async def health(_: web.Request) -> web.Response:
    return web.json_response(
        {
            "ok": True,
            "service": "hinglish-agent-token-server",
        },
        headers=cors_headers(),
    )


async def get_token(request: web.Request) -> web.Response:
    if request.method == "OPTIONS":
        return web.Response(
            status=204,
            headers=cors_headers(),
        )

    try:
        body = await request.json()
    except Exception:
        body = {}

    room_name = body.get("room_name") or f"bhashaflow-{uuid.uuid4().hex[:10]}"
    participant_identity = (
        body.get("participant_identity")
        or f"user-{uuid.uuid4().hex[:10]}"
    )
    participant_name = body.get(
        "participant_name",
        "BhashaFlow User",
    )

    token = (
        AccessToken(
            LIVEKIT_API_KEY,
            LIVEKIT_API_SECRET,
        )
        .with_identity(participant_identity)
        .with_name(participant_name)
        .with_ttl(timedelta(minutes=10))
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            RoomConfiguration(
                agents=[
                    RoomAgentDispatch(
                        agent_name=AGENT_NAME,
                        metadata='{"persona":"Priya","language":"hinglish"}',
                    )
                ]
            )
        )
    )

    return web.json_response(
        {
            "server_url": LIVEKIT_URL,
            "participant_token": token.to_jwt(),
            "room_name": room_name,
            "participant_identity": participant_identity,
        },
        status=201,
        headers=cors_headers(),
    )


app = web.Application()

app.router.add_get("/health", health)
app.router.add_get("/getToken", get_token)
app.router.add_post("/getToken", get_token)
app.router.add_options("/getToken", get_token)


if __name__ == "__main__":
    web.run_app(
        app,
        host=HOST,
        port=PORT,
    )
