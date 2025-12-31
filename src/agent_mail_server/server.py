"""
Agent Mail Server - Simple message routing between agents.

In-memory storage - messages don't persist across restarts.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

from fastapi import FastAPI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Mail", version="1.0.0")

# In-memory storage
registered_agents: Dict[str, Set[str]] = defaultdict(set)  # project -> agent names
messages: Dict[str, List[dict]] = defaultdict(list)  # "project:agent" -> messages


class RegisterRequest(BaseModel):
    agent_name: str
    project: str


class SendRequest(BaseModel):
    id: Optional[str] = None
    from_agent: str
    to_agent: str
    project: str
    message_type: str
    payload: dict = Field(default_factory=dict)


class BroadcastRequest(BaseModel):
    from_agent: str
    project: str
    message_type: str
    payload: dict = Field(default_factory=dict)
    exclude: List[str] = Field(default_factory=list)


@app.post("/register")
async def register(request: RegisterRequest):
    """Register an agent."""
    registered_agents[request.project].add(request.agent_name)
    logger.info(f"Registered {request.agent_name} in {request.project}")
    return {"status": "registered"}


@app.post("/send")
async def send(request: SendRequest):
    """Send a message to a specific agent."""
    message_id = request.id or str(uuid4())
    key = f"{request.project}:{request.to_agent}"

    message = {
        "id": message_id,
        "from_agent": request.from_agent,
        "to_agent": request.to_agent,
        "message_type": request.message_type,
        "payload": request.payload,
        "created_at": datetime.utcnow().isoformat(),
        "read": False,
    }

    messages[key].append(message)
    logger.debug(f"Message {message_id}: {request.from_agent} -> {request.to_agent}")
    return {"status": "sent", "id": message_id}


@app.post("/broadcast")
async def broadcast(request: BroadcastRequest):
    """Broadcast a message to all agents in a project."""
    message_ids = []
    exclude = set(request.exclude)
    exclude.add(request.from_agent)  # Don't send to self

    for agent_name in registered_agents[request.project]:
        if agent_name in exclude:
            continue

        message_id = str(uuid4())
        key = f"{request.project}:{agent_name}"

        message = {
            "id": message_id,
            "from_agent": request.from_agent,
            "to_agent": agent_name,
            "message_type": request.message_type,
            "payload": request.payload,
            "created_at": datetime.utcnow().isoformat(),
            "read": False,
        }

        messages[key].append(message)
        message_ids.append(message_id)

    logger.debug(f"Broadcast from {request.from_agent}: {len(message_ids)} messages")
    return {"status": "broadcast", "message_ids": message_ids}


@app.get("/messages")
async def get_messages(
    agent_name: str,
    project: str,
    mark_read: str = "true",
):
    """Fetch messages for an agent."""
    key = f"{project}:{agent_name}"
    agent_messages = messages.get(key, [])

    # Filter unread
    unread = [m for m in agent_messages if not m["read"]]

    # Mark as read if requested
    if mark_read.lower() == "true":
        for m in unread:
            m["read"] = True

    return {"messages": unread}


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}


def main():
    """Run the server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    uvicorn.run(app, host="0.0.0.0", port=8765)


if __name__ == "__main__":
    main()
