"""Agent Mail client for inter-agent communication."""

import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable, List
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4
import json
import logging

import httpx

from .config import settings


logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message between agents."""
    id: UUID
    from_agent: str
    to_agent: str
    message_type: str
    payload: Dict[str, Any]
    created_at: datetime
    read: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            message_type=data["message_type"],
            payload=data.get("payload", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"],
            read=data.get("read", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "read": self.read,
        }


MessageHandler = Callable[[Message], Awaitable[Optional[Dict[str, Any]]]]


class AgentMailClient:
    """Client for Agent Mail inter-agent communication."""

    def __init__(
        self,
        agent_name: str,
        project: str = "workbench-rag",
        base_url: Optional[str] = None,
    ):
        """
        Initialize Agent Mail client.

        Args:
            agent_name: This agent's identifier (e.g., "gatekeeper", "librarian")
            project: Project namespace for message routing
            base_url: Agent Mail server URL
        """
        self.agent_name = agent_name
        self.project = project
        self.base_url = base_url or settings.agent_mail_url
        self._handlers: Dict[str, MessageHandler] = {}
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None

    def on_message(self, message_type: str) -> Callable[[MessageHandler], MessageHandler]:
        """
        Decorator to register a message handler.

        Usage:
            @client.on_message("remember")
            async def handle_remember(msg: Message) -> Optional[Dict]:
                ...
        """
        def decorator(handler: MessageHandler) -> MessageHandler:
            self._handlers[message_type] = handler
            return handler
        return decorator

    def register_handler(self, message_type: str, handler: MessageHandler) -> None:
        """Register a message handler programmatically."""
        self._handlers[message_type] = handler

    async def register(self) -> bool:
        """
        Register this agent with the Agent Mail server.

        Returns:
            True if registration successful
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/register",
                    json={
                        "agent_name": self.agent_name,
                        "project": self.project,
                    }
                )
                response.raise_for_status()
                logger.info(f"Registered agent '{self.agent_name}' with Agent Mail")
                return True
            except httpx.HTTPError as e:
                logger.error(f"Failed to register with Agent Mail: {e}")
                return False

    async def send(
        self,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
    ) -> Optional[UUID]:
        """
        Send a message to another agent.

        Args:
            to_agent: Recipient agent name
            message_type: Type of message
            payload: Message payload

        Returns:
            Message ID if sent successfully
        """
        message_id = uuid4()

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/send",
                    json={
                        "id": str(message_id),
                        "from_agent": self.agent_name,
                        "to_agent": to_agent,
                        "project": self.project,
                        "message_type": message_type,
                        "payload": payload,
                    }
                )
                response.raise_for_status()
                logger.debug(f"Sent {message_type} to {to_agent}")
                return message_id
            except httpx.HTTPError as e:
                logger.error(f"Failed to send message: {e}")
                return None

    async def broadcast(
        self,
        message_type: str,
        payload: Dict[str, Any],
        exclude: Optional[List[str]] = None,
    ) -> List[UUID]:
        """
        Broadcast a message to all agents in the project.

        Args:
            message_type: Type of message
            payload: Message payload
            exclude: Agent names to exclude

        Returns:
            List of message IDs
        """
        exclude = exclude or []

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/broadcast",
                    json={
                        "from_agent": self.agent_name,
                        "project": self.project,
                        "message_type": message_type,
                        "payload": payload,
                        "exclude": exclude,
                    }
                )
                response.raise_for_status()
                data = response.json()
                return [UUID(mid) for mid in data.get("message_ids", [])]
            except httpx.HTTPError as e:
                logger.error(f"Failed to broadcast message: {e}")
                return []

    async def fetch_messages(self, mark_read: bool = True) -> List[Message]:
        """
        Fetch unread messages for this agent.

        Args:
            mark_read: Whether to mark messages as read

        Returns:
            List of messages
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/messages",
                    params={
                        "agent_name": self.agent_name,
                        "project": self.project,
                        "mark_read": str(mark_read).lower(),
                    }
                )
                response.raise_for_status()
                data = response.json()
                return [Message.from_dict(m) for m in data.get("messages", [])]
            except httpx.HTTPError as e:
                logger.error(f"Failed to fetch messages: {e}")
                return []

    async def _process_messages(self) -> int:
        """
        Fetch and process pending messages.

        Returns:
            Number of messages processed
        """
        messages = await self.fetch_messages()
        processed = 0

        for message in messages:
            handler = self._handlers.get(message.message_type)
            if handler:
                try:
                    result = await handler(message)
                    processed += 1
                    logger.debug(f"Processed {message.message_type} from {message.from_agent}")
                except Exception as e:
                    logger.error(f"Error handling {message.message_type}: {e}")
            else:
                logger.warning(f"No handler for message type: {message.message_type}")

        return processed

    async def start_polling(self, interval: float = 1.0) -> None:
        """
        Start polling for messages in the background.

        Args:
            interval: Polling interval in seconds
        """
        self._running = True

        async def poll_loop():
            backoff = interval
            while self._running:
                try:
                    count = await self._process_messages()
                    if count > 0:
                        backoff = interval  # Reset backoff on activity
                    else:
                        backoff = min(backoff * 1.5, 30.0)  # Increase backoff up to 30s
                except Exception as e:
                    logger.error(f"Polling error: {e}")
                    backoff = min(backoff * 2, 60.0)

                await asyncio.sleep(backoff)

        self._poll_task = asyncio.create_task(poll_loop())
        logger.info(f"Started polling for messages (interval={interval}s)")

    async def stop_polling(self) -> None:
        """Stop the message polling loop."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped polling for messages")

    async def reply(
        self,
        original: Message,
        message_type: str,
        payload: Dict[str, Any],
    ) -> Optional[UUID]:
        """
        Reply to a message.

        Args:
            original: The message to reply to
            message_type: Response message type
            payload: Response payload

        Returns:
            Message ID if sent successfully
        """
        payload["in_reply_to"] = str(original.id)
        return await self.send(original.from_agent, message_type, payload)
