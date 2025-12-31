"""
Librarian Agent - Owns the read path for memories.

Responsibilities:
- Handle recall requests with composite ranking
- Process feedback to update usefulness scores
- Create links between related memories
- Manage concept clusters
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import UUID

from sqlalchemy import select, update, text, func, and_, bindparam
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.config import settings
from src.shared.database import (
    Memory,
    MemoryLink,
    MemoryEvent,
    MemoryTier,
    EventType,
    RelationshipType,
    Concept,
    ConceptMember,
    async_session,
)
from src.shared.embeddings import get_embedding, EmbeddingError
from src.shared.agent_mail import AgentMailClient, Message


logger = logging.getLogger(__name__)


class LibrarianAgent:
    """Agent responsible for memory retrieval and relationship management."""

    def __init__(self):
        self.mail = AgentMailClient("librarian")

    async def start(self) -> None:
        """Start the Librarian agent."""
        await self.mail.register()

        # Register message handlers
        self.mail.register_handler("recall", self._handle_recall)
        self.mail.register_handler("feedback", self._handle_feedback)
        self.mail.register_handler("memory_inserted", self._handle_memory_inserted)
        self.mail.register_handler("get_concepts", self._handle_get_concepts)

        await self.mail.start_polling()
        logger.info("Librarian agent started")

    async def stop(self) -> None:
        """Stop the Librarian agent."""
        await self.mail.stop_polling()
        logger.info("Librarian agent stopped")

    async def _handle_recall(self, message: Message) -> Dict[str, Any]:
        """
        Handle a recall request.

        Expected payload:
            query: str (required)
            project: str (optional)
            tags: list[str] (optional)
            limit: int (optional, default 10)

        Returns:
            memories: list of memory dicts with scores
        """
        payload = message.payload
        query = payload.get("query")

        if not query:
            return {"error": "No query provided", "memories": []}

        results = await self.recall(
            query=query,
            project=payload.get("project"),
            tags=payload.get("tags"),
            limit=payload.get("limit", 10),
        )

        return {"memories": results}

    async def recall(
        self,
        query: str,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using composite ranking.

        Ranking weights:
        - 40% vector similarity
        - 25% predicted quality
        - 25% proven usefulness
        - 10% recency

        Args:
            query: Search query
            project: Optional project filter
            tags: Optional tag filter
            limit: Max results to return

        Returns:
            List of memory dicts with scores
        """
        # Generate query embedding
        try:
            query_embedding = await get_embedding(query)
        except EmbeddingError as e:
            logger.error(f"Embedding error: {e}")
            return []

        async with async_session() as session:
            # Build the composite ranking query
            # Note: pgvector uses <=> for cosine distance, so similarity = 1 - distance
            # Use bindparam() for asyncpg compatibility (requires $1 positional params)
            sql = text("""
                WITH ranked AS (
                    SELECT
                        m.id,
                        m.content,
                        m.category,
                        m.tags,
                        m.source,
                        m.project,
                        m.extra_data,
                        m.predicted_quality,
                        m.usefulness_score,
                        m.access_count,
                        m.tier,
                        m.created_at,
                        1 - (m.embedding <=> CAST(:query_embedding AS vector)) as similarity,
                        EXTRACT(EPOCH FROM (NOW() - m.created_at)) / 86400.0 as days_old
                    FROM memories m
                    WHERE m.tier != 'quarantine'
                        AND m.predicted_quality >= :min_quality
                        AND m.embedding IS NOT NULL
                        AND (CAST(:project AS TEXT) IS NULL OR m.project = :project)
                        AND (CAST(:tags AS text[]) IS NULL OR m.tags && CAST(:tags AS text[]))
                )
                SELECT
                    *,
                    (
                        :sim_weight * similarity +
                        :qual_weight * predicted_quality +
                        :use_weight * usefulness_score +
                        :rec_weight * (1.0 / (1.0 + days_old / 30.0))
                    ) as composite_score
                FROM ranked
                ORDER BY composite_score DESC
                LIMIT :limit
            """).bindparams(
                bindparam("query_embedding", value=str(query_embedding)),
                bindparam("min_quality", value=settings.min_quality_threshold),
                bindparam("project", value=project),
                bindparam("tags", value=tags),
                bindparam("sim_weight", value=settings.similarity_weight),
                bindparam("qual_weight", value=settings.quality_weight),
                bindparam("use_weight", value=settings.usefulness_weight),
                bindparam("rec_weight", value=settings.recency_weight),
                bindparam("limit", value=limit),
            )

            result = await session.execute(sql)

            rows = result.fetchall()
            memories = []

            for row in rows:
                # Log retrieval event and increment access count
                await self._log_retrieval(session, row.id)

                memories.append({
                    "id": str(row.id),
                    "content": row.content,
                    "category": row.category,
                    "tags": row.tags,
                    "source": row.source,
                    "project": row.project,
                    "metadata": row.extra_data,
                    "predicted_quality": row.predicted_quality,
                    "usefulness_score": row.usefulness_score,
                    "similarity": row.similarity,
                    "composite_score": row.composite_score,
                    "tier": row.tier,
                    "created_at": row.created_at.isoformat(),
                })

            await session.commit()

            logger.info(f"Recalled {len(memories)} memories for query: {query[:50]}...")
            return memories

    async def _log_retrieval(self, session: AsyncSession, memory_id: UUID) -> None:
        """Log a retrieval event and increment access count."""
        # Create event
        event = MemoryEvent(
            memory_id=memory_id,
            event=EventType.RETRIEVAL,
        )
        session.add(event)

        # Increment access count
        await session.execute(
            update(Memory)
            .where(Memory.id == memory_id)
            .values(access_count=Memory.access_count + 1)
        )

    async def _handle_feedback(self, message: Message) -> Dict[str, Any]:
        """
        Handle feedback on a memory.

        Expected payload:
            memory_id: str (required)
            helpful: bool (required)
            context: str (optional)

        Returns:
            success: bool
        """
        payload = message.payload
        memory_id = payload.get("memory_id")
        helpful = payload.get("helpful")

        if not memory_id or helpful is None:
            return {"success": False, "error": "Missing memory_id or helpful flag"}

        try:
            memory_uuid = UUID(memory_id)
        except ValueError:
            return {"success": False, "error": "Invalid memory_id"}

        await self.feedback(
            memory_id=memory_uuid,
            helpful=helpful,
            context=payload.get("context"),
        )

        return {"success": True}

    async def feedback(
        self,
        memory_id: UUID,
        helpful: bool,
        context: Optional[str] = None,
    ) -> None:
        """
        Record feedback on a memory and update usefulness score.

        Args:
            memory_id: The memory ID
            helpful: Whether the memory was helpful
            context: Optional context about why
        """
        async with async_session() as session:
            # Get current memory
            result = await session.execute(
                select(Memory).where(Memory.id == memory_id)
            )
            memory = result.scalar_one_or_none()

            if not memory:
                logger.warning(f"Feedback for unknown memory: {memory_id}")
                return

            # Log the feedback event
            event = MemoryEvent(
                memory_id=memory_id,
                event=EventType.HELPFUL if helpful else EventType.NOT_HELPFUL,
                context={"reason": context} if context else {},
            )
            session.add(event)

            # Update usefulness score using exponential moving average
            alpha = 0.3  # Learning rate
            new_signal = 1.0 if helpful else 0.0
            new_score = (1 - alpha) * memory.usefulness_score + alpha * new_signal

            memory.usefulness_score = new_score

            await session.commit()

            logger.info(
                f"Feedback recorded for {memory_id}: helpful={helpful}, "
                f"new_usefulness={new_score:.3f}"
            )

    async def _handle_memory_inserted(self, message: Message) -> Dict[str, Any]:
        """
        Handle notification of a new memory insertion.

        Find and create links to related memories.
        """
        payload = message.payload
        memory_id = payload.get("memory_id")

        if not memory_id:
            return {"error": "No memory_id provided"}

        try:
            memory_uuid = UUID(memory_id)
        except ValueError:
            return {"error": "Invalid memory_id"}

        links_created = await self._find_and_link_related(memory_uuid)

        return {"links_created": links_created}

    async def _find_and_link_related(self, memory_id: UUID) -> int:
        """
        Find related memories and create links.

        Returns:
            Number of links created
        """
        async with async_session() as session:
            # Get the new memory
            result = await session.execute(
                select(Memory).where(Memory.id == memory_id)
            )
            memory = result.scalar_one_or_none()

            if not memory or not memory.embedding:
                return 0

            # Find similar memories (excluding self)
            # Use bindparam() for asyncpg compatibility (requires $1 positional params)
            sql = text("""
                SELECT
                    id,
                    content,
                    1 - (embedding <=> CAST(:embedding AS vector)) as similarity
                FROM memories
                WHERE id != :memory_id
                    AND embedding IS NOT NULL
                    AND tier != 'quarantine'
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT 5
            """).bindparams(
                bindparam("embedding", value=str(memory.embedding)),
                bindparam("memory_id", value=memory_id),
            )

            result = await session.execute(sql)

            rows = result.fetchall()
            links_created = 0

            for row in rows:
                # Only link if similarity is meaningful but not a duplicate
                if 0.5 <= row.similarity < settings.duplicate_similarity_threshold:
                    # Determine relationship type based on similarity
                    if row.similarity >= 0.8:
                        rel_type = RelationshipType.EXTENDS
                    elif row.similarity >= 0.6:
                        rel_type = RelationshipType.RELATED
                    else:
                        rel_type = RelationshipType.RELATED

                    # Create bidirectional links
                    link1 = MemoryLink(
                        source_id=memory_id,
                        target_id=row.id,
                        link_type=rel_type,
                        strength=row.similarity,
                    )
                    link2 = MemoryLink(
                        source_id=row.id,
                        target_id=memory_id,
                        link_type=rel_type,
                        strength=row.similarity,
                    )

                    session.add(link1)
                    session.add(link2)
                    links_created += 2

            await session.commit()

            if links_created > 0:
                logger.info(f"Created {links_created} links for memory {memory_id}")

            return links_created

    async def _handle_get_concepts(self, message: Message) -> Dict[str, Any]:
        """Get all concepts with their memory counts."""
        concepts = await self.get_concepts()
        return {"concepts": concepts}

    async def get_concepts(self) -> List[Dict[str, Any]]:
        """
        Get all emerged concepts.

        Returns:
            List of concept dicts
        """
        async with async_session() as session:
            result = await session.execute(
                select(Concept).order_by(Concept.memory_count.desc())
            )
            concepts = result.scalars().all()

            return [
                {
                    "id": str(c.id),
                    "name": c.name,
                    "description": c.description,
                    "memory_count": c.memory_count,
                    "created_at": c.created_at.isoformat(),
                }
                for c in concepts
            ]


async def main():
    """Run the Librarian agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    agent = LibrarianAgent()
    await agent.start()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
