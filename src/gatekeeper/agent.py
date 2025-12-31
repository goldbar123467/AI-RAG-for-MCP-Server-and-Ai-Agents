"""
Gatekeeper Agent - Owns the write path for memories.

Responsibilities:
- Receive remember requests from external agents
- Extract features and predict quality
- Check for duplicates
- Insert memories and generate embeddings
- Notify Librarian of new memories
"""

import asyncio
import logging
import pickle
from typing import Optional, Dict, Any, Tuple
from uuid import UUID

from sqlalchemy import select, text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.config import settings
from src.shared.database import (
    Memory,
    MemoryCategory,
    MemoryTier,
    TrainingState,
    Model,
    async_session,
)
from src.shared.embeddings import get_embedding, EmbeddingError
from src.shared.features import extract_features, heuristic_quality_score, Features
from src.shared.agent_mail import AgentMailClient, Message


logger = logging.getLogger(__name__)


class GatekeeperAgent:
    """Agent responsible for memory ingestion and quality gating."""

    def __init__(self):
        self.mail = AgentMailClient("gatekeeper")
        self._model: Optional[Any] = None
        self._model_version: int = 0

    async def start(self) -> None:
        """Start the Gatekeeper agent."""
        # Register with Agent Mail
        await self.mail.register()

        # Register message handlers
        self.mail.register_handler("remember", self._handle_remember)
        self.mail.register_handler("reload_model", self._handle_reload_model)

        # Load current model
        await self._load_active_model()

        # Start polling for messages
        await self.mail.start_polling()

        logger.info("Gatekeeper agent started")

    async def stop(self) -> None:
        """Stop the Gatekeeper agent."""
        await self.mail.stop_polling()
        logger.info("Gatekeeper agent stopped")

    async def _load_active_model(self) -> None:
        """Load the currently active XGBoost model from database."""
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.is_active == True)
            )
            model = result.scalar_one_or_none()

            if model and model.model_blob:
                try:
                    self._model = pickle.loads(model.model_blob)
                    self._model_version = model.version
                    logger.info(f"Loaded model version {model.version}")
                except Exception as e:
                    logger.error(f"Failed to load model: {e}")
                    self._model = None
            else:
                logger.info("No active model found, using heuristic scoring")
                self._model = None

    async def _handle_reload_model(self, message: Message) -> Dict[str, Any]:
        """Handle request to reload the model."""
        await self._load_active_model()
        return {"status": "reloaded", "version": self._model_version}

    async def _handle_remember(self, message: Message) -> Dict[str, Any]:
        """
        Handle a remember request.

        Expected payload:
            content: str (required)
            category: str (optional)
            tags: list[str] (optional)
            source: str (optional)
            project: str (optional)
            metadata: dict (optional)

        Returns:
            memory_id: str if accepted
            rejected: bool
            reason: str if rejected
        """
        payload = message.payload
        content = payload.get("content")

        if not content:
            return {"rejected": True, "reason": "No content provided"}

        # Extract basic info
        category = payload.get("category", "other")
        tags = payload.get("tags", [])
        source = payload.get("source", message.from_agent)
        project = payload.get("project")
        metadata = payload.get("metadata", {})

        # Process the memory
        result = await self.remember(
            content=content,
            category=category,
            tags=tags,
            source=source,
            project=project,
            metadata=metadata,
        )

        # Reply to the sender
        if result.get("rejected"):
            return result

        # Notify Librarian of new memory
        await self.mail.send(
            "librarian",
            "memory_inserted",
            {"memory_id": result["memory_id"]}
        )

        return result

    async def remember(
        self,
        content: str,
        category: str = "other",
        tags: Optional[list] = None,
        source: Optional[str] = None,
        project: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Process and store a new memory.

        Args:
            content: The memory content
            category: Memory category
            tags: Associated tags
            source: Source of the memory
            project: Associated project
            metadata: Additional metadata

        Returns:
            Result dict with memory_id or rejection reason
        """
        tags = tags or []
        metadata = metadata or {}

        # Generate embedding first (needed for duplicate check and features)
        try:
            embedding = await get_embedding(content)
        except EmbeddingError as e:
            logger.error(f"Embedding error: {e}")
            return {"rejected": True, "reason": f"Failed to generate embedding: {e}"}

        # Extract features
        features = extract_features(content, source, embedding)

        # Predict quality
        quality_score = self._predict_quality(features)

        # Check quality threshold
        if quality_score < settings.min_quality_threshold:
            logger.info(f"Rejected memory (quality={quality_score:.3f}): {content[:50]}...")
            return {
                "rejected": True,
                "reason": f"Quality score {quality_score:.3f} below threshold {settings.min_quality_threshold}",
                "quality_score": quality_score,
            }

        async with async_session() as session:
            # Check for duplicates
            is_duplicate, existing_id = await self._check_duplicate(session, embedding)
            if is_duplicate:
                logger.info(f"Duplicate detected: {existing_id}")
                return {
                    "rejected": True,
                    "reason": "Near-duplicate memory already exists",
                    "existing_id": str(existing_id),
                }

            # Determine tier
            tier = self._determine_tier(quality_score, source)

            # Parse category
            try:
                category_enum = MemoryCategory(category.lower())
            except ValueError:
                category_enum = MemoryCategory.OTHER

            # Create memory
            memory = Memory(
                content=content,
                embedding=embedding,
                category=category_enum,
                tags=tags,
                source=source,
                project=project,
                extra_data=metadata,
                predicted_quality=quality_score,
                tier=tier,
            )

            session.add(memory)
            await session.flush()

            # Increment training counter
            await self._increment_training_counter(session)

            await session.commit()

            memory_id = str(memory.id)
            logger.info(f"Stored memory {memory_id} (quality={quality_score:.3f}, tier={tier.value})")

            return {
                "rejected": False,
                "memory_id": memory_id,
                "quality_score": quality_score,
                "tier": tier.value,
            }

    def _predict_quality(self, features: Features) -> float:
        """Predict quality score using model or heuristic."""
        if self._model is not None:
            try:
                feature_array = features.to_array().reshape(1, -1)
                prediction = self._model.predict_proba(feature_array)[0][1]
                return float(prediction)
            except Exception as e:
                logger.error(f"Model prediction error: {e}")

        # Fall back to heuristic
        return heuristic_quality_score(features)

    async def _check_duplicate(
        self,
        session: AsyncSession,
        embedding: list,
    ) -> Tuple[bool, Optional[UUID]]:
        """
        Check if a near-duplicate memory exists.

        Returns:
            (is_duplicate, existing_memory_id)
        """
        # Use cosine similarity with pgvector
        # Use bindparam() for asyncpg compatibility (requires $1 positional params)
        query = text("""
            SELECT id, 1 - (embedding <=> CAST(:embedding AS vector)) as similarity
            FROM memories
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT 1
        """).bindparams(bindparam("embedding", value=str(embedding)))

        result = await session.execute(query)
        row = result.first()

        if row and row.similarity >= settings.duplicate_similarity_threshold:
            return True, row.id

        return False, None

    def _determine_tier(self, quality_score: float, source: Optional[str]) -> MemoryTier:
        """Determine initial tier based on quality and source."""
        # High quality from trusted source -> could start as core
        if quality_score >= 0.85 and source in ("human", "manual", "verified"):
            return MemoryTier.CORE

        # Normal quality -> active
        if quality_score >= settings.min_quality_threshold:
            return MemoryTier.ACTIVE

        # Low quality but above threshold -> still active but will be watched
        return MemoryTier.ACTIVE

    async def _increment_training_counter(self, session: AsyncSession) -> None:
        """Increment the training counter and check threshold."""
        result = await session.execute(select(TrainingState))
        state = result.scalar_one_or_none()

        if state:
            state.memories_since_last_train += 1
            count = state.memories_since_last_train
        else:
            state = TrainingState(memories_since_last_train=1)
            session.add(state)
            count = 1

        # Notify trainer if threshold reached
        if count >= settings.training_threshold:
            await self.mail.send(
                "trainer",
                "training_threshold_reached",
                {"count": count}
            )
            logger.info(f"Training threshold reached: {count} memories")


async def main():
    """Run the Gatekeeper agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    agent = GatekeeperAgent()
    await agent.start()

    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
