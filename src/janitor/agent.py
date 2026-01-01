"""
Janitor Agent - Owns maintenance, rescoring, and tier management.

Responsibilities:
- Rescore all memories when new model is promoted
- Adjust memory tiers based on scores and usage
- Run daily cleanup of stale quarantined memories
- Broadcast maintenance summaries
"""

import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from sqlalchemy import select, update, func, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.config import settings
from src.shared.database import (
    Memory,
    MemoryTier,
    Model,
    async_session,
)
from src.shared.features import extract_features
from src.shared.agent_mail import AgentMailClient, Message


logger = logging.getLogger(__name__)


class JanitorAgent:
    """Agent responsible for memory maintenance and tier management."""

    def __init__(self):
        self.mail = AgentMailClient("janitor")
        self._model: Optional[Any] = None
        self._model_version: int = 0

    async def start(self) -> None:
        """Start the Janitor agent."""
        await self.mail.register()

        # Register message handlers
        self.mail.register_handler("model_promoted", self._handle_model_promoted)
        self.mail.register_handler("run_maintenance", self._handle_run_maintenance)

        await self.mail.start_polling()

        # Start daily maintenance task
        asyncio.create_task(self._daily_maintenance_loop())

        logger.info("Janitor agent started")

    async def stop(self) -> None:
        """Stop the Janitor agent."""
        await self.mail.stop_polling()
        logger.info("Janitor agent stopped")

    async def _daily_maintenance_loop(self) -> None:
        """Run daily maintenance tasks."""
        while True:
            try:
                # Wait until next day at 3 AM
                now = datetime.utcnow()
                next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)

                wait_seconds = (next_run - now).total_seconds()
                logger.info(f"Next maintenance in {wait_seconds / 3600:.1f} hours")

                await asyncio.sleep(wait_seconds)
                await self.run_maintenance()

            except Exception as e:
                logger.error(f"Daily maintenance error: {e}")
                await asyncio.sleep(3600)  # Wait an hour before retrying

    async def _handle_model_promoted(self, message: Message) -> Dict[str, Any]:
        """Handle notification of model promotion."""
        version = message.payload.get("version", 0)
        logger.info(f"Model promoted to version {version}, starting rescore...")

        # Load the new model
        await self._load_model(version)

        # Rescore all memories
        stats = await self.rescore_all()

        # Adjust tiers
        tier_stats = await self.adjust_tiers()

        # Broadcast completion
        await self.mail.broadcast(
            "maintenance_complete",
            {
                "trigger": "model_promotion",
                "version": version,
                "rescore_stats": stats,
                "tier_stats": tier_stats,
            },
            exclude=["janitor"],
        )

        return {"rescored": stats, "tiers": tier_stats}

    async def _handle_run_maintenance(self, message: Message) -> Dict[str, Any]:
        """Handle manual maintenance request."""
        logger.info("Manual maintenance requested")
        return await self.run_maintenance()

    async def _load_model(self, version: Optional[int] = None) -> None:
        """Load a specific or active model."""
        async with async_session() as session:
            if version:
                result = await session.execute(
                    select(Model).where(Model.version == version)
                )
            else:
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

    async def rescore_all(self, batch_size: int = 100) -> Dict[str, Any]:
        """
        Rescore all memories using the current model.

        Returns:
            Statistics about rescoring
        """
        if not self._model:
            await self._load_model()

        if not self._model:
            logger.warning("No model available for rescoring")
            return {"error": "No model available"}

        total = 0
        updated = 0
        errors = 0

        async with async_session() as session:
            # Count total memories
            result = await session.execute(select(func.count(Memory.id)))
            total = result.scalar() or 0

            # Process in batches
            offset = 0
            while offset < total:
                result = await session.execute(
                    select(Memory)
                    .where(Memory.embedding != None)
                    .offset(offset)
                    .limit(batch_size)
                )
                memories = result.scalars().all()

                if not memories:
                    break

                for memory in memories:
                    try:
                        # Extract features
                        features = extract_features(
                            memory.content,
                            memory.source,
                            memory.embedding,
                        )

                        # Predict new quality
                        feature_array = features.to_array().reshape(1, -1)
                        new_quality = float(
                            self._model.predict_proba(feature_array)[0][1]
                        )

                        # Update if significantly different
                        if abs(new_quality - memory.predicted_quality) > 0.01:
                            memory.predicted_quality = new_quality
                            updated += 1

                    except Exception as e:
                        logger.error(f"Error rescoring {memory.id}: {e}")
                        errors += 1

                await session.commit()
                offset += batch_size

                logger.info(f"Rescored {min(offset, total)}/{total} memories")

        logger.info(f"Rescoring complete: {updated}/{total} updated, {errors} errors")

        return {
            "total": total,
            "updated": updated,
            "errors": errors,
        }

    async def adjust_tiers(self) -> Dict[str, int]:
        """
        Adjust memory tiers based on quality, usefulness, and access patterns.

        Returns:
            Counts of tier changes
        """
        stats = {
            "promoted_to_core": 0,
            "demoted_from_core": 0,
            "quarantined": 0,
            "archived": 0,
            "rehabilitated": 0,
        }

        async with async_session() as session:
            # Promote to core: high quality + usefulness + access
            result = await session.execute(
                select(Memory).where(
                    and_(
                        Memory.tier != MemoryTier.CORE,
                        Memory.tier != MemoryTier.QUARANTINE,
                        Memory.predicted_quality >= settings.core_quality_threshold,
                        Memory.usefulness_score >= settings.core_usefulness_threshold,
                        Memory.access_count >= settings.core_access_threshold,
                    )
                )
            )
            for memory in result.scalars():
                memory.tier = MemoryTier.CORE
                stats["promoted_to_core"] += 1

            # Demote from core: quality or usefulness dropped
            result = await session.execute(
                select(Memory).where(
                    and_(
                        Memory.tier == MemoryTier.CORE,
                        ((Memory.predicted_quality < 0.5) | (Memory.usefulness_score < 0.4)),
                    )
                )
            )
            for memory in result.scalars():
                memory.tier = MemoryTier.ACTIVE
                stats["demoted_from_core"] += 1

            # Quarantine: low quality and usefulness
            result = await session.execute(
                select(Memory).where(
                    and_(
                        Memory.tier != MemoryTier.QUARANTINE,
                        Memory.predicted_quality < settings.quarantine_quality_threshold,
                        Memory.usefulness_score < settings.quarantine_usefulness_threshold,
                    )
                )
            )
            for memory in result.scalars():
                memory.tier = MemoryTier.QUARANTINE
                stats["quarantined"] += 1

            # Archive: active but not accessed in N days
            archive_cutoff = datetime.utcnow() - timedelta(days=settings.archive_days_inactive)
            result = await session.execute(
                select(Memory).where(
                    and_(
                        Memory.tier == MemoryTier.ACTIVE,
                        Memory.updated_at < archive_cutoff,
                        Memory.access_count < 2,
                    )
                )
            )
            for memory in result.scalars():
                memory.tier = MemoryTier.ARCHIVE
                stats["archived"] += 1

            # Rehabilitate: quarantined but now scoring better
            result = await session.execute(
                select(Memory).where(
                    and_(
                        Memory.tier == MemoryTier.QUARANTINE,
                        Memory.predicted_quality >= settings.min_quality_threshold + 0.1,
                    )
                )
            )
            for memory in result.scalars():
                memory.tier = MemoryTier.ACTIVE
                stats["rehabilitated"] += 1

            await session.commit()

        logger.info(f"Tier adjustments: {stats}")
        return stats

    async def run_maintenance(self) -> Dict[str, Any]:
        """
        Run full maintenance cycle.

        Returns:
            Combined statistics
        """
        logger.info("Starting maintenance cycle...")

        # Load current model
        await self._load_model()

        # Rescore if we have a model
        rescore_stats = {}
        if self._model:
            rescore_stats = await self.rescore_all()

        # Adjust tiers
        tier_stats = await self.adjust_tiers()

        # Cleanup stale quarantined memories
        cleanup_stats = await self._cleanup_stale()

        # Get overall statistics
        overall_stats = await self._get_stats()

        result = {
            "rescore": rescore_stats,
            "tiers": tier_stats,
            "cleanup": cleanup_stats,
            "overall": overall_stats,
            "completed_at": datetime.utcnow().isoformat(),
        }

        # Broadcast completion
        await self.mail.broadcast(
            "maintenance_complete",
            {
                "trigger": "scheduled",
                **result,
            },
            exclude=["janitor"],
        )

        logger.info("Maintenance cycle complete")
        return result

    async def _cleanup_stale(self) -> Dict[str, int]:
        """
        Flag stale quarantined memories for deletion.

        Returns:
            Cleanup statistics
        """
        stats = {"flagged_for_deletion": 0}

        async with async_session() as session:
            # Find memories quarantined for over 30 days with no improvement
            cutoff = datetime.utcnow() - timedelta(days=30)

            result = await session.execute(
                select(Memory).where(
                    and_(
                        Memory.tier == MemoryTier.QUARANTINE,
                        Memory.updated_at < cutoff,
                        Memory.predicted_quality < settings.min_quality_threshold,
                    )
                )
            )

            for memory in result.scalars():
                # Mark for deletion in extra_data (actual deletion requires approval)
                memory.extra_data = {
                    **memory.extra_data,
                    "flagged_for_deletion": True,
                    "flagged_at": datetime.utcnow().isoformat(),
                    "reason": "Quarantined for 30+ days with no improvement",
                }
                stats["flagged_for_deletion"] += 1

            await session.commit()

        if stats["flagged_for_deletion"] > 0:
            logger.info(f"Flagged {stats['flagged_for_deletion']} memories for deletion")

        return stats

    async def _get_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics."""
        async with async_session() as session:
            # Count by tier
            tier_counts = {}
            for tier in MemoryTier:
                result = await session.execute(
                    select(func.count(Memory.id)).where(Memory.tier == tier)
                )
                tier_counts[tier.value] = result.scalar() or 0

            # Average quality
            result = await session.execute(
                select(func.avg(Memory.predicted_quality))
            )
            avg_quality = result.scalar() or 0

            # Average usefulness
            result = await session.execute(
                select(func.avg(Memory.usefulness_score))
            )
            avg_usefulness = result.scalar() or 0

            # Total count
            total = sum(tier_counts.values())

            return {
                "total_memories": total,
                "tier_counts": tier_counts,
                "avg_quality": float(avg_quality),
                "avg_usefulness": float(avg_usefulness),
                "model_version": self._model_version,
            }


async def main():
    """Run the Janitor agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    agent = JanitorAgent()
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
