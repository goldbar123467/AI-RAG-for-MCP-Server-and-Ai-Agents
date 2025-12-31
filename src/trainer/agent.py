"""
Trainer Agent - Owns the machine learning lifecycle.

Responsibilities:
- Watch training_state for threshold triggers
- Pull labeled memories and extract features
- Train new XGBoost models
- Evaluate and promote improvements
- Notify Janitor to rescore on promotion
"""

import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

from sqlalchemy import select, update, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.shared.config import settings
from src.shared.database import (
    Memory,
    MemoryEvent,
    TrainingState,
    Model,
    EventType,
    async_session,
)
from src.shared.features import extract_features, Features
from src.shared.agent_mail import AgentMailClient, Message


logger = logging.getLogger(__name__)


class TrainerAgent:
    """Agent responsible for training and evaluating quality models."""

    def __init__(self):
        self.mail = AgentMailClient("trainer")
        self._training_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the Trainer agent."""
        await self.mail.register()

        # Register message handlers
        self.mail.register_handler("training_threshold_reached", self._handle_threshold)
        self.mail.register_handler("force_train", self._handle_force_train)

        await self.mail.start_polling()

        # Start background task to periodically check training state
        asyncio.create_task(self._training_check_loop())

        logger.info("Trainer agent started")

    async def stop(self) -> None:
        """Stop the Trainer agent."""
        await self.mail.stop_polling()
        logger.info("Trainer agent stopped")

    async def _training_check_loop(self) -> None:
        """Periodically check if training is needed."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._check_and_train()
            except Exception as e:
                logger.error(f"Training check error: {e}")

    async def _check_and_train(self) -> None:
        """Check training state and trigger training if needed."""
        async with async_session() as session:
            result = await session.execute(select(TrainingState))
            state = result.scalar_one_or_none()

            if state and state.memories_since_last_train >= settings.training_threshold:
                await self.train()

    async def _handle_threshold(self, message: Message) -> Dict[str, Any]:
        """Handle training threshold notification."""
        count = message.payload.get("count", 0)
        logger.info(f"Training threshold reached: {count} memories")
        result = await self.train()
        return result

    async def _handle_force_train(self, message: Message) -> Dict[str, Any]:
        """Handle force training request."""
        logger.info("Force training requested")
        result = await self.train()
        return result

    async def train(self) -> Dict[str, Any]:
        """
        Train a new quality model.

        Returns:
            Training result with metrics and promotion status
        """
        async with self._training_lock:
            logger.info("Starting model training...")

            # Get labeled training data
            X, y, memory_ids = await self._get_training_data()

            if len(X) < 50:
                logger.warning(f"Insufficient training data: {len(X)} samples")
                return {
                    "success": False,
                    "reason": f"Insufficient training data ({len(X)} samples, need 50+)",
                }

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)}")

            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=42,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            # Evaluate
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]

            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "f1": float(f1_score(y_val, y_pred, zero_division=0)),
                "auc_roc": float(roc_auc_score(y_val, y_pred_proba)) if len(np.unique(y_val)) > 1 else 0.5,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            }

            logger.info(f"Model metrics: F1={metrics['f1']:.3f}, AUC={metrics['auc_roc']:.3f}")

            # Get feature importance
            feature_importance = dict(zip(
                Features.feature_names(),
                [float(x) for x in model.feature_importances_]
            ))

            # Check if we should promote
            should_promote, comparison = await self._should_promote(metrics)

            if should_promote:
                new_version = await self._promote_model(model, metrics, feature_importance)
                logger.info(f"Promoted new model version {new_version}")

                # Notify Janitor to rescore
                await self.mail.send(
                    "janitor",
                    "model_promoted",
                    {"version": new_version, "metrics": metrics}
                )

                # Notify Gatekeeper to reload model
                await self.mail.send(
                    "gatekeeper",
                    "reload_model",
                    {"version": new_version}
                )

                return {
                    "success": True,
                    "promoted": True,
                    "version": new_version,
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "comparison": comparison,
                }
            else:
                logger.info("Model not promoted - no significant improvement")
                return {
                    "success": True,
                    "promoted": False,
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "comparison": comparison,
                }

    async def _get_training_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get labeled training data from memories with feedback.

        Returns:
            (features_array, labels_array, memory_ids)
        """
        async with async_session() as session:
            # Get memories that are at least 24 hours old and have feedback
            cutoff = datetime.utcnow() - timedelta(hours=24)

            # Get memories with explicit feedback
            sql = """
                SELECT DISTINCT m.id, m.content, m.source, m.embedding,
                    (
                        SELECT COUNT(*) FROM memory_events e
                        WHERE e.memory_id = m.id AND e.event = 'helpful'
                    ) as helpful_count,
                    (
                        SELECT COUNT(*) FROM memory_events e
                        WHERE e.memory_id = m.id AND e.event = 'not_helpful'
                    ) as not_helpful_count,
                    m.access_count
                FROM memories m
                WHERE m.created_at < :cutoff
                    AND m.embedding IS NOT NULL
                    AND (
                        EXISTS (SELECT 1 FROM memory_events e WHERE e.memory_id = m.id AND e.event IN ('helpful', 'not_helpful'))
                        OR m.access_count >= 3
                    )
            """

            from sqlalchemy import text
            result = await session.execute(text(sql), {"cutoff": cutoff})
            rows = result.fetchall()

            X = []
            y = []
            memory_ids = []

            for row in rows:
                # Determine label
                helpful = row.helpful_count
                not_helpful = row.not_helpful_count

                if helpful + not_helpful > 0:
                    # Use feedback ratio
                    label = 1 if helpful > not_helpful else 0
                elif row.access_count >= 5:
                    # High access count without negative feedback = probably useful
                    label = 1
                else:
                    # Moderate access, assume neutral-positive
                    label = 1 if row.access_count >= 3 else 0

                # Extract features
                embedding = row.embedding if row.embedding else None
                features = extract_features(row.content, row.source, embedding)

                X.append(features.to_array())
                y.append(label)
                memory_ids.append(str(row.id))

            return np.array(X), np.array(y), memory_ids

    async def _should_promote(self, new_metrics: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if new model should be promoted.

        Criteria:
        - F1 improves by at least 0.01
        - AUC doesn't drop by more than 0.02

        Returns:
            (should_promote, comparison_dict)
        """
        async with async_session() as session:
            result = await session.execute(
                select(Model).where(Model.is_active == True)
            )
            current_model = result.scalar_one_or_none()

            if not current_model:
                # No current model - always promote first one
                return True, {"reason": "First model"}

            old_metrics = current_model.metrics

            f1_improvement = new_metrics["f1"] - old_metrics.get("f1", 0)
            auc_change = new_metrics["auc_roc"] - old_metrics.get("auc_roc", 0)

            comparison = {
                "old_f1": old_metrics.get("f1", 0),
                "new_f1": new_metrics["f1"],
                "f1_improvement": f1_improvement,
                "old_auc": old_metrics.get("auc_roc", 0),
                "new_auc": new_metrics["auc_roc"],
                "auc_change": auc_change,
            }

            should_promote = f1_improvement >= 0.01 and auc_change >= -0.02

            comparison["reason"] = (
                "Meets promotion criteria" if should_promote
                else f"F1 improvement {f1_improvement:.3f} < 0.01 or AUC drop {-auc_change:.3f} > 0.02"
            )

            return should_promote, comparison

    async def _promote_model(
        self,
        model: xgb.XGBClassifier,
        metrics: Dict[str, float],
        feature_importance: Dict[str, float],
    ) -> int:
        """
        Promote a new model to active status.

        Returns:
            New model version number
        """
        async with async_session() as session:
            # Get next version number
            result = await session.execute(
                select(func.max(Model.version))
            )
            max_version = result.scalar() or 0
            new_version = max_version + 1

            # Deactivate current model
            await session.execute(
                update(Model).where(Model.is_active == True).values(is_active=False)
            )

            # Serialize model
            model_blob = pickle.dumps(model)

            # Create new model record
            new_model = Model(
                version=new_version,
                metrics=metrics,
                model_blob=model_blob,
                feature_importance=feature_importance,
                is_active=True,
            )
            session.add(new_model)

            # Reset training counter
            result = await session.execute(select(TrainingState))
            state = result.scalar_one_or_none()
            if state:
                state.memories_since_last_train = 0
                state.current_model_version = new_version
                state.last_train_at = datetime.utcnow()

            await session.commit()

            return new_version


async def main():
    """Run the Trainer agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    agent = TrainerAgent()
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
