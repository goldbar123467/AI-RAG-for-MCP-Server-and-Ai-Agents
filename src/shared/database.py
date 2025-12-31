"""Database models and connection management."""

from datetime import datetime
from typing import Optional, List, AsyncGenerator
from uuid import UUID
import enum

from sqlalchemy import (
    String,
    Text,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Enum,
    LargeBinary,
    ARRAY,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from .config import settings


class MemoryCategory(str, enum.Enum):
    """Types of memories."""
    DECISION = "decision"
    BUG_FIX = "bug_fix"
    PATTERN = "pattern"
    OUTCOME = "outcome"
    INSIGHT = "insight"
    CODE_SNIPPET = "code_snippet"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class MemoryTier(str, enum.Enum):
    """Memory quality tiers."""
    CORE = "core"
    ACTIVE = "active"
    ARCHIVE = "archive"
    QUARANTINE = "quarantine"


class RelationshipType(str, enum.Enum):
    """Types of memory relationships."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    CAUSED_BY = "caused_by"
    RELATED = "related"


class EventType(str, enum.Enum):
    """Types of memory events."""
    RETRIEVAL = "retrieval"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    STALE = "stale"


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Memory(Base):
    """A stored memory with embedding and quality scores."""

    __tablename__ = "memories"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(768))
    category: Mapped[MemoryCategory] = mapped_column(
        Enum(MemoryCategory, name="memory_category", create_type=False, values_callable=lambda x: [e.value for e in x]),
        default=MemoryCategory.OTHER
    )
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    source: Mapped[Optional[str]] = mapped_column(String)
    project: Mapped[Optional[str]] = mapped_column(String)
    extra_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    predicted_quality: Mapped[float] = mapped_column(Float, default=0.5)
    usefulness_score: Mapped[float] = mapped_column(Float, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    tier: Mapped[MemoryTier] = mapped_column(
        Enum(MemoryTier, name="memory_tier", create_type=False, values_callable=lambda x: [e.value for e in x]),
        default=MemoryTier.ACTIVE
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    # Relationships
    events: Mapped[List["MemoryEvent"]] = relationship(back_populates="memory", cascade="all, delete-orphan")
    outgoing_links: Mapped[List["MemoryLink"]] = relationship(
        back_populates="source",
        foreign_keys="MemoryLink.source_id",
        cascade="all, delete-orphan"
    )
    incoming_links: Mapped[List["MemoryLink"]] = relationship(
        back_populates="target",
        foreign_keys="MemoryLink.target_id",
        cascade="all, delete-orphan"
    )


class MemoryLink(Base):
    """A relationship between two memories."""

    __tablename__ = "memory_links"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    source_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"))
    target_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"))
    link_type: Mapped[RelationshipType] = mapped_column(
        Enum(RelationshipType, name="relationship_type", create_type=False, values_callable=lambda x: [e.value for e in x])
    )
    strength: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    # Relationships
    source: Mapped["Memory"] = relationship(back_populates="outgoing_links", foreign_keys=[source_id])
    target: Mapped["Memory"] = relationship(back_populates="incoming_links", foreign_keys=[target_id])


class MemoryEvent(Base):
    """An event associated with a memory (retrieval, feedback, etc.)."""

    __tablename__ = "memory_events"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    memory_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("memories.id", ondelete="CASCADE"))
    event: Mapped[EventType] = mapped_column(Enum(EventType, name="event_type", create_type=False, values_callable=lambda x: [e.value for e in x]))
    context: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    # Relationships
    memory: Mapped["Memory"] = relationship(back_populates="events")


class TrainingState(Base):
    """Singleton tracking training progress."""

    __tablename__ = "training_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    memories_since_last_train: Mapped[int] = mapped_column(Integer, default=0)
    current_model_version: Mapped[int] = mapped_column(Integer, default=0)
    last_train_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Model(Base):
    """A trained XGBoost model."""

    __tablename__ = "models"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    version: Mapped[int] = mapped_column(Integer, unique=True)
    metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    model_blob: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    feature_importance: Mapped[dict] = mapped_column(JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")


class Concept(Base):
    """An emerged concept cluster."""

    __tablename__ = "concepts"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    centroid: Mapped[Optional[List[float]]] = mapped_column(Vector(768))
    memory_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")


class ConceptMember(Base):
    """Membership of a memory in a concept."""

    __tablename__ = "concept_members"

    concept_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("concepts.id", ondelete="CASCADE"),
        primary_key=True
    )
    memory_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("memories.id", ondelete="CASCADE"),
        primary_key=True
    )
    similarity: Mapped[float] = mapped_column(Float)


# Database engine and session
engine = create_async_engine(settings.postgres_url, echo=False, pool_size=10, max_overflow=20)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
