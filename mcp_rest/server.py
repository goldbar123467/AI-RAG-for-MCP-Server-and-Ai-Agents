"""
MCP REST Server - FastAPI server exposing memory tools.

Endpoints:
- POST /remember - Store a new memory
- POST /recall - Retrieve relevant memories
- POST /feedback - Record feedback on a memory
- GET /concepts - List emerged concepts
- POST /forget - Mark a memory for deletion
- GET /stats - Get system statistics
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, update, func

from src.shared.config import settings
from src.shared.database import Memory, MemoryTier, Concept, TrainingState, Model, async_session
from src.shared.agent_mail import AgentMailClient


logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Brain MCP Server",
    description="Self-improving memory system API",
    version="1.0.0",
)

# Agent Mail client for routing to agents
mail = AgentMailClient("mcp-server")


# Request/Response Models

class RememberRequest(BaseModel):
    """Request to store a new memory."""
    content: str = Field(..., description="The memory content")
    category: Optional[str] = Field("other", description="Memory category")
    tags: Optional[List[str]] = Field(default_factory=list, description="Associated tags")
    source: Optional[str] = Field(None, description="Source of the memory")
    project: Optional[str] = Field(None, description="Associated project")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class RememberResponse(BaseModel):
    """Response from storing a memory."""
    rejected: bool
    memory_id: Optional[str] = None
    quality_score: Optional[float] = None
    tier: Optional[str] = None
    reason: Optional[str] = None
    existing_id: Optional[str] = None


class RecallRequest(BaseModel):
    """Request to retrieve memories."""
    query: str = Field(..., description="Search query")
    project: Optional[str] = Field(None, description="Filter by project")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: Optional[int] = Field(10, ge=1, le=50, description="Max results")


class MemoryResult(BaseModel):
    """A memory in recall results."""
    id: str
    content: str
    category: str
    tags: List[str]
    source: Optional[str]
    project: Optional[str]
    metadata: Dict[str, Any]
    predicted_quality: float
    usefulness_score: float
    similarity: float
    composite_score: float
    tier: str
    created_at: str


class RecallResponse(BaseModel):
    """Response from memory recall."""
    memories: List[MemoryResult]


class FeedbackRequest(BaseModel):
    """Request to record feedback."""
    memory_id: str = Field(..., description="Memory ID")
    helpful: bool = Field(..., description="Whether the memory was helpful")
    context: Optional[str] = Field(None, description="Optional context about why")


class FeedbackResponse(BaseModel):
    """Response from feedback recording."""
    success: bool
    error: Optional[str] = None


class ForgetRequest(BaseModel):
    """Request to mark a memory for deletion."""
    memory_id: str = Field(..., description="Memory ID to forget")
    reason: Optional[str] = Field(None, description="Reason for deletion")


class ForgetResponse(BaseModel):
    """Response from forget request."""
    success: bool
    error: Optional[str] = None


class ConceptInfo(BaseModel):
    """Information about a concept."""
    id: str
    name: str
    description: Optional[str]
    memory_count: int
    created_at: str


class ConceptsResponse(BaseModel):
    """Response with concepts list."""
    concepts: List[ConceptInfo]


class TierCounts(BaseModel):
    """Memory counts by tier."""
    core: int
    active: int
    archive: int
    quarantine: int


class TrainingInfo(BaseModel):
    """Information about training state."""
    memories_since_last_train: int
    current_model_version: int
    last_train_at: Optional[str]
    training_threshold: int


class StatsResponse(BaseModel):
    """Response with system statistics."""
    total_memories: int
    tier_counts: TierCounts
    avg_quality: float
    avg_usefulness: float
    training: TrainingInfo


# Startup/Shutdown

@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    await mail.register()
    logger.info("MCP server started")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("MCP server stopping")


# Endpoints

@app.post("/remember", response_model=RememberResponse)
async def remember(request: RememberRequest) -> RememberResponse:
    """
    Store a new memory in the system.

    The memory goes through quality gating - it may be rejected if quality
    is below threshold or if a near-duplicate already exists.
    """
    # Import here to avoid circular imports
    from src.gatekeeper.agent import GatekeeperAgent

    agent = GatekeeperAgent()
    result = await agent.remember(
        content=request.content,
        category=request.category,
        tags=request.tags,
        source=request.source,
        project=request.project,
        metadata=request.metadata,
    )

    return RememberResponse(**result)


@app.post("/recall", response_model=RecallResponse)
async def recall(request: RecallRequest) -> RecallResponse:
    """
    Retrieve relevant memories using semantic search.

    Results are ranked by a composite score of:
    - 40% vector similarity
    - 25% predicted quality
    - 25% proven usefulness
    - 10% recency
    """
    from src.librarian.agent import LibrarianAgent

    agent = LibrarianAgent()
    results = await agent.recall(
        query=request.query,
        project=request.project,
        tags=request.tags,
        limit=request.limit,
    )

    return RecallResponse(memories=[MemoryResult(**m) for m in results])


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    Record feedback on whether a memory was helpful.

    This feedback is used to train the quality prediction model and
    adjust memory usefulness scores.
    """
    from src.librarian.agent import LibrarianAgent

    try:
        memory_uuid = UUID(request.memory_id)
    except ValueError:
        return FeedbackResponse(success=False, error="Invalid memory_id")

    agent = LibrarianAgent()
    await agent.feedback(
        memory_id=memory_uuid,
        helpful=request.helpful,
        context=request.context,
    )

    return FeedbackResponse(success=True)


@app.post("/forget", response_model=ForgetResponse)
async def forget(request: ForgetRequest) -> ForgetResponse:
    """
    Mark a memory for deletion.

    The memory is not immediately deleted but marked for review.
    """
    try:
        memory_uuid = UUID(request.memory_id)
    except ValueError:
        return ForgetResponse(success=False, error="Invalid memory_id")

    async with async_session() as session:
        result = await session.execute(
            select(Memory).where(Memory.id == memory_uuid)
        )
        memory = result.scalar_one_or_none()

        if not memory:
            return ForgetResponse(success=False, error="Memory not found")

        # Mark for deletion in extra_data
        from datetime import datetime
        memory.extra_data = {
            **memory.extra_data,
            "flagged_for_deletion": True,
            "flagged_at": datetime.utcnow().isoformat(),
            "reason": request.reason or "User requested deletion",
        }
        memory.tier = MemoryTier.QUARANTINE

        await session.commit()

    return ForgetResponse(success=True)


@app.get("/concepts", response_model=ConceptsResponse)
async def get_concepts() -> ConceptsResponse:
    """
    Get all emerged concept clusters.

    Concepts are automatically created when memories cluster around themes.
    """
    async with async_session() as session:
        result = await session.execute(
            select(Concept).order_by(Concept.memory_count.desc())
        )
        concepts = result.scalars().all()

        return ConceptsResponse(
            concepts=[
                ConceptInfo(
                    id=str(c.id),
                    name=c.name,
                    description=c.description,
                    memory_count=c.memory_count,
                    created_at=c.created_at.isoformat(),
                )
                for c in concepts
            ]
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(project: Optional[str] = None) -> StatsResponse:
    """
    Get system statistics.

    Optionally filter by project.
    """
    async with async_session() as session:
        # Base query condition
        base_condition = Memory.project == project if project else True

        # Count by tier
        tier_counts = {}
        for tier in MemoryTier:
            result = await session.execute(
                select(func.count(Memory.id)).where(
                    Memory.tier == tier,
                    base_condition,
                )
            )
            tier_counts[tier.value] = result.scalar() or 0

        # Average quality
        result = await session.execute(
            select(func.avg(Memory.predicted_quality)).where(base_condition)
        )
        avg_quality = result.scalar() or 0

        # Average usefulness
        result = await session.execute(
            select(func.avg(Memory.usefulness_score)).where(base_condition)
        )
        avg_usefulness = result.scalar() or 0

        # Training state
        result = await session.execute(select(TrainingState))
        state = result.scalar_one_or_none()

        training_info = TrainingInfo(
            memories_since_last_train=state.memories_since_last_train if state else 0,
            current_model_version=state.current_model_version if state else 0,
            last_train_at=state.last_train_at.isoformat() if state and state.last_train_at else None,
            training_threshold=settings.training_threshold,
        )

        total = sum(tier_counts.values())

        return StatsResponse(
            total_memories=total,
            tier_counts=TierCounts(**tier_counts),
            avg_quality=float(avg_quality),
            avg_usefulness=float(avg_usefulness),
            training=training_info,
        )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


def main():
    """Run the MCP server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    uvicorn.run(
        "mcp_rest.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
