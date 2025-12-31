"""
Integration tests for RAG Brain system.

These tests verify the full learning loop:
1. Insert memories via MCP
2. Recall and verify ranking
3. Provide feedback
4. Verify tier changes
"""

import asyncio
import pytest
from uuid import UUID
from datetime import datetime

import httpx


# Test configuration
MCP_URL = "http://localhost:8000"
TEST_TIMEOUT = 30.0


@pytest.fixture
def client():
    """Create HTTP client for MCP server."""
    return httpx.AsyncClient(base_url=MCP_URL, timeout=TEST_TIMEOUT)


@pytest.mark.asyncio
async def test_health(client):
    """Test health endpoint."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_remember_and_recall(client):
    """Test storing and retrieving a memory."""
    # Store a memory
    remember_response = await client.post(
        "/remember",
        json={
            "content": "Python's asyncio library uses an event loop to handle concurrent operations efficiently. Always use async/await syntax for non-blocking I/O.",
            "category": "pattern",
            "tags": ["python", "async", "concurrency"],
            "source": "human",
            "project": "test",
        }
    )
    assert remember_response.status_code == 200
    result = remember_response.json()

    # Should be accepted
    assert result["rejected"] == False
    assert "memory_id" in result
    assert result["quality_score"] > 0

    memory_id = result["memory_id"]

    # Recall the memory
    recall_response = await client.post(
        "/recall",
        json={
            "query": "How do I use async in Python?",
            "limit": 5,
        }
    )
    assert recall_response.status_code == 200
    memories = recall_response.json()["memories"]

    # Should find our memory
    assert len(memories) > 0

    # Check if our memory is in results
    found = any(m["id"] == memory_id for m in memories)
    assert found, f"Memory {memory_id} not found in recall results"


@pytest.mark.asyncio
async def test_quality_rejection(client):
    """Test that low-quality memories are rejected."""
    # Try to store a very short, low-quality memory
    response = await client.post(
        "/remember",
        json={
            "content": "ok",
            "source": "auto",
        }
    )
    assert response.status_code == 200
    result = response.json()

    # Should be rejected due to low quality
    assert result["rejected"] == True
    assert "quality" in result.get("reason", "").lower() or "threshold" in result.get("reason", "").lower()


@pytest.mark.asyncio
async def test_feedback(client):
    """Test providing feedback on a memory."""
    # First store a memory
    remember_response = await client.post(
        "/remember",
        json={
            "content": "When debugging async code, use asyncio.run() in the main entry point and asyncio.create_task() for concurrent operations.",
            "category": "pattern",
            "tags": ["python", "debugging"],
            "source": "human",
        }
    )
    result = remember_response.json()
    assert not result["rejected"]
    memory_id = result["memory_id"]

    # Provide positive feedback
    feedback_response = await client.post(
        "/feedback",
        json={
            "memory_id": memory_id,
            "helpful": True,
            "context": "This helped me fix my async issue",
        }
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json()["success"] == True


@pytest.mark.asyncio
async def test_forget(client):
    """Test marking a memory for deletion."""
    # Store a memory
    remember_response = await client.post(
        "/remember",
        json={
            "content": "This is a test memory that should be deleted. It contains specific test patterns for verification.",
            "category": "other",
            "source": "human",
        }
    )
    result = remember_response.json()
    assert not result["rejected"]
    memory_id = result["memory_id"]

    # Mark for deletion
    forget_response = await client.post(
        "/forget",
        json={
            "memory_id": memory_id,
            "reason": "Test deletion",
        }
    )
    assert forget_response.status_code == 200
    assert forget_response.json()["success"] == True


@pytest.mark.asyncio
async def test_stats(client):
    """Test statistics endpoint."""
    response = await client.get("/stats")
    assert response.status_code == 200

    stats = response.json()
    assert "total_memories" in stats
    assert "tier_counts" in stats
    assert "avg_quality" in stats
    assert "training" in stats


@pytest.mark.asyncio
async def test_concepts(client):
    """Test concepts endpoint."""
    response = await client.get("/concepts")
    assert response.status_code == 200

    data = response.json()
    assert "concepts" in data
    assert isinstance(data["concepts"], list)


@pytest.mark.asyncio
async def test_duplicate_detection(client):
    """Test that near-duplicates are detected."""
    content = "SQLAlchemy's async session should be used with async with for proper resource cleanup. This is a best practice for database connections."

    # Store first memory
    response1 = await client.post(
        "/remember",
        json={
            "content": content,
            "source": "human",
        }
    )
    result1 = response1.json()
    assert not result1["rejected"]

    # Try to store nearly identical content
    response2 = await client.post(
        "/remember",
        json={
            "content": content + " ",  # Very minor change
            "source": "human",
        }
    )
    result2 = response2.json()

    # Should be rejected as duplicate
    assert result2["rejected"] == True
    assert "duplicate" in result2.get("reason", "").lower()


@pytest.mark.asyncio
async def test_project_filtering(client):
    """Test recall filtering by project."""
    # Store memories in different projects
    await client.post(
        "/remember",
        json={
            "content": "This is a frontend pattern for React components. Always use functional components with hooks.",
            "project": "frontend",
            "source": "human",
        }
    )

    await client.post(
        "/remember",
        json={
            "content": "This is a backend pattern for API design. Use REST conventions with proper status codes.",
            "project": "backend",
            "source": "human",
        }
    )

    # Recall with frontend filter
    response = await client.post(
        "/recall",
        json={
            "query": "patterns",
            "project": "frontend",
            "limit": 10,
        }
    )
    memories = response.json()["memories"]

    # Should only get frontend memories
    for memory in memories:
        if memory["project"]:
            assert memory["project"] == "frontend"


@pytest.mark.asyncio
async def test_ranking_factors(client):
    """Test that ranking considers multiple factors."""
    # Store high-quality memory with code
    await client.post(
        "/remember",
        json={
            "content": """
            To optimize database queries in SQLAlchemy, use eager loading:

            ```python
            from sqlalchemy.orm import selectinload

            query = select(User).options(selectinload(User.posts))
            result = await session.execute(query)
            ```

            This prevents N+1 query problems and improves performance significantly.
            """,
            "category": "pattern",
            "tags": ["sqlalchemy", "performance", "optimization"],
            "source": "human",
        }
    )

    # Recall and verify it ranks high
    response = await client.post(
        "/recall",
        json={
            "query": "How to optimize SQLAlchemy queries?",
            "limit": 5,
        }
    )
    memories = response.json()["memories"]

    # Should have results
    assert len(memories) > 0

    # First result should have high composite score
    assert memories[0]["composite_score"] > 0.5


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
