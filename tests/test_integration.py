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
from uuid import UUID, uuid4
from datetime import datetime

import httpx


# Test configuration
MCP_URL = "http://localhost:8000"
TEST_TIMEOUT = 30.0


def unique_id():
    """Generate a unique ID for test content to avoid duplicates."""
    return str(uuid4())[:8]


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
    uid = unique_id()
    # Store a memory with unique content based on timestamp
    content = f"Test memory {uid}: When working with GraphQL resolvers, always use DataLoader to batch database queries and prevent N+1 problems. This pattern is essential for performance."

    remember_response = await client.post(
        "/remember",
        json={
            "content": content,
            "category": "pattern",
            "tags": ["graphql", "dataloader", uid],
            "source": "human",
            "project": "test",
        }
    )
    assert remember_response.status_code == 200
    result = remember_response.json()

    # Should be accepted
    assert result["rejected"] == False, f"Memory rejected: {result.get('reason')} (uid={uid})"
    assert "memory_id" in result
    assert result["quality_score"] > 0

    memory_id = result["memory_id"]

    # Recall the memory using the unique tag
    recall_response = await client.post(
        "/recall",
        json={
            "query": f"GraphQL DataLoader {uid}",
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
    uid = unique_id()
    # First store a memory with unique content
    content = f"Test feedback {uid}: Redis cache invalidation should happen atomically with database writes to prevent stale data. Use transaction pipelines for this."

    remember_response = await client.post(
        "/remember",
        json={
            "content": content,
            "category": "pattern",
            "tags": ["redis", "caching", uid],
            "source": "human",
        }
    )
    result = remember_response.json()
    assert not result["rejected"], f"Memory rejected: {result.get('reason')} (uid={uid})"
    memory_id = result["memory_id"]

    # Provide positive feedback
    feedback_response = await client.post(
        "/feedback",
        json={
            "memory_id": memory_id,
            "helpful": True,
            "context": "This helped me fix my caching issue",
        }
    )
    assert feedback_response.status_code == 200
    assert feedback_response.json()["success"] == True


@pytest.mark.asyncio
async def test_forget(client):
    """Test marking a memory for deletion."""
    uid = unique_id()
    # Store a memory with unique content
    content = f"Test forget {uid}: Kubernetes pod autoscaling should be configured based on CPU and memory metrics. Use HPA for horizontal scaling."

    remember_response = await client.post(
        "/remember",
        json={
            "content": content,
            "category": "other",
            "tags": ["kubernetes", uid],
            "source": "human",
        }
    )
    result = remember_response.json()
    assert not result["rejected"], f"Memory rejected: {result.get('reason')} (uid={uid})"
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
    uid = unique_id()
    # Use very unique content unlikely to match anything else
    content = f"Duplicate test {uid}: Terraform state files should be stored in S3 with DynamoDB locking enabled for team collaboration and conflict prevention."

    # Store first memory
    response1 = await client.post(
        "/remember",
        json={
            "content": content,
            "tags": ["terraform", uid],
            "source": "human",
        }
    )
    result1 = response1.json()
    assert not result1["rejected"], f"First memory rejected: {result1.get('reason')} (uid={uid})"

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
    uid = unique_id()
    # Store memories in different projects with unique content
    frontend_content = f"Project filter test {uid}: Vue.js composition API provides better TypeScript support than the options API. Use ref() and reactive() for state management."
    backend_content = f"Project filter test {uid}: FastAPI dependency injection system allows clean separation of concerns. Use Depends() for reusable components."

    await client.post(
        "/remember",
        json={
            "content": frontend_content,
            "project": f"frontend-{uid}",
            "tags": ["vue", uid],
            "source": "human",
        }
    )

    await client.post(
        "/remember",
        json={
            "content": backend_content,
            "project": f"backend-{uid}",
            "tags": ["fastapi", uid],
            "source": "human",
        }
    )

    # Recall with frontend filter
    response = await client.post(
        "/recall",
        json={
            "query": f"Vue composition API {uid}",
            "project": f"frontend-{uid}",
            "limit": 10,
        }
    )
    memories = response.json()["memories"]

    # Should only get frontend memories
    for memory in memories:
        if memory["project"]:
            assert memory["project"] == f"frontend-{uid}"


@pytest.mark.asyncio
async def test_ranking_factors(client):
    """Test that ranking considers multiple factors."""
    uid = unique_id()
    # Store high-quality memory with code - unique topic
    content = f"""
    Ranking test {uid}: To implement JWT refresh token rotation in Express.js:

    ```javascript
    const refreshTokens = new Map();

    app.post('/refresh', async (req, res) => {{
        const {{ refreshToken }} = req.body;
        if (!refreshTokens.has(refreshToken)) {{
            return res.status(401).json({{ error: 'Invalid token' }});
        }}
        const newAccessToken = generateAccessToken(user);
        const newRefreshToken = generateRefreshToken();
        refreshTokens.delete(refreshToken);
        refreshTokens.set(newRefreshToken, user.id);
        res.json({{ accessToken: newAccessToken, refreshToken: newRefreshToken }});
    }});
    ```

    This prevents token reuse attacks and improves security significantly.
    """

    await client.post(
        "/remember",
        json={
            "content": content,
            "category": "pattern",
            "tags": ["jwt", "security", uid],
            "source": "human",
        }
    )

    # Recall and verify it ranks high
    response = await client.post(
        "/recall",
        json={
            "query": f"JWT refresh token rotation Express {uid}",
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
