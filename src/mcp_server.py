"""
MCP Server for RAG Brain - Exposes memory tools via Model Context Protocol.

This allows Claude Code and other MCP-compatible clients to directly
interact with the memory system using natural tool calls.
"""

import asyncio
import logging
from typing import Optional
from uuid import UUID

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.gatekeeper.agent import GatekeeperAgent
from src.librarian.agent import LibrarianAgent
from src.shared.database import async_session, Memory, MemoryTier, Concept, TrainingState
from src.shared.config import settings
from sqlalchemy import select, func


logger = logging.getLogger(__name__)

# Create MCP server
server = Server("rag-brain")

# Agent instances (created on demand)
_gatekeeper: Optional[GatekeeperAgent] = None
_librarian: Optional[LibrarianAgent] = None


def get_gatekeeper() -> GatekeeperAgent:
    global _gatekeeper
    if _gatekeeper is None:
        _gatekeeper = GatekeeperAgent()
    return _gatekeeper


def get_librarian() -> LibrarianAgent:
    global _librarian
    if _librarian is None:
        _librarian = LibrarianAgent()
    return _librarian


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available RAG Brain tools."""
    return [
        Tool(
            name="remember",
            description="Store a new memory in the RAG Brain. Use this to save insights, decisions, patterns, bug fixes, or any knowledge worth preserving. The system will grade quality and reject low-quality inputs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store. Be specific and include context."
                    },
                    "category": {
                        "type": "string",
                        "enum": ["decision", "bug_fix", "pattern", "outcome", "insight", "code_snippet", "documentation", "other"],
                        "description": "Category of the memory",
                        "default": "insight"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for filtering/organizing",
                        "default": []
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of the memory (human, claude, auto)",
                        "default": "claude"
                    },
                    "project": {
                        "type": "string",
                        "description": "Project this memory belongs to"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="recall",
            description="Search the RAG Brain for relevant memories. Use this to find past decisions, patterns, solutions, or any stored knowledge. Results are ranked by relevance, quality, and usefulness.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for - can be a question or keywords"
                    },
                    "project": {
                        "type": "string",
                        "description": "Filter by project name"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (1-50)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 50
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="feedback",
            description="Provide feedback on whether a memory was helpful. This trains the system to improve quality predictions and rankings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The UUID of the memory"
                    },
                    "helpful": {
                        "type": "boolean",
                        "description": "Whether the memory was helpful"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional explanation of why"
                    }
                },
                "required": ["memory_id", "helpful"]
            }
        ),
        Tool(
            name="forget",
            description="Mark a memory for deletion. Use this to remove incorrect, outdated, or irrelevant memories.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The UUID of the memory to forget"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this memory should be forgotten"
                    }
                },
                "required": ["memory_id"]
            }
        ),
        Tool(
            name="stats",
            description="Get statistics about the RAG Brain - total memories, quality averages, tier distribution, training status.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Filter stats by project"
                    }
                }
            }
        ),
        Tool(
            name="concepts",
            description="List emerged concept clusters. Concepts are automatically created when memories cluster around themes.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "remember":
            result = await handle_remember(arguments)
        elif name == "recall":
            result = await handle_recall(arguments)
        elif name == "feedback":
            result = await handle_feedback(arguments)
        elif name == "forget":
            result = await handle_forget(arguments)
        elif name == "stats":
            result = await handle_stats(arguments)
        elif name == "concepts":
            result = await handle_concepts(arguments)
        else:
            result = f"Unknown tool: {name}"

        return [TextContent(type="text", text=result)]
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_remember(args: dict) -> str:
    """Store a new memory."""
    content = args.get("content")
    if not content:
        return "Error: content is required"

    gatekeeper = get_gatekeeper()
    result = await gatekeeper.remember(
        content=content,
        category=args.get("category", "insight"),
        tags=args.get("tags", []),
        source=args.get("source", "claude"),
        project=args.get("project"),
        metadata={}
    )

    if result.get("rejected"):
        return f"Memory rejected: {result.get('reason', 'Unknown reason')}"

    return f"Memory stored successfully!\n- ID: {result['memory_id']}\n- Quality: {result['quality_score']:.2f}\n- Tier: {result['tier']}"


async def handle_recall(args: dict) -> str:
    """Search for memories."""
    query = args.get("query")
    if not query:
        return "Error: query is required"

    librarian = get_librarian()
    results = await librarian.recall(
        query=query,
        project=args.get("project"),
        tags=args.get("tags"),
        limit=args.get("limit", 5)
    )

    if not results:
        return "No memories found matching your query."

    output = [f"Found {len(results)} memories:\n"]
    for i, mem in enumerate(results, 1):
        output.append(f"## {i}. [{mem['category']}] (score: {mem['composite_score']:.2f})")
        output.append(f"ID: {mem['id']}")
        output.append(f"Content: {mem['content'][:500]}{'...' if len(mem['content']) > 500 else ''}")
        if mem['tags']:
            output.append(f"Tags: {', '.join(mem['tags'])}")
        if mem['project']:
            output.append(f"Project: {mem['project']}")
        output.append(f"Quality: {mem['predicted_quality']:.2f} | Usefulness: {mem['usefulness_score']:.2f} | Similarity: {mem['similarity']:.2f}")
        output.append("")

    return "\n".join(output)


async def handle_feedback(args: dict) -> str:
    """Record feedback on a memory."""
    memory_id = args.get("memory_id")
    helpful = args.get("helpful")

    if not memory_id:
        return "Error: memory_id is required"
    if helpful is None:
        return "Error: helpful (true/false) is required"

    try:
        memory_uuid = UUID(memory_id)
    except ValueError:
        return "Error: Invalid memory_id format"

    librarian = get_librarian()
    await librarian.feedback(
        memory_id=memory_uuid,
        helpful=helpful,
        context=args.get("context")
    )

    return f"Feedback recorded: {'helpful' if helpful else 'not helpful'}"


async def handle_forget(args: dict) -> str:
    """Mark a memory for deletion."""
    memory_id = args.get("memory_id")
    if not memory_id:
        return "Error: memory_id is required"

    try:
        memory_uuid = UUID(memory_id)
    except ValueError:
        return "Error: Invalid memory_id format"

    from datetime import datetime

    async with async_session() as session:
        result = await session.execute(
            select(Memory).where(Memory.id == memory_uuid)
        )
        memory = result.scalar_one_or_none()

        if not memory:
            return "Error: Memory not found"

        memory.extra_data = {
            **memory.extra_data,
            "flagged_for_deletion": True,
            "flagged_at": datetime.utcnow().isoformat(),
            "reason": args.get("reason", "User requested deletion"),
        }
        memory.tier = MemoryTier.QUARANTINE

        await session.commit()

    return f"Memory {memory_id} marked for deletion and moved to quarantine."


async def handle_stats(args: dict) -> str:
    """Get system statistics."""
    project = args.get("project")

    async with async_session() as session:
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

        # Averages
        result = await session.execute(
            select(func.avg(Memory.predicted_quality)).where(base_condition)
        )
        avg_quality = result.scalar() or 0

        result = await session.execute(
            select(func.avg(Memory.usefulness_score)).where(base_condition)
        )
        avg_usefulness = result.scalar() or 0

        # Training state
        result = await session.execute(select(TrainingState))
        state = result.scalar_one_or_none()

        total = sum(tier_counts.values())

        output = [
            "# RAG Brain Statistics",
            f"{'Project: ' + project if project else 'All projects'}",
            "",
            f"**Total Memories:** {total}",
            "",
            "## Tier Distribution",
            f"- Core: {tier_counts.get('core', 0)}",
            f"- Active: {tier_counts.get('active', 0)}",
            f"- Archive: {tier_counts.get('archive', 0)}",
            f"- Quarantine: {tier_counts.get('quarantine', 0)}",
            "",
            "## Quality Metrics",
            f"- Avg Quality: {float(avg_quality):.2f}",
            f"- Avg Usefulness: {float(avg_usefulness):.2f}",
            "",
            "## Training Status",
            f"- Memories since last train: {state.memories_since_last_train if state else 0}",
            f"- Current model version: {state.current_model_version if state else 0}",
            f"- Training threshold: {settings.training_threshold}",
        ]

        return "\n".join(output)


async def handle_concepts(args: dict) -> str:
    """List emerged concepts."""
    async with async_session() as session:
        result = await session.execute(
            select(Concept).order_by(Concept.memory_count.desc())
        )
        concepts = result.scalars().all()

        if not concepts:
            return "No concepts have emerged yet. Concepts form automatically as memories cluster around themes."

        output = ["# Emerged Concepts", ""]
        for c in concepts:
            output.append(f"## {c.name}")
            if c.description:
                output.append(f"{c.description}")
            output.append(f"Memories: {c.memory_count}")
            output.append("")

        return "\n".join(output)


async def main():
    """Run the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    logger.info("Starting RAG Brain MCP server...")

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
