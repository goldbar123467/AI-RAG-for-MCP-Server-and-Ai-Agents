# RAG Brain

A self-improving memory system that acts as a shared brain across all your projects. Every decision, bug fix, pattern, and outcome gets stored, graded, and retrievable by any agent or application.

## What Makes This Different

This isn't a notepad. It's a **learning system**:

- **Quality Gating**: Bad inputs get rejected automatically using ML-based quality prediction
- **Feedback Loop**: The system learns from explicit feedback (helpful/not helpful) and implicit signals (retrieval frequency)
- **Self-Training**: Every 500 memories, the system retrains its quality model
- **Tier Management**: Memories rise and fall through tiers based on proven usefulness

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Gatekeeper │     │  Librarian  │     │   Trainer   │     │   Janitor   │
│  (writes)   │────▶│  (reads)    │────▶│  (learning) │────▶│  (maintenance)
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       └───────────────────┴───────────────────┴───────────────────┘
                                    │
                           ┌───────────────┐
                           │  PostgreSQL   │
                           │  + pgvector   │
                           └───────────────┘
```

### The Four Agents

| Agent | Responsibility |
|-------|----------------|
| **Gatekeeper** | Owns the write path. Extracts features, predicts quality, rejects bad inputs, detects duplicates, generates embeddings |
| **Librarian** | Owns the read path. Semantic search with composite ranking, feedback collection, memory linking |
| **Trainer** | ML lifecycle. Trains XGBoost models on accumulated feedback, promotes models that improve metrics |
| **Janitor** | Maintenance. Rescores memories with new models, manages tier promotions/demotions, cleanup |

## Technology Stack

- **PostgreSQL + pgvector** - Vector storage and similarity search
- **Ollama + nomic-embed-text** - Local embeddings (768 dimensions, free)
- **XGBoost** - Quality prediction model
- **FastAPI** - REST API server
- **MCP Protocol** - Claude Code integration
- **SQLAlchemy** - Async database ORM

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+

### Start the System

```bash
# Start all services
docker compose up -d

# Wait for initialization (Ollama pulls the embedding model on first run)
docker compose logs -f ollama

# Run tests to verify everything works
python -m pytest tests/ -v
```

## Claude Code Integration (MCP)

RAG Brain includes an MCP (Model Context Protocol) server that integrates directly with Claude Code, allowing natural language interaction with your memory system.

### Setup

1. **Create a virtual environment and install dependencies:**
```bash
cd rag-brain
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. **Add to Claude Code settings** (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "rag-brain": {
      "command": "/path/to/rag-brain/.venv/bin/python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/rag-brain"
    }
  }
}
```

3. **Restart Claude Code** to load the MCP server.

### MCP Tools

| Tool | Description |
|------|-------------|
| `remember` | Store a memory with automatic quality gating |
| `recall` | Search memories with semantic ranking |
| `feedback` | Mark a memory as helpful/not helpful |
| `forget` | Mark a memory for deletion |
| `stats` | Get system statistics |
| `concepts` | List emerged concept clusters |

### Natural Language Usage

Once configured, you can interact naturally:

- **"Remember this: When using PostgreSQL with pgvector, always create an IVFFlat index for better performance"**
- **"What do I know about database optimization?"**
- **"That last memory was really helpful"**
- **"Show me my brain stats"**
- **"Forget memory [uuid]"**

## REST API

The REST API runs on `http://localhost:8000`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/remember` | POST | Store a new memory |
| `/recall` | POST | Retrieve relevant memories |
| `/feedback` | POST | Record helpful/not helpful feedback |
| `/concepts` | GET | List emerged concept clusters |
| `/forget` | POST | Mark a memory for deletion |
| `/stats` | GET | System statistics |
| `/health` | GET | Health check |

### Example Usage

**Store a memory:**
```bash
curl -X POST http://localhost:8000/remember \
  -H "Content-Type: application/json" \
  -d '{
    "content": "When debugging async Python code, always check if you forgot to await a coroutine. The error messages can be misleading.",
    "category": "pattern",
    "tags": ["python", "async", "debugging"],
    "source": "human",
    "project": "my-project"
  }'
```

**Retrieve memories:**
```bash
curl -X POST http://localhost:8000/recall \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how to debug async code",
    "limit": 5
  }'
```

**Provide feedback:**
```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "uuid-from-recall",
    "helpful": true,
    "context": "Exactly what I needed"
  }'
```

## Memory Ranking

Results are ranked using a composite score:

| Factor | Weight | Description |
|--------|--------|-------------|
| Vector Similarity | 40% | Semantic match to query |
| Predicted Quality | 25% | ML model's quality prediction |
| Proven Usefulness | 25% | Historical helpfulness from feedback |
| Recency | 10% | Newer memories get slight boost |

## Memory Tiers

| Tier | Criteria | Behavior |
|------|----------|----------|
| **Core** | Quality > 0.8, usefulness > 0.6, accessed 3+ times | Always included in search |
| **Active** | Default tier for accepted memories | Normal search inclusion |
| **Archive** | Not accessed in 90+ days | Included but ranked lower |
| **Quarantine** | Quality < 0.3 or usefulness < 0.3 | Excluded from search |

## Configuration

Environment variables (set in `.env` or docker-compose):

```bash
# Database
POSTGRES_URL=postgresql+asyncpg://postgres:ragbrain@localhost:5433/workbench

# Ollama
OLLAMA_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text

# Training
TRAINING_THRESHOLD=500        # Memories between retraining
MIN_QUALITY_THRESHOLD=0.3     # Minimum quality to accept

# Ranking weights
SIMILARITY_WEIGHT=0.4
QUALITY_WEIGHT=0.25
USEFULNESS_WEIGHT=0.25
RECENCY_WEIGHT=0.1

# Duplicate detection
DUPLICATE_SIMILARITY_THRESHOLD=0.95
```

## Project Structure

```
rag-brain/
├── src/
│   ├── gatekeeper/      # Write path agent
│   ├── librarian/       # Read path agent
│   ├── trainer/         # ML training agent
│   ├── janitor/         # Maintenance agent
│   ├── shared/          # Common utilities
│   │   ├── config.py    # Settings management
│   │   ├── database.py  # SQLAlchemy models
│   │   ├── embeddings.py# Ollama integration
│   │   └── features.py  # Feature extraction
│   └── mcp_server.py    # MCP protocol server for Claude Code
├── mcp_rest/
│   └── server.py        # FastAPI REST server
├── migrations/
│   └── 001_initial_schema.sql
├── tests/
│   └── test_integration.py
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## The Learning Loop

1. **Month 1**: System accepts most inputs, learns from mistakes
2. **Month 3**: Quality gate has learned your patterns, auto-rejects low-quality inputs
3. **Month 6**: System knows trusted sources, understands that context matters
4. **Year 1**: A second brain that understands your work style

## Memory Categories

- `decision` - Architectural or design decisions
- `bug_fix` - Bug fixes and their solutions
- `pattern` - Code patterns and best practices
- `outcome` - Results of experiments or changes
- `insight` - Learnings and realizations
- `code_snippet` - Reusable code examples
- `documentation` - Documentation notes
- `other` - Everything else

## License

MIT
