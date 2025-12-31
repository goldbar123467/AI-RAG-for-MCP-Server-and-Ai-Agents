# Hickup v1
<p align="center">
  <img src="https://img.shields.io/badge/RAG-Brain-6366f1?style=for-the-badge" alt="RAG Brain" height="50"/>
</p>
<p align="center">
  <img src="https://img.shields.io/badge/RAG-Brain-6366f1?style=for-the-badge" alt="RAG Brain" height="50"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/docker-ready-2496ED.svg?style=flat-square&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg?style=flat-square&logo=postgresql&logoColor=white" alt="PostgreSQL"/>
</p>

---

AI agents forget everything between sessions. Your coding assistant solves the same problems repeatedly, relearns your preferences, and loses context the moment you close the terminal.

RAG Brain is a persistent memory layer that any agent can write to and read from. Decisions, patterns, bug fixes, lessons learned. But it's not just storage. It learns what makes a memory useful. Bad inputs get rejected. Good memories surface first. Every 500 memories, the system retrains itself based on what actually helped you.

## Install

```bash
git clone https://github.com/goldbar123467/AI-RAG-for-MCP-Server-and-Ai-Agents.git
cd AI-RAG-for-MCP-Server-and-Ai-Agents
docker-compose up
```

Wait for healthy containers. Done.

**Test it:**
```bash
# Store a memory
curl -X POST http://localhost:8000/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Always use connection pooling with PostgreSQL in production"}'

# Retrieve memories
curl -X POST http://localhost:8000/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "database best practices"}'
```

## What You Get

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/remember` | POST | Store a memory |
| `/recall` | POST | Search memories semantically |
| `/feedback` | POST | Mark memory as helpful/not helpful |
| `/stats` | GET | System statistics |
| `/concepts` | GET | Emerged topic clusters |
| `/forget` | POST | Mark for deletion |
| `/health` | GET | Health check |

## Claude Code Integration

Add to `~/.claude/settings.json`:

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

Then talk naturally: "Remember this: always validate user input before database queries" or "What do I know about security?"

---

# Technical Deep Dive

## Why This Exists

Most memory systems are dumb storage. You put things in, you get things out. Retrieval quality depends entirely on what you stored.

RAG Brain tracks:
- Which memories get retrieved
- Which ones users mark as helpful
- Which ones solve problems

Every 500 memories, the system retrains its quality model. Bad memories get quarantined. Good memories rise. The gate gets smarter over time.

## System Architecture

```
                                 docker-compose up
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
   ┌─────────┐                   ┌─────────────┐                 ┌──────────┐
   │PostgreSQL│                   │   Ollama    │                 │Agent Mail│
   │+pgvector │                   │nomic-embed  │                 │ (router) │
   └────┬─────┘                   └──────┬──────┘                 └────┬─────┘
        │                                │                              │
        │    ┌───────────────────────────┼──────────────────────────────┤
        │    │                           │                              │
        │    ▼                           ▼                              ▼
        │ ┌──────────┐  messages  ┌──────────┐  messages  ┌──────────┐
        │ │Gatekeeper│───────────▶│Librarian │───────────▶│ Trainer  │
        │ │ (write)  │            │ (read)   │            │  (ML)    │
        │ └────┬─────┘            └────┬─────┘            └────┬─────┘
        │      │                       │                       │
        │      │                       │                       │ messages
        │      │                       │                       ▼
        │      │                       │               ┌──────────┐
        │      │                       │               │ Janitor  │
        │      │                       │               │(maintain)│
        │      │                       │               └────┬─────┘
        │      │                       │                    │
        └──────┴───────────────────────┴────────────────────┘
                              │
                              ▼
                       ┌────────────┐
                       │ MCP Server │ ◀── Claude Code / REST API
                       │  :8000     │
                       └────────────┘
```

## The Four Agents

### Gatekeeper (Write Path)

Every memory request hits Gatekeeper first. It decides if the memory is worth keeping.

**Feature Extraction:**
- Content length and word count
- Unique word ratio (vocabulary richness)
- Presence of numbers (specificity signal)
- Presence of code blocks
- Reasoning words ("because", "therefore", "since")
- Source trust level (human > agent > unknown)
- Embedding statistics (mean, std dev of vector)

**Quality Prediction:**
Gatekeeper runs features through an XGBoost classifier. The model outputs a score 0-1. Below threshold (default 0.3) means rejected.

**Duplicate Detection:**
Before insert, Gatekeeper searches for existing memories with cosine similarity > 0.95. If found, rejects as duplicate or merges.

**On Accept:**
1. Generate 768-dim embedding via Ollama
2. Assign tier (active by default)
3. Insert to PostgreSQL
4. Increment training counter
5. Notify Librarian to find related memories

### Librarian (Read Path)

Handles all recall requests and manages relationships between memories.

**Composite Ranking:**
```
score = (0.40 × similarity) +
        (0.25 × predicted_quality) +
        (0.25 × usefulness_score) +
        (0.10 × recency_score)
```

- **Similarity**: Cosine distance from query embedding to memory embedding
- **Predicted Quality**: XGBoost model's prediction for this memory
- **Usefulness Score**: Rolling average from explicit feedback
- **Recency Score**: Decay function, newer = higher

**Memory Linking:**
After Gatekeeper inserts a memory, Librarian:
1. Finds top 10 similar existing memories
2. Creates relationship links (supports, contradicts, extends, related)
3. If contradiction detected, flags both for review

**Concept Emergence:**
When multiple memories cluster semantically, Librarian creates a concept entry. This enables queries like "show me everything about rate limiting."

**Event Logging:**
Every retrieval logs an event. This implicit signal (memory was relevant enough to return) feeds into training.

### Trainer (ML Lifecycle)

Watches the training counter. At 500 memories, training triggers.

**Training Data Selection:**
- Memories with explicit feedback (helpful/not helpful marks)
- Memories with clear usage trends (retrieved often vs never)
- Excludes memories < 24 hours old (signals need time to settle)

**Model Training:**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='binary:logistic',
    eval_metric='auc'
)
```

80/20 train/validation split. Computes accuracy, precision, recall, F1, AUC-ROC.

**Promotion Criteria:**
New model must improve F1 by at least 0.01 without dropping AUC by more than 0.02. Otherwise, keep old model.

**On Promotion:**
1. Serialize model to database
2. Mark as active
3. Notify Janitor to rescore everything

### Janitor (Maintenance)

Runs batch operations to keep the system healthy.

**Rescore (triggered by model promotion):**
Iterate all memories in batches of 100. Extract features, run new model, update predicted_quality column.

**Tier Management:**

| From | To | Criteria |
|------|----|----------|
| Active | Core | quality > 0.8 AND usefulness > 0.6 AND accesses > 3 |
| Active | Quarantine | quality < 0.3 AND usefulness < 0.3 |
| Core | Active | quality < 0.5 OR usefulness < 0.4 |
| Active | Archive | No access in 90 days |
| Quarantine | Delete | Quarantined 30+ days, no score improvement |

**Daily Cleanup:**
- Flag stale quarantined memories for deletion
- Send summary to monitoring (counts of promotions, demotions, etc.)

## Agent Mail (Coordination)

Agents communicate through a simple message queue. No external dependencies. Runs as part of docker-compose.

**Message Types:**

| From | To | Type | Purpose |
|------|----|------|---------|
| Gatekeeper | Librarian | `memory_inserted` | Trigger link discovery |
| Gatekeeper | Trainer | `counter_incremented` | Check if training needed |
| Trainer | Janitor | `model_promoted` | Trigger full rescore |
| Janitor | All | `maintenance_complete` | Summary statistics |
| External | Gatekeeper | `remember` | Store memory request |
| External | Librarian | `recall` | Search request |
| Any | Librarian | `feedback` | Mark helpful/not helpful |

**API:**
```
POST /register     - Agent joins the system
POST /send         - Send to specific agent
POST /broadcast    - Send to all agents
GET  /messages     - Poll for messages
GET  /health       - Health check
```

## Database Schema

### memories
```sql
id              UUID PRIMARY KEY
content         TEXT NOT NULL
embedding       vector(768)
category        VARCHAR(50)  -- decision, bug_fix, pattern, outcome, insight, code_snippet, documentation, other
tags            TEXT[]
source          VARCHAR(100) -- human, agent, system
project         VARCHAR(100)
predicted_quality FLOAT      -- 0-1, from XGBoost
usefulness_score  FLOAT      -- 0-1, rolling average from feedback
tier            ENUM         -- core, active, archive, quarantine
access_count    INTEGER
extra_data      JSONB
created_at      TIMESTAMP
updated_at      TIMESTAMP
```

### memory_links
```sql
id              UUID PRIMARY KEY
source_id       UUID REFERENCES memories
target_id       UUID REFERENCES memories
relationship    VARCHAR(50)  -- supports, contradicts, extends, caused_by, related
strength        FLOAT        -- 0-1
created_at      TIMESTAMP
```

### memory_events
```sql
id              UUID PRIMARY KEY
memory_id       UUID REFERENCES memories
event_type      VARCHAR(50)  -- retrieved, helpful, not_helpful, stale
context         JSONB
created_at      TIMESTAMP
```

### training_state
```sql
id                      INTEGER PRIMARY KEY
memories_since_last_train INTEGER
current_model_version   INTEGER
last_train_at           TIMESTAMP
```

### models
```sql
id              UUID PRIMARY KEY
version         INTEGER
metrics         JSONB        -- accuracy, precision, recall, f1, auc
model_blob      BYTEA        -- pickled XGBoost model
feature_importance JSONB
is_active       BOOLEAN
created_at      TIMESTAMP
```

### concepts
```sql
id              UUID PRIMARY KEY
name            VARCHAR(200)
description     TEXT
centroid        vector(768)  -- average embedding of member memories
memory_count    INTEGER
created_at      TIMESTAMP
```

**Indexes:**
```sql
-- Vector similarity search (IVFFlat for performance)
CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Common queries
CREATE INDEX ON memories (tier);
CREATE INDEX ON memories (project);
CREATE INDEX ON memories USING gin (tags);
CREATE INDEX ON memory_events (memory_id, event_type);
```

## Embedding Model

**Model:** nomic-embed-text via Ollama
**Dimensions:** 768

Why this model:
- Runs locally (no API costs)
- Good performance on retrieval benchmarks
- Small enough for CPU inference
- Apache 2.0 license

**Embedding Flow:**
1. Request hits Gatekeeper/Librarian
2. Call Ollama at `http://ollama:11434/api/embeddings`
3. Receive 768-dim float array
4. Store/compare using pgvector

## Quality Model Features

Features extracted for XGBoost:

| Feature | Type | Description |
|---------|------|-------------|
| `length` | int | Character count |
| `word_count` | int | Word count |
| `unique_ratio` | float | Unique words / total words |
| `has_numbers` | bool | Contains digits |
| `has_code` | bool | Contains code markers (```, indentation) |
| `has_reasoning` | bool | Contains "because", "therefore", "since", etc. |
| `source_trust` | float | human=1.0, agent=0.7, unknown=0.5 |
| `embedding_mean` | float | Mean of embedding vector |
| `embedding_std` | float | Std dev of embedding vector |

## Configuration Reference

```bash
# Database
POSTGRES_URL=postgresql+asyncpg://postgres:ragbrain@postgres:5432/workbench

# Embeddings
OLLAMA_URL=http://ollama:11434
EMBEDDING_MODEL=nomic-embed-text

# Quality gating
MIN_QUALITY_THRESHOLD=0.3      # Reject below this
DUPLICATE_SIMILARITY_THRESHOLD=0.95

# Training
TRAINING_THRESHOLD=500         # Memories between retraining

# Ranking weights (must sum to 1.0)
SIMILARITY_WEIGHT=0.4
QUALITY_WEIGHT=0.25
USEFULNESS_WEIGHT=0.25
RECENCY_WEIGHT=0.1

# Tier thresholds
CORE_QUALITY_THRESHOLD=0.8
CORE_USEFULNESS_THRESHOLD=0.6
CORE_ACCESS_THRESHOLD=3
QUARANTINE_QUALITY_THRESHOLD=0.3
QUARANTINE_USEFULNESS_THRESHOLD=0.3
ARCHIVE_DAYS=90
```

## Project Structure

```
rag-brain/
├── src/
│   ├── gatekeeper/
│   │   └── agent.py          # Write path, quality gating
│   ├── librarian/
│   │   └── agent.py          # Read path, ranking, linking
│   ├── trainer/
│   │   └── agent.py          # XGBoost training lifecycle
│   ├── janitor/
│   │   └── agent.py          # Rescoring, tier management
│   ├── agent_mail_server/
│   │   └── server.py         # Inter-agent message routing
│   ├── shared/
│   │   ├── config.py         # Pydantic settings
│   │   ├── database.py       # SQLAlchemy models
│   │   ├── embeddings.py     # Ollama client
│   │   ├── features.py       # Feature extraction
│   │   └── agent_mail.py     # Agent Mail client
│   └── mcp_server.py         # MCP protocol for Claude Code
├── mcp_rest/
│   └── server.py             # FastAPI REST server
├── migrations/
│   └── 001_initial_schema.sql
├── tests/
│   ├── conftest.py
│   └── test_integration.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env
├── CLAUDE.md                 # Instructions for Claude Code
└── README.md
```

## The Learning Curve

**Week 1-2:** System accepts most inputs. Quality model uses baseline heuristics. You're building training data.

**Month 1:** First few training cycles complete. Model starts recognizing patterns in what you mark as helpful.

**Month 3:** Gate rejects vague or low-value inputs automatically. Retrieval consistently returns useful results.

**Month 6:** System has learned your style. Knows that your manual inputs are trustworthy, that code with context beats naked snippets, that certain agents produce better memories than others.

**Year 1:** You have a brain that actually knows your work.

## Troubleshooting

**Ollama slow to start:**
First run downloads ~1GB model. Check: `docker-compose logs -f ollama`

**Memories rejected unexpectedly:**
Quality threshold might be too high. Check `/stats` endpoint, lower `MIN_QUALITY_THRESHOLD` if needed.

**No results from recall:**
Memories might be quarantined. Check `/stats` for tier counts. Review quarantine threshold.

**Agents not starting:**
Check Agent Mail is healthy: `curl http://localhost:8766/health`

## License

MIT
