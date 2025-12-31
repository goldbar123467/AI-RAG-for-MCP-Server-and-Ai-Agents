# Workbench RAG System

## What This Is

A self-improving memory system that acts as a shared brain across all projects. Every decision, bug fix, pattern, and outcome gets stored, graded, and retrievable by any agent working on any project. The system learns what good memories look like and continuously improves its quality gates.

## Core Philosophy

This is not a notepad. This is a learning system. Bad inputs get rejected or quarantined. Good inputs rise to the top. The system trains itself every 500 memories and re-evaluates everything it knows.

## Technology Stack

PostgreSQL with pgvector extension for vector storage and similarity search. Ollama running nomic-embed-text locally for free embeddings at 768 dimensions. XGBoost for quality prediction and gating. Python FastAPI for the MCP server. Agent Mail for coordination between the four agents.

## Database Design

The memories table is the core. Each memory has content, a 768-dimension embedding vector, a category type, tags, source attribution, and quality scores. The quality score is computed from specificity, actionability, source trust, and confidence. Each memory also has a tier assignment of core, active, archive, or quarantine.

The memory_links table connects related memories with relationship types like supports, contradicts, extends, caused_by, and related. Each link has a strength score.

The memory_events table logs every interaction with a memory including retrievals, explicit helpful and not helpful feedback, and staleness flags. This is the training signal.

The training_state table tracks how many memories since last training and the current model version. When the counter hits 500, training triggers automatically.

The models table stores each trained model version with its metrics, the pickled model blob, feature importance, and deployment status.

## The Four Agents

### Gatekeeper Agent

This agent owns the write path. Every memory request comes through Gatekeeper first.

When a remember request arrives, Gatekeeper extracts features from the content. Features include length, word count, unique word ratio, presence of numbers, presence of code, presence of reasoning words like because and therefore, source trust level, and embedding statistics like mean and standard deviation of the vector values.

Gatekeeper runs these features through the current XGBoost model to get a quality prediction between 0 and 1. If the prediction falls below the threshold, Gatekeeper rejects the memory and sends a message back to the requesting agent explaining why.

If the memory passes, Gatekeeper calls Ollama to generate the embedding, determines the appropriate category and tier, and inserts into PostgreSQL. After insert, Gatekeeper increments the training counter in training_state.

Gatekeeper also checks for near-duplicates by running a quick vector similarity search before insert. If a memory with similarity above 0.95 already exists, Gatekeeper rejects as duplicate or optionally merges the new information into the existing memory.

Gatekeeper sends a message to Librarian after every successful insert so Librarian can update its indexes and find related memories to link.

### Librarian Agent

This agent owns the read path and relationship management.

When a recall request arrives, Librarian embeds the query through Ollama, then runs a composite ranking query against PostgreSQL. The ranking combines vector similarity at 40 percent weight, predicted quality at 25 percent, proven usefulness at 25 percent, and recency at 10 percent. Librarian only searches memories above a minimum quality threshold and excludes quarantined memories entirely.

Librarian returns the top K results and logs a retrieval event for each memory returned. This implicit signal feeds back into training.

When agents mark a result as helpful or not helpful, Librarian updates the usefulness score on that memory and logs the feedback event. This explicit signal is the strongest training data.

Librarian also manages the memory_links table. After Gatekeeper inserts a new memory, Librarian searches for semantically similar existing memories and creates relationship links. If a new memory appears to contradict an existing one based on low but non-zero similarity with conflicting keywords, Librarian creates a contradicts link and flags both for human review.

Librarian maintains a concepts index that emerges organically. When multiple memories cluster around a theme, Librarian creates a concept entry and links the relevant memories to it. This allows queries like show me everything related to the API rate limiting concept.

### Trainer Agent

This agent owns the machine learning lifecycle.

Trainer watches the training_state table. When memories_since_last_train hits 500, Trainer takes over.

First, Trainer pulls all memories that have enough signal to label. A memory is labelable if it has explicit feedback events or if it has been accessed multiple times with a clear usefulness trend. Memories less than 24 hours old are excluded to let signals settle.

Trainer extracts features for each labeled memory and splits into 80 percent training and 20 percent validation sets.

Trainer trains a new XGBoost classifier with binary logistic objective predicting whether a memory will be useful. Training uses 100 estimators, max depth 6, learning rate 0.1, and AUC as the evaluation metric.

After training, Trainer evaluates on the validation set and computes accuracy, precision, recall, F1, and AUC-ROC. Trainer compares these metrics to the current deployed model.

If the new model improves F1 by at least 0.01 without tanking AUC by more than 0.02, Trainer promotes the new model. Promotion means serializing the model to the models table, setting is_active true, and updating training_state with the new version.

If the new model does not improve, Trainer logs the attempt and keeps the old model.

After promoting a new model, Trainer sends a message to Janitor to trigger a full rescore.

### Janitor Agent

This agent owns maintenance, rescoring, and tier management.

When Trainer promotes a new model, Janitor receives a message to rescore all memories. Janitor loads the new model, iterates through every memory in batches of 100, extracts features, runs prediction, and updates the predicted_quality column.

After rescoring, Janitor adjusts tiers based on the new scores combined with usage data.

Memories with predicted quality above 0.8, usefulness above 0.6, and access count above 3 get promoted to core tier. Core memories are always considered in retrieval.

Memories with predicted quality below 0.3 and usefulness below 0.3 get moved to quarantine tier. Quarantined memories are excluded from retrieval but kept for potential rehabilitation if the model changes.

Memories in core tier that now score below 0.5 predicted quality or below 0.4 usefulness get demoted to active tier.

Memories in active tier that have not been accessed in 90 days get moved to archive tier. Archived memories are retrievable but ranked lower.

Janitor also runs a daily cleanup job. This job identifies memories that have been quarantined for over 30 days with no improvement in score and flags them for permanent deletion pending human approval.

Janitor sends a summary message to a monitoring channel after each maintenance run with counts of promotions, demotions, quarantines, and archives.

## Agent Mail Coordination

All four agents register with Agent Mail on startup using the workbench-rag project identifier.

Message types between agents:

Gatekeeper to Librarian: memory_inserted with the new memory ID. Librarian should find and create links.

Librarian to Gatekeeper: duplicate_detected when a near-duplicate is found during insert check. Gatekeeper decides whether to reject or merge.

Gatekeeper to Trainer: counter_incremented after each insert. Trainer checks if threshold reached.

Trainer to Janitor: model_promoted with the new version number. Janitor should rescore everything.

Janitor to all agents: maintenance_complete with summary statistics.

Any agent to Librarian: feedback with memory ID and helpful boolean. Librarian updates scores.

External agents to Gatekeeper: remember with content, category, tags, source, and optional metadata.

External agents to Librarian: recall with query, optional project filter, optional tag filter, and limit.

## MCP Tools Exposed

The MCP server exposes these tools for external agents and Claude Code to use.

remember takes content as required, plus optional category, tags, project, source, and metadata. Returns the memory ID if accepted or rejection reason if not.

recall takes query as required, plus optional project, tags, and limit. Returns ranked list of memories with content, score, and ID.

feedback takes memory_id and helpful boolean, plus optional context explaining why. Returns confirmation.

concepts takes no arguments. Returns list of emerged concepts with memory counts.

forget takes memory_id. Marks for deletion pending approval.

stats takes optional project. Returns counts by tier, average quality, recent training metrics.

## The Learning Loop Visualized

Memories flow in through Gatekeeper who grades and stores them. Librarian serves them back when queried and collects feedback. Every 500 memories, Trainer builds a new model from the accumulated feedback. Janitor applies the new model to everything and shuffles memories between tiers. The cycle repeats with each iteration producing a smarter gate and better rankings.

Month one the system accepts most things and learns from mistakes. Month three the gate has learned your patterns and rejects low-quality inputs automatically. Month six the system knows that your manual inputs are trustworthy, that vague agent inferences are usually wrong, and that code snippets with context outperform naked code. Year one you have a second brain that understands your work style better than a new hire could learn in months.

## Project Structure

The repository should have a docker-compose.yml that brings up PostgreSQL with pgvector and Ollama with nomic-embed-text preloaded.

The src directory contains four subdirectories for each agent plus a shared directory for database models, embedding utilities, and feature extraction.

The mcp directory contains the FastAPI server that exposes MCP tools and routes requests to the appropriate agent.

The migrations directory contains SQL files for schema setup versioned with timestamps.

The models directory is gitignored and stores pickled XGBoost models locally.

The tests directory contains integration tests that spin up the full system and verify the learning loop works end to end.

## Environment Variables

POSTGRES_URL for the database connection string.

OLLAMA_URL defaulting to http://localhost:11434 for the embedding service.

AGENT_MAIL_URL for the coordination server.

AGENT_MAIL_TOKEN for authentication.

TRAINING_THRESHOLD defaulting to 500 for memories between retraining.

MIN_QUALITY_THRESHOLD defaulting to 0.3 for retrieval filtering.

## Getting Started

Start the infrastructure with docker compose up. Run migrations to create the schema. Start each agent as a separate process or container. Register the MCP server with Claude Code. Start remembering things.

The first 500 memories run with a baseline heuristic model. After that first training cycle, the real learning begins.
