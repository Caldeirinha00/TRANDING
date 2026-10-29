---
phase: 4
plan: 9
wave: 1
depends_on: ["08"]
files_modified: ["memory/embeddings.py", "memory/trade_journal.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "System can embed text locally using FastEmbed"
    - "System can store and retrieve similar trades from Qdrant"
  artifacts:
    - "memory/embeddings.py exists"
    - "memory/trade_journal.py exists"
---

# Plan 4.9: RAG Memory and Trade Journaling

<objective>
Implement the vector database integration to allow the AI Orchestrator to "remember" past trades and outcomes. 
Purpose: Provides contextual history to prevent the AI from repeating mistakes in similar market conditions.
Output: embeddings.py and trade_journal.py.
</objective>

<context>
Load for context:
- core/config.py
</context>

<tasks>

<task type="auto">
  <name>Implement Qdrant Embeddings Client</name>
  <files>memory/embeddings.py</files>
  <action>
    Create `QdrantMemory` class using `QdrantClient`.
    Connect to `CONFIG.QDRANT_HOST` on port 6333.
    Set up the collection "trades" and configure the embedding model (FastEmbed).
  </action>
  <verify>python -c "import memory.embeddings"</verify>
  <done>Client syntax is valid and connection logic is sound</done>
</task>

<task type="auto">
  <name>Implement Trade Journal</name>
  <files>memory/trade_journal.py</files>
  <action>
    Create functions to `log_trade_outcome(trade_data, outcome, rationale)` and `find_similar_trades(current_context, limit=3)`.
    These functions will serialize the context into a string, embed it, and interact with `QdrantMemory`.
  </action>
  <verify>python -c "import memory.trade_journal"</verify>
  <done>Syntax is valid</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Both modules compile successfully.
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
