# GSD Roadmap: AI Trading System (MT5 + MCP)

## Phase 1: High-Speed Data Pipeline
**Goal:** Establish ultra-fast local infrastructure and extract data from MT5 to QuestDB and DragonflyDB.
**Must-haves:**
- QuestDB, DragonflyDB, and Qdrant running via Docker
- Python project configured with dependencies
- MCP Client wrapper to communicate with MT5
- Dragonfly bus utility for pub/sub
- Tick publisher broadcasting prices
- Candle collector ingesting data

## Phase 2: Risk Shield (Kill Switch)
**Goal:** Standalone Python daemon monitoring equity and stopping MCP if rules are violated.
**Must-haves:**
- Risk daemon independent process
- OS-level kill switch for MCP server
- Hardcoded risk rules

## Phase 3: OpenRouter LLM Orchestration
**Goal:** AI Agents committee using OpenRouter API.
**Must-haves:**
- OpenRouter API client
- Market Analyzer (Owl Alpha)
- News Analyzer (GPT-OSS-120B)
- Trade Executor (Deterministic, no LLM)

## Phase 4: Vector Memory (RAG)
**Goal:** Qdrant integration to save and query historical trade context.
**Must-haves:**
- Embeddings generator
- Performance Coach agent (Qwen3 Next 80B)
- Qdrant integration

## Phase 5: FastAPI + React Dashboard
**Goal:** Ultra-fast WebSockets backend and SPA frontend.
**Must-haves:**
- FastAPI websocket bridge
- React/Vite dashboard
- Live TradingView charts
