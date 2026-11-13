# AI_TRADING V2 (Variant: MQL5 Hybrid)

Architecture: Hybrid MQL5 Expert Advisor (`AI_TRADING_Twin_Final.mq5`) with Windows Named Pipe IPC to a Python Fast Veto Engine and LLM Multi-Agent Copilot.

---

## 🏛️ Repository Architecture & Branches Guide

| Branch | Architecture / Variant | Description | Target Environment |
| :--- | :--- | :--- | :--- |
| **`main`** | **AI_TRADING_NEW (Current)** | 100% Python H4/D1 Quantitative System with Triple Barrier & Meta-Labeling | Windows 11 / MT5 Python API |
| **`variant/mql5-hybrid`** | **AI_TRADING V2** | Hybrid MQL5 Expert Advisor + IPC Named Pipe Server + Fast Veto Python Copilot | Windows 11 / MT5 Terminal |
| **`variant/linux`** | **AI_TRADING_LINUX** | Headless Linux Docker architecture for asynchronous agent microservices | Linux / Docker / Docker-Compose |
| **`variant/scalper`** | **AI_TRADING_SCALPPER** | Autonomous parameter optimization, fast alpha discovery, and M1 scalper research | Windows / MetaTrader Strategy Tester |
| **`v1-monorepo-history`** | **Historical Monorepo** | Full linear commit history of initial development phases 1-5 | Reference & Audit Trail |

---

## Architecture Overview

1. **MQL5 Layer (`mql5/`):**
   - `AI_TRADING_Twin_Final.mq5`: Primary Expert Advisor executing on MetaTrader 5 terminal.
   - `Include/AI_TRADING_IPC_Pipe.mqh`: Low-latency IPC named pipe client connecting to Python daemon.
   - `AutonomousDefenseManager.mqh`: Client-side drawdown controls and slippage protection.
   - `SmartOrderRouter.mqh`: Execution routing and partial TP handling.

2. **Python Copilot Layer (`agents/` & `core/`):**
   - `agents/fast_veto_engine.py`: High-speed binary classifier validating MQL5 trade intent in <10ms.
   - `agents/ipc_pipe_server.py`: Windows Named Pipe server communicating bidirectionally with MT5.
   - `agents/qdrant_rag_modulator.py`: Vector memory for historical market regime retrieval.

3. **Risk Management (`risk/`):**
   - Strict adherence to ESMA capital preservation and max daily loss limits.

## Setup

```powershell
cp .env.example .env
# Edit .env with your broker demo credentials and LAN hosts
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python start_all.py
```
