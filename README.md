# AI_TRADING_LINUX (Variant: Headless Linux Microservices)

Headless Linux containerized architecture using Docker, Dragonfly, QuestDB, Qdrant, and Python multi-agent microservices.

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

## Instructions

For full setup and running instructions on Linux/Docker, please consult [`COMO_RODAR_LINUX.md`](COMO_RODAR_LINUX.md).

### Quick Start

```bash
cp .env.example .env
docker compose up -d
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-linux.txt
python3 start_all.py
```
