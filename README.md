# AI_TRADING_SCALPPER (Variant: Autonomous Scalper & Quant Research)

Research scripts, fast alpha / sweetspot search, parameter optimization, and M1 scalper prototypes.

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

## Key Modules

- `AI_TRADING_Twin.mq5`, `AI_TRADING_Twin_Final.mq5`: High-frequency scalping EAs.
- `fast_alpha.py`: Fast statistical alpha feature discovery.
- `fast_sweetspot.py`: Parameter sweetspot optimizer.
- `quant_search.py`: Quantitative grid search.
- `optimize_autonomous.py`: Autonomous optimizer script.
- `orchestrator.py`: Multi-symbol research orchestrator.
