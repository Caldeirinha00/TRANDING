# AI_TRADING_NEW

Unified 100% Python quantitative trading system targeting **H4 / D1** horizons with deterministic primary signals and secondary **Meta-Labeling** (Marcos López de Prado), engineered specifically for a **1,000 EUR** account under **ESMA Retail (1:30 / 1:20)** leverage constraints on **MetaTrader 5 (PepperstoneUK-Demo)**.

---

## 🏛️ Repository Architecture & Branches Guide

This repository (`TRANDING`) organizes all system variants and evolutionary phases into dedicated branches:

| Branch | Architecture / Variant | Description | Target Environment |
| :--- | :--- | :--- | :--- |
| **`main`** | **AI_TRADING_NEW (Current)** | 100% Python H4/D1 Quantitative System with Triple Barrier & Meta-Labeling | Windows 11 / MT5 Python API |
| **`variant/mql5-hybrid`** | **AI_TRADING V2** | Hybrid MQL5 Expert Advisor + IPC Named Pipe Server + Fast Veto Python Copilot | Windows 11 / MT5 Terminal |
| **`variant/linux`** | **AI_TRADING_LINUX** | Headless Linux Docker architecture for asynchronous agent microservices | Linux / Docker / Docker-Compose |
| **`variant/scalper`** | **AI_TRADING_SCALPPER** | Autonomous parameter optimization, fast alpha discovery, and M1 scalper research | Windows / MetaTrader Strategy Tester |
| **`v1-monorepo-history`** | **Historical Monorepo** | Full linear commit history of initial development phases 1-5 | Reference & Audit Trail |

---

## Core Architecture & Risk Rules

- **Capital Base:** `1,000.00 EUR` (ESMA 1:30 majors / 1:20 crosses).
- **Approved Universe (8 pairs):** `EURUSD`, `USDJPY`, `AUDUSD`, `NZDUSD`, `USDCAD`, `USDCHF`, `GBPUSD`, `AUDJPY`.
- **Forbidden under 3,000 EUR:** `XAUUSD`, `XAGUSD`, `US500`, `NAS100`, `GER40` (minimum `0.01` lot structural risk exceeds `1.0%` equity).
- **Risk Limits:**
  - Max single trade risk: `1.00%` of equity (`<= 10.00 EUR`).
  - Max open Portfolio Heat (non-BE positions): `2.10%` of equity (`<= 21.00 EUR`).
  - Max active risk positions: `3` concurrent non-breakeven positions.
  - Currency Bucket Guard: at most 1 active-risk directional exposure per currency.
  - Hostile Swap Veto: rejects entries where projected 4-day swap (including Wednesday triple rollover) exceeds `15%` of `1R`.
  - Tiered Watchdog Kill Switches: `2.85%` daily drawdown / `19.50%` total drawdown, plus `< 5s` server-side naked-position protection.

## Setup & Verification

```powershell
Set-Location "C:\Users\mob70\Documents\AI_TRADING_NEW"
.\.venv\Scripts\Activate.ps1
pytest -v
python scripts\01_profile_broker_symbols.py
```
