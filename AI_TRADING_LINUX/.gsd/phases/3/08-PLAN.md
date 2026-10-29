---
phase: 3
plan: 8
wave: 2
depends_on: ["07"]
files_modified: ["agents/market_analyzer.py", "agents/news_analyzer.py", "agents/orchestrator.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Market analyzer uses owl-alpha"
    - "News analyzer uses gpt-oss-120b"
    - "Orchestrator manages the flow and uses Qwen3 for RAG context"
  artifacts:
    - "agents/market_analyzer.py exists"
    - "agents/news_analyzer.py exists"
    - "agents/orchestrator.py exists"
---

# Plan 3.8: AI Analyzer Agents & Orchestrator

<objective>
Implement the specific AI roles using OpenRouter models and the main orchestrator that coordinates the AI committee.
Purpose: This is the brain of the trading system, taking raw data, extracting insights, and deciding on actions.
Output: market_analyzer.py, news_analyzer.py, orchestrator.py.
</objective>

<context>
Load for context:
- agents/llm/openrouter_client.py
- agents/trade_executor.py
</context>

<tasks>

<task type="auto">
  <name>Implement Market Analyzer</name>
  <files>agents/market_analyzer.py</files>
  <action>
    Create `MarketAnalyzer` class.
    Implement method that takes latest candles and uses `openrouter/owl-alpha` to return technical analysis (trend, support, resistance) in JSON.
  </action>
  <verify>python -c "import agents.market_analyzer"</verify>
  <done>Syntax is valid</done>
</task>

<task type="auto">
  <name>Implement News Analyzer</name>
  <files>agents/news_analyzer.py</files>
  <action>
    Create `NewsAnalyzer` class.
    Implement method that takes a news string and uses `openrouter/openai/gpt-oss-120b` to return sentiment (bullish/bearish/neutral) and confidence in JSON.
  </action>
  <verify>python -c "import agents.news_analyzer"</verify>
  <done>Syntax is valid</done>
</task>

<task type="auto">
  <name>Implement Orchestrator</name>
  <files>agents/orchestrator.py</files>
  <action>
    Create `Orchestrator` class using `openrouter/qwen/qwen3-next-80b-a3b-instruct` for final decisions.
    Takes inputs from MarketAnalyzer and NewsAnalyzer.
    If consensus is reached, it triggers `TradeExecutor`.
  </action>
  <verify>python -c "import agents.orchestrator"</verify>
  <done>Orchestrator ties all agents together</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] All three modules import correctly.
- [ ] Orchestrator imports analyzers and executors properly.
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
