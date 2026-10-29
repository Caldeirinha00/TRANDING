---
phase: 3
plan: 7
wave: 1
depends_on: ["06"]
files_modified: ["agents/llm/openrouter_client.py", "agents/trade_executor.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Can call OpenRouter API via AsyncOpenAI"
    - "Trade executor can place market/pending orders deterministically"
  artifacts:
    - "agents/llm/openrouter_client.py exists"
    - "agents/trade_executor.py exists"
---

# Plan 3.7: OpenRouter Bridge & Deterministic Executor

<objective>
Implement the core LLM client to talk to OpenRouter and the deterministic execution module that places trades based on signals without LLM hallucination risk.
Purpose: Secure the AI connection and ensure trade placement is hard-coded.
Output: openrouter_client.py and trade_executor.py.
</objective>

<context>
Load for context:
- core/config.py
- core/mcp_client.py
</context>

<tasks>

<task type="auto">
  <name>Implement OpenRouter Client</name>
  <files>agents/llm/openrouter_client.py</files>
  <action>
    Create a wrapper using `AsyncOpenAI`.
    Set `base_url="https://openrouter.ai/api/v1"` and use `CONFIG.OPENROUTER_API_KEY`.
    Implement `get_json_analysis(model, prompt)` to force JSON output using OpenRouter's capabilities.
  </action>
  <verify>python -c "import agents.llm.openrouter_client"</verify>
  <done>Client syntax is valid and ready for calls</done>
</task>

<task type="auto">
  <name>Implement Deterministic Trade Executor</name>
  <files>agents/trade_executor.py</files>
  <action>
    Create a `TradeExecutor` class (no LLM).
    Implement methods to execute BUY/SELL orders via `MetaTrader5` python package.
    Ensure strict lot size checks (`CONFIG.MAX_LOT_SIZE`).
  </action>
  <verify>python -c "import agents.trade_executor"</verify>
  <done>Syntax is correct</done>
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
