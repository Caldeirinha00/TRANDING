---
phase: 2
plan: 5
wave: 1
depends_on: []
files_modified: ["risk/rules.py", "risk/kill_switch.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Risk rules can be evaluated against positions"
    - "Kill switch can terminate the system immediately"
  artifacts:
    - "risk/rules.py exists"
    - "risk/kill_switch.py exists"
---

# Plan 2.5: Risk Rules and Kill Switch

<objective>
Implement the risk logic module and the system-level kill switch that can forcibly terminate the MT5 MCP process if limits are breached.
Purpose: Ensures capital preservation by physically terminating trading capabilities during extreme drawdown.
Output: rules.py and kill_switch.py.
</objective>

<context>
Load for context:
- core/config.py
</context>

<tasks>

<task type="auto">
  <name>Implement Risk Rules</name>
  <files>risk/rules.py</files>
  <action>
    Create a `RiskManager` class.
    Implement methods to check `MAX_DAILY_DRAWDOWN_PCT`, `MAX_TOTAL_DRAWDOWN_PCT`, `MAX_CONCURRENT_POSITIONS`.
    Return boolean (True if safe, False if risk breach).
  </action>
  <verify>python -c "import risk.rules"</verify>
  <done>Syntax is correct and class is implemented</done>
</task>

<task type="auto">
  <name>Implement System Kill Switch</name>
  <files>risk/kill_switch.py</files>
  <action>
    Create a `trigger_kill_switch` function.
    It should log a CRITICAL error and use `psutil` or `os` to forcefully kill the MCP server process (`metatrader-mcp-server`) and Python trading loops.
    Broadcast a fatal alert to DragonflyDB.
  </action>
  <verify>python -c "import risk.kill_switch"</verify>
  <done>Syntax is correct</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Modules compile without errors.
- [ ] Kill switch has system-level termination logic.
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
