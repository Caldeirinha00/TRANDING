---
phase: 2
plan: 6
wave: 2
depends_on: ["05"]
files_modified: ["risk/risk_daemon.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Daemon continuously monitors open positions via DragonflyDB/MCP"
    - "Daemon can trigger kill switch on breach"
  artifacts:
    - "risk/risk_daemon.py exists"
---

# Plan 2.6: The Risk Daemon

<objective>
Implement the independent risk daemon process that monitors the system and executes the kill switch.
Purpose: Constant vigilance of account health and position limits.
Output: risk_daemon.py.
</objective>

<context>
Load for context:
- core/config.py
- risk/rules.py
- risk/kill_switch.py
- core/dragonfly_bus.py
</context>

<tasks>

<task type="auto">
  <name>Implement Risk Daemon Loop</name>
  <files>risk/risk_daemon.py</files>
  <action>
    Create asyncio loop checking MT5 account info and positions periodically (e.g., every 1 second).
    Listen to `system:risk` channel on DragonflyDB for emergency broadcasts.
    If `RiskManager` detects a breach, invoke `trigger_kill_switch`.
  </action>
  <verify>python -c "import risk.risk_daemon"</verify>
  <done>Daemon logic is syntactically sound</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Daemon can pull MT5 account state
- [ ] Daemon imports kill switch properly
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
