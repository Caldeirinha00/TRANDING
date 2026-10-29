---
phase: 1
plan: 3
wave: 3
depends_on: ["02"]
files_modified: ["core/mcp_client.py", "core/dragonfly_bus.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "MCP wrapper can connect to MT5 server"
    - "Pub/sub messages can be passed via DragonflyDB"
  artifacts:
    - "core/mcp_client.py exists"
    - "core/dragonfly_bus.py exists"
---

# Plan 1.3: Core Client Wrappers

<objective>
Build the asynchronous client wrappers to abstract communication with the MT5 MCP Server and the DragonflyDB pub/sub system.
Purpose: These wrappers provide the building blocks for the data ingestion loops.
Output: mcp_client.py and dragonfly_bus.py.
</objective>

<context>
Load for context:
- .gsd/phases/1/02-PLAN.md
- core/config.py
</context>

<tasks>

<task type="auto">
  <name>Implement MCP Client Wrapper</name>
  <files>core/mcp_client.py</files>
  <action>
    Create a wrapper for communicating with the MCP server.
    Implement methods for `get_price`, `get_candles_latest`.
    Handle timeouts and retries robustness.
  </action>
  <verify>python -c "import core.mcp_client"</verify>
  <done>MCP client syntax is correct</done>
</task>

<task type="auto">
  <name>Implement Dragonfly Bus</name>
  <files>core/dragonfly_bus.py</files>
  <action>
    Create an aioredis-based wrapper for DragonflyDB.
    Implement async `publish(channel, message)` and `subscribe(channel, callback)`.
    Use the LAN_IP from config for connection.
  </action>
  <verify>python -c "import core.dragonfly_bus"</verify>
  <done>Dragonfly wrapper syntax is correct</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Both clients handle their connection gracefully.
- [ ] The modules are importable.
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
