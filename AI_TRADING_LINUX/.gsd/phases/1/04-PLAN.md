---
phase: 1
plan: 4
wave: 4
depends_on: ["03"]
files_modified: ["core/tick_publisher.py", "core/candle_collector.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Ticks are streamed to DragonflyDB"
    - "Candles are ingested into QuestDB using ILP"
  artifacts:
    - "core/tick_publisher.py exists"
    - "core/candle_collector.py exists"
---

# Plan 1.4: High-Speed Data Ingestion Loops

<objective>
Implement the continuous background processes that pull data from MT5 and push it to the respective databases.
Purpose: Feeds the system with live financial data (ticks to Dragonfly, candles to QuestDB).
Output: Two standalone asyncio scripts.
</objective>

<context>
Load for context:
- .gsd/phases/1/03-PLAN.md
- core/mcp_client.py
- core/dragonfly_bus.py
</context>

<tasks>

<task type="auto">
  <name>Implement Tick Publisher</name>
  <files>core/tick_publisher.py</files>
  <action>
    Create asyncio loop polling `get_symbol_price` every 200ms.
    Publish result to DragonflyDB on channel `ticks:{symbol}`.
  </action>
  <verify>python -c "import core.tick_publisher"</verify>
  <done>Script handles async loop correctly</done>
</task>

<task type="auto">
  <name>Implement Candle Collector</name>
  <files>core/candle_collector.py</files>
  <action>
    Create asyncio loop fetching new 1M candles.
    Insert into QuestDB using socket connection (Influx Line Protocol) on port 9009 for maximum speed.
  </action>
  <verify>python -c "import core.candle_collector"</verify>
  <done>Script handles ILP sockets and async loop correctly</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] Tick publisher is ready to broadcast to DragonflyDB
- [ ] Candle collector is ready to insert into QuestDB
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
