---
phase: 1
plan: 1
wave: 1
depends_on: []
files_modified: ["docker-compose.yml", ".env.example"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Infrastructure containers start successfully"
  artifacts:
    - "docker-compose.yml exists with QuestDB, DragonflyDB, Qdrant"
---

# Plan 1.1: Setup High-Speed Infrastructure

<objective>
Create the Docker compose file containing the fastest available time-series database (QuestDB), message broker (DragonflyDB), and vector memory (Qdrant).
Purpose: This is the foundation of the high-speed data pipeline for financial ticks.
Output: A docker-compose.yml and .env.example file.
</objective>

<context>
Load for context:
- .gsd/ROADMAP.md
</context>

<tasks>

<task type="auto">
  <name>Create docker-compose.yml</name>
  <files>docker-compose.yml</files>
  <action>
    Create docker-compose.yml defining three services:
    1. questdb: image `questdb/questdb:latest`, expose ports 9000, 8812, 9009.
    2. dragonfly: image `docker.dragonflydb.io/dragonflydb/dragonfly`, expose port 6379, set memlock limit to -1, command `dragonfly --maxmemory=2g`.
    3. qdrant: image `qdrant/qdrant:latest`, expose port 6333.
    Use Docker volumes for all three.
  </action>
  <verify>docker-compose config</verify>
  <done>Docker compose configuration is valid</done>
</task>

<task type="auto">
  <name>Create .env.example</name>
  <files>.env.example</files>
  <action>
    Create the environment variables template. Include:
    LAN_IP (for DragonflyDB and QuestDB connection)
    OPENROUTER_API_KEY
  </action>
  <verify>cat .env.example</verify>
  <done>.env.example contains necessary variables</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] docker-compose.yml exists and has the correct services
- [ ] .env.example is populated
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
