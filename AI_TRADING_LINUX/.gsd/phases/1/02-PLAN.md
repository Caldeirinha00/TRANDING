---
phase: 1
plan: 2
wave: 2
depends_on: ["01"]
files_modified: ["pyproject.toml", "core/config.py"]
autonomous: true
user_setup: []

must_haves:
  truths:
    - "Python environment is properly configured"
  artifacts:
    - "pyproject.toml exists with dependencies"
    - "core/config.py validates environment variables"
---

# Plan 1.2: Python Project Setup & Configuration

<objective>
Setup the Python environment for the trading platform, utilizing uv/poetry for dependencies and pydantic for configuration.
Purpose: We need a structured way to load configurations and run async processes.
Output: pyproject.toml and the central configuration module.
</objective>

<context>
Load for context:
- .gsd/ROADMAP.md
- .gsd/phases/1/01-PLAN.md
</context>

<tasks>

<task type="auto">
  <name>Initialize pyproject.toml</name>
  <files>pyproject.toml</files>
  <action>
    Create pyproject.toml defining the project `ai-trading`.
    Include dependencies: asyncio, aioredis (for Dragonfly), MetaTrader5, pydantic, pydantic-settings, structlog.
  </action>
  <verify>cat pyproject.toml</verify>
  <done>pyproject.toml has all required dependencies</done>
</task>

<task type="auto">
  <name>Create config module</name>
  <files>core/config.py</files>
  <action>
    Create core/config.py.
    Use `pydantic_settings.BaseSettings` to load:
    - LAN_IP
    - OPENROUTER_API_KEY
    - SYMBOLS (default ["EURUSD"])
    - MT5 configuration
  </action>
  <verify>python -c "import core.config"</verify>
  <done>Config module can be imported without syntax errors</done>
</task>

</tasks>

<verification>
After all tasks, verify:
- [ ] pyproject.toml contains all necessary libraries
- [ ] core/config.py correctly maps to the environment variables
</verification>

<success_criteria>
- [ ] All tasks verified
- [ ] Must-haves confirmed
</success_criteria>
