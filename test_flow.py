import pytest
import asyncio
import logging
import sys

from agents.orchestrator import orchestrator
from core.mcp_client import mcp_client

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

@pytest.mark.asyncio
async def test_workflow():
    print("--- STARTING WORKFLOW TEST ---")
    symbol = "EURUSD"
    
    # Ensure MCP MT5 connection
    print("1. Connecting to MT5...")
    if not mcp_client.connect():
        print("ERROR: Could not connect to MT5.")
        return

    print(f"2. Triggering Orchestrator for {symbol}...")
    success = await orchestrator.evaluate_and_trade(symbol)
    
    if success:
        print("--- WORKFLOW SUCCESSFUL: Trade was executed! ---")
    else:
        print("--- WORKFLOW FINISHED: No trade was executed (either LLM rejected the trade due to risk, or an error occurred). Check logs for details. ---")

if __name__ == "__main__":
    asyncio.run(test_workflow())
