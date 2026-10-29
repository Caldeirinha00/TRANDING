import sys
import logging

def check_all():
    try:
        from core.config import CONFIG
        print("CONFIG:", CONFIG.MT5_SERVER)
        
        from core.mcp_client import mcp_client
        print("MCP_CLIENT OK")
        
        from agents.llm.mistral_client import mistral_client
        print("MISTRAL CLIENT OK")
        
        from agents.market_analyzer import market_analyzer
        print("MARKET_ANALYZER OK")
        
        from agents.news_analyzer import news_analyzer
        print("NEWS_ANALYZER OK")
        
        from agents.trade_executor import trade_executor
        print("TRADE_EXECUTOR OK")
        
        from agents.orchestrator import orchestrator
        print("ORCHESTRATOR OK")
        
        from risk.rules import RiskManager
        print("RISK_RULES OK")
        
        from risk.risk_daemon import risk_daemon_loop
        print("RISK_DAEMON OK")
        
        print("ALL IMPORTS SUCCESSFUL")
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_all()
