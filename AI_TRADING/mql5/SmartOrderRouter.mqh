//+------------------------------------------------------------------+
//| SmartOrderRouter.mqh - Execução Institucional Limit/IOC          |
//|                         Copyright 2026, Institutional Quant Core |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

class CSmartOrderRouter
  {
private:
   CTrade            m_trade;
   ulong             m_magic;

public:
                     CSmartOrderRouter(const ulong magic) : m_magic(magic)
     {
      m_trade.SetExpertMagicNumber(magic);
      m_trade.SetAsyncMode(false);
     }

   bool              ExecuteSmartEntry(const string symbol,
                                       const int direction,
                                       const double lots,
                                       const double atr_m5,
                                       const double sl_price,
                                       const double tp_price,
                                       const int max_spread_pts = 18)
     {
      int current_spread = (int)SymbolInfoInteger(symbol, SYMBOL_SPREAD);
      if(current_spread > max_spread_pts)
        {
         PrintFormat("[SOR] Entrada abortada: Spread atual (%d pts) > Máximo (%d pts)",
                     current_spread, max_spread_pts);
         return false;
        }

      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      int digits   = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      double bid   = SymbolInfoDouble(symbol, SYMBOL_BID);
      double ask   = SymbolInfoDouble(symbol, SYMBOL_ASK);

      // Desvio máximo dinâmico = 12% do ATR M5 (limitado entre 2 e 15 pontos)
      ulong dyn_dev_pts = (ulong)MathMax(2.0, MathMin(15.0, MathRound((0.12 * atr_m5) / point)));
      m_trade.SetDeviationInPoints(dyn_dev_pts);

      // Seleciona o modo de preenchimento ótimo suportado pela corretora (IOC preferencial)
      uint filling = (uint)SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
      if((filling & SYMBOL_FILLING_IOC) != 0)
         m_trade.SetTypeFilling(ORDER_FILLING_IOC);
      else if((filling & SYMBOL_FILLING_FOK) != 0)
         m_trade.SetTypeFilling(ORDER_FILLING_FOK);
      else
         m_trade.SetTypeFilling(ORDER_FILLING_RETURN);

      ENUM_ORDER_TYPE order_type = (direction > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      double exec_price          = (direction > 0) ? ask : bid;

      return m_trade.PositionOpen(symbol,
                                  order_type,
                                  lots,
                                  NormalizeDouble(exec_price, digits),
                                  NormalizeDouble(sl_price, digits),
                                  NormalizeDouble(tp_price, digits),
                                  "AI_TWIN_SOR");
     }
  };
