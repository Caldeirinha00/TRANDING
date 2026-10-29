//+------------------------------------------------------------------+
//|                                  AutonomousDefenseManager.mqh   |
//|                         Copyright 2026, Institutional Quant Core |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

enum ENUM_SYSTEM_HEALTH
  {
   HEALTH_GREEN  = 0, // Python Online (< 2 falhas de pipe)
   HEALTH_YELLOW = 1, // Falha transitória: Bloqueia novas entradas, gere abertas normal
   HEALTH_ORANGE = 2, // Python Offline prolongado: Aperta Trailing & Break-Even defensivo
   HEALTH_RED    = 3  // Violação do Piso de Capital Diário (3%) -> Liquidação Local
  };

class CAutonomousDefenseManager
  {
private:
   CTrade            m_trade;
   ulong             m_magic;
   int               m_consecutive_ipc_failures;
   datetime          m_last_ipc_success;
   double            m_day_start_equity;
   int               m_current_day_of_year;

public:
                     CAutonomousDefenseManager(const ulong magic)
                     : m_magic(magic),
                       m_consecutive_ipc_failures(0),
                       m_last_ipc_success(TimeCurrent()),
                       m_day_start_equity(AccountInfoDouble(ACCOUNT_EQUITY)),
                       m_current_day_of_year(-1)
     {
      m_trade.SetExpertMagicNumber(magic);
     }

   void              RecordIPCSuccess()
     {
      m_consecutive_ipc_failures = 0;
      m_last_ipc_success = TimeCurrent();
     }

   void              RecordIPCFailure()
     {
      m_consecutive_ipc_failures++;
     }

   ENUM_SYSTEM_HEALTH EvaluateSystemHealth(const double max_daily_dd_pct = 0.029)
     {
      MqlDateTime dt;
      TimeCurrent(dt);
      if(dt.day_of_year != m_current_day_of_year)
        {
         m_current_day_of_year = dt.day_of_year;
         m_day_start_equity    = AccountInfoDouble(ACCOUNT_EQUITY);
        }

      double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
      double daily_dd = (m_day_start_equity - current_equity) / MathMax(1.0, m_day_start_equity);

      if(daily_dd >= max_daily_dd_pct)
         return HEALTH_RED;

      int silence_sec = (int)(TimeCurrent() - m_last_ipc_success);
      if(m_consecutive_ipc_failures == 0 && silence_sec < 300)
         return HEALTH_GREEN;
      if(m_consecutive_ipc_failures <= 2 && silence_sec < 600)
         return HEALTH_YELLOW;

      return HEALTH_ORANGE;
     }

   // Executado a cada tick no OnTick() do EA independentemente do estado do Python
   void              ManageOpenPositionsTick(const string symbol, const double atr_m5)
     {
      ENUM_SYSTEM_HEALTH health = EvaluateSystemHealth();

      if(health == HEALTH_RED)
        {
         for(int i = PositionsTotal() - 1; i >= 0; i--)
           {
            ulong ticket = PositionGetTicket(i);
            if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC) == (long)m_magic)
               m_trade.PositionClose(ticket, 20);
           }
         return;
        }

      // Em modo ORANGE (Python offline), reduz os multiplicadores de BE e Trailing em 35%
      double be_trigger_atr = (health == HEALTH_ORANGE) ? 0.75 * atr_m5 : 1.15 * atr_m5;
      double trail_dist_atr = (health == HEALTH_ORANGE) ? 0.65 * atr_m5 : 1.00 * atr_m5;

      int digits   = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      double bid   = SymbolInfoDouble(symbol, SYMBOL_BID);
      double ask   = SymbolInfoDouble(symbol, SYMBOL_ASK);

      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(!PositionSelectByTicket(ticket))
            continue;
         if(PositionGetInteger(POSITION_MAGIC) != (long)m_magic || PositionGetString(POSITION_SYMBOL) != symbol)
            continue;

         long pos_type     = PositionGetInteger(POSITION_TYPE);
         double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
         double curr_sl    = PositionGetDouble(POSITION_SL);
         double curr_tp    = PositionGetDouble(POSITION_TP);

         if(pos_type == POSITION_TYPE_BUY)
           {
            double profit_dist = bid - open_price;
            double new_sl = curr_sl;

            // 1. Break-Even Institucional (+2 pontos de comissão/slippage buffer)
            if(profit_dist >= be_trigger_atr && curr_sl < open_price)
               new_sl = NormalizeDouble(open_price + 2.0 * point, digits);

            // 2. Trailing Stop Monotónico
            if(profit_dist > trail_dist_atr)
              {
               double candidate_sl = NormalizeDouble(bid - trail_dist_atr, digits);
               if(candidate_sl > new_sl + point)
                  new_sl = candidate_sl;
              }

            if(new_sl > curr_sl + point)
               m_trade.PositionModify(ticket, new_sl, curr_tp);
           }
         else if(pos_type == POSITION_TYPE_SELL)
           {
            double profit_dist = open_price - ask;
            double new_sl = (curr_sl <= 0.0) ? 999999.0 : curr_sl;

            if(profit_dist >= be_trigger_atr && curr_sl > open_price)
               new_sl = NormalizeDouble(open_price - 2.0 * point, digits);

            if(profit_dist > trail_dist_atr)
              {
               double candidate_sl = NormalizeDouble(ask + trail_dist_atr, digits);
               if(candidate_sl < new_sl - point)
                  new_sl = candidate_sl;
              }

            if(new_sl < curr_sl - point || curr_sl <= 0.0)
               m_trade.PositionModify(ticket, new_sl, curr_tp);
           }
        }
     }
  };
