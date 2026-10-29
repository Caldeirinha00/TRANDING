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

   // Proteção de Weekend Gap (Fase 4.2): Fecho preventivo compulsório na Sexta-feira às 20:55 UTC
   void              CheckWeekendProtection(const string symbol)
     {
      MqlDateTime dt;
      TimeToStruct(TimeGMT(), dt);
      if(dt.day_of_week == 5 && (dt.hour > 20 || (dt.hour == 20 && dt.min >= 55)))
        {
         for(int i = PositionsTotal() - 1; i >= 0; i--)
           {
            ulong ticket = PositionGetTicket(i);
            if(!PositionSelectByTicket(ticket))
               continue;
            if(PositionGetInteger(POSITION_MAGIC) == (long)m_magic && PositionGetString(POSITION_SYMBOL) == symbol)
              {
               PrintFormat("[DEFESA INSTITUCIONAL] Weekend Gap Shield: Fecho preventivo posicao #%I64u (%02d:%02d UTC).",
                           ticket, dt.hour, dt.min);
               m_trade.PositionClose(ticket, 20);
              }
           }
        }
     }

   // Executado a cada tick no OnTick() do EA independentemente do estado do Python
   void              ManageOpenPositionsTick(const string symbol, const double atr_m5)
     {
      // 0. Proteção de Weekend Gap ativa
      CheckWeekendProtection(symbol);

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
         double vol        = PositionGetDouble(POSITION_VOLUME);
         string comment    = PositionGetString(POSITION_COMMENT);
         bool partial_taken = (StringFind(comment, "[PARTIAL_TAKEN]") >= 0);

         if(pos_type == POSITION_TYPE_BUY)
           {
            double profit_dist = bid - open_price;
            double sl_dist     = (open_price > curr_sl && curr_sl > 0.0) ? (open_price - curr_sl) : (1.5 * atr_m5);
            double new_sl      = curr_sl;

            // 1. Break-Even Institucional (Ativado apenas a 1.4R para evitar o efeito assimétrico de ruído M5)
            if(profit_dist >= 1.4 * sl_dist && curr_sl < open_price)
               new_sl = NormalizeDouble(open_price + 2.0 * point, digits);

            // 2. Trailing TP Parcial: Fecha 50% ao atingir 1.5R e remove teto de TP para surfar a tendência
            if(!partial_taken && profit_dist >= 1.5 * sl_dist && vol >= 0.02)
              {
               double close_vol = NormalizeDouble(MathFloor((vol * 0.5) / 0.01) * 0.01, 2);
               if(close_vol >= 0.01 && m_trade.PositionClosePartial(ticket, close_vol, 20))
                 {
                  partial_taken = true;
                  new_sl = NormalizeDouble(open_price + 0.5 * sl_dist, digits);
                  curr_tp = 0.0;
                 }
              }

            // 3. Trailing Stop Monotónico (0.70 ATR para posição parcial restante, 1.0 ATR normal)
            double effective_trail = partial_taken ? 0.70 * atr_m5 : trail_dist_atr;
            if(profit_dist > effective_trail)
              {
               double candidate_sl = NormalizeDouble(bid - effective_trail, digits);
               if(candidate_sl > new_sl + point)
                  new_sl = candidate_sl;
              }

            if(new_sl > curr_sl + point || curr_tp != PositionGetDouble(POSITION_TP))
               m_trade.PositionModify(ticket, new_sl, curr_tp);
           }
         else if(pos_type == POSITION_TYPE_SELL)
           {
            double profit_dist = open_price - ask;
            double sl_dist     = (curr_sl > open_price) ? (curr_sl - open_price) : (1.5 * atr_m5);
            double new_sl      = (curr_sl <= 0.0) ? 999999.0 : curr_sl;

            // 1. Break-Even Institucional (Ativado apenas a 1.4R para evitar o efeito assimétrico de ruído M5)
            if(profit_dist >= 1.4 * sl_dist && curr_sl > open_price)
               new_sl = NormalizeDouble(open_price - 2.0 * point, digits);

            // 2. Trailing TP Parcial: Fecha 50% ao atingir 1.5R e remove teto de TP
            if(!partial_taken && profit_dist >= 1.5 * sl_dist && vol >= 0.02)
              {
               double close_vol = NormalizeDouble(MathFloor((vol * 0.5) / 0.01) * 0.01, 2);
               if(close_vol >= 0.01 && m_trade.PositionClosePartial(ticket, close_vol, 20))
                 {
                  partial_taken = true;
                  new_sl = NormalizeDouble(open_price - 0.5 * sl_dist, digits);
                  curr_tp = 0.0;
                 }
              }

            // 3. Trailing Stop Monotónico
            double effective_trail = partial_taken ? 0.70 * atr_m5 : trail_dist_atr;
            if(profit_dist > effective_trail)
              {
               double candidate_sl = NormalizeDouble(ask + effective_trail, digits);
               if(candidate_sl < new_sl - point)
                  new_sl = candidate_sl;
              }

            if(new_sl < curr_sl - point || curr_tp != PositionGetDouble(POSITION_TP))
               m_trade.PositionModify(ticket, new_sl, curr_tp);
           }
        }
     }
  };
