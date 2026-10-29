//+------------------------------------------------------------------+
//|                                              AI_TRADING_Twin.mq5 |
//|                                  Copyright 2026, AI_TRADING Team |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, AI_TRADING Team"
#property link      ""
#property version   "1.00"

#include <Trade\Trade.mqh>
#include "Indicators_Lib.mqh"

input double InpRiskPercent = 0.5; // Fixed Risk Percent per trade for WFO

// 1.5 Semântica temporal do ciclo
bool NewM5Bar()
{
   static datetime last = 0;
   datetime t = iTime(_Symbol, PERIOD_M5, 0);
   if(t == 0 || t == last) return false;
   last = t;
   return true;
}

//===================================================================
// EA EVENTS
//===================================================================
int OnInit()
{
   // Footprint: FileWrite em JSONL no single-run
   file_handle = FileOpen("AI_TRADING_Footprint.jsonl", FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(file_handle != INVALID_HANDLE) FileClose(file_handle);
}

void OnTick()
{
   // Gerir Break-Even Institucional e Trailing Stop Monótono
   ManageOpenPositions();
   
   if(NewM5Bar())
   {
      EvaluateAndTrade();
   }
}

//===================================================================
// WFO Custom Fitness (Stage 3)
//===================================================================
double OnTester()
{
   double net_profit = TesterStatistics(STAT_PROFIT);
   double max_dd_pct = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   int total_trades = (int)TesterStatistics(STAT_TRADES);
   double profit_factor = TesterStatistics(STAT_PROFIT_FACTOR);
   
   // Hard Cutoffs
   if(total_trades < 60 || net_profit <= 0 || max_dd_pct > 20.0 || profit_factor < 1.2)
      return 0.0;
      
   double dd_penalty = MathPow(max_dd_pct / 10.0, 2.0);
   double avg_trade_profit = net_profit / total_trades;
   
   return (avg_trade_profit * MathSqrt(total_trades) * profit_factor) / (1.0 + dd_penalty);
}

void EvaluateAndTrade()
{
   MqlRates m5[], h1[];
   // 1.4 Alinhamento multi-timeframe estrito: start=1
   if(CopyRates(_Symbol, PERIOD_M5, 1, 100, m5) != 100) return;
   if(CopyRates(_Symbol, PERIOD_H1, 1, 100, h1) != 100) return;
   
   // Arrays de closes puros para as EMAs
   double h1_closes[], m5_closes[];
   ArrayResize(h1_closes, 100);
   ArrayResize(m5_closes, 100);
   for(int i=0; i<100; i++)
   {
      h1_closes[i] = h1[i].close;
      m5_closes[i] = m5[i].close;
   }
   
   // 1. FILTRO DE TENDÊNCIA (H1)
   double h1_ema20 = PyEMA(h1_closes, InpH1_EMA_Fast);
   double h1_ema50 = PyEMA(h1_closes, InpH1_EMA_Slow);
   double h1_adx = PyADX(h1, InpADX_Period);
   
   // 2. FILTRO DE VOLATILIDADE (M5)
   double m5_atr = PyATR(m5, InpATR_Period);
   double m5_atr_series[];
   PyATRSeries(m5, InpATR_Period, InpATR_Median_Lookback, m5_atr_series);
   double m5_atr_median = ArraySize(m5_atr_series) > 0 ? NpMedian(m5_atr_series) : m5_atr;
   
   double m5_ema20 = PyEMA(m5_closes, 20);
   
   string direction = "null";
   string setup_type = "NONE";
   string reason_code = "VALID";
   
   bool reject = false;
   
   // Gates na ordem canónica exata do Python para diff de logs perfeito
   if(h1_adx < InpADX_Min)
   {
      reason_code = "ADX_LOW";
      reject = true;
   }
   
   bool trend_bullish = h1_ema20 > h1_ema50;
   bool trend_bearish = h1_ema20 < h1_ema50;
   
   if(!reject)
   {
      if(m5_atr < (m5_atr_median * InpATR_MinMult))
      {
         reason_code = "ATR_LOW";
         reject = true;
      }
      else if(m5_atr > (m5_atr_median * InpATR_MaxMult))
      {
         reason_code = "ATR_HIGH";
         reject = true;
      }
   }
   
   MqlRates last_candle = m5[99];
   if(!reject)
   {
      // 3. GATILHO DE ENTRADA (M5 Pullback)
      if(trend_bullish)
      {
         if(last_candle.low <= m5_ema20 && last_candle.close > m5_ema20)
         {
            direction = "\"BUY\"";
            setup_type = "H1_BULL_PULLBACK_M5";
         }
         else
         {
            reason_code = "NO_BULL_PULLBACK";
            reject = true;
         }
      }
      else if(trend_bearish)
      {
         if(last_candle.high >= m5_ema20 && last_candle.close < m5_ema20)
         {
            direction = "\"SELL\"";
            setup_type = "H1_BEAR_PULLBACK_M5";
         }
         else
         {
            reason_code = "NO_BEAR_PULLBACK";
            reject = true;
         }
      }
      else
      {
         reason_code = "NO_TREND";
         reject = true;
      }
   }
   
   string direction_str = (direction == "null") ? "null" : direction;
   string json = StringFormat(
      "{\"bar_time_m5\": %d, \"h1_ctx_bar_time\": %d, \"symbol\": \"%s\", \"direction\": %s, \"setup_type\": \"%s\", \"h1_adx\": %.12f, \"m5_atr\": %.12f, \"m5_atr_median\": %.12f, \"m5_ema20\": %.12f, \"trigger_open\": %.5f, \"trigger_high\": %.5f, \"trigger_low\": %.5f, \"trigger_close\": %.5f, \"vetoed_by_llm\": false, \"reason_code\": \"%s\"}",
      last_candle.time, h1[99].time, _Symbol, direction_str, setup_type,
      h1_adx, m5_atr, m5_atr_median, m5_ema20,
      last_candle.open, last_candle.high, last_candle.low, last_candle.close,
      reason_code
   );
      
   if(file_handle != INVALID_HANDLE)
   {
      FileWrite(file_handle, json);
   }
   if(!reject && direction != "null")
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      
      double open_price = (direction == "\"BUY\"") ? ask : bid;
      double sl_init = 0.0;
      double tp_init = 0.0;
      
      if(direction == "\"BUY\"")
      {
         sl_init = MathMin(last_candle.low, m5_ema20) - (m5_atr * InpSL_ATR_Buffer);
         tp_init = open_price + InpTP_RR * (open_price - sl_init);
      }
      else
      {
         sl_init = MathMax(last_candle.high, m5_ema20) + (m5_atr * InpSL_ATR_Buffer);
         tp_init = open_price - InpTP_RR * (sl_init - open_price);
      }
      
      // Sizing Base
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double risk_money = balance * (InpRiskPercent / 100.0);
      
      double sl_distance_ticks = MathAbs(open_price - sl_init) / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double money_lost_per_lot = sl_distance_ticks * tick_value;
      
      if(money_lost_per_lot <= 0.0) return; // Guard: Divisão por Zero
      
      double raw_volume = risk_money / money_lost_per_lot;
      
      double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      double vol_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double vol_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      
      // Truncar rigoroso institucional (MathFloor) para NUNCA exceder InpRiskPercent
      double volume = MathFloor(raw_volume / vol_step) * vol_step;
      volume = NormalizeDouble(volume, (int)MathRound(-MathLog10(vol_step)));
      
      // Rejeita o trade institucionalmente se o risco mínimo quebrar o teto InpRiskPercent
      if(volume < vol_min)
      {
         // Aqui seria logado o reason_code: "VOLUME_BELOW_MIN" no footprint Stage 1 estendido
         return; 
      }
      if(volume > vol_max) volume = vol_max;
      
      if(direction == "\"BUY\"")
         trade.Buy(volume, _Symbol, ask, sl_init, tp_init, "AI_WFO_B");
      else
         trade.Sell(volume, _Symbol, bid, sl_init, tp_init, "AI_WFO_S");
   }

//// =========================================================
// INPUTS & GLOBAIS
// =========================================================
input double InpRiskPercent = 1.0;
input double InpCommissionPerLotRoundTurn = 7.0; // Espelha o 7.0 do orchestrator.py

// --- WFO OPTIMIZATION INPUTS ---
sinput string WFO_Params = "--- Strategy Parameters ---";
input int InpH1_EMA_Fast = 20;
input int InpH1_EMA_Slow = 50;
input int InpADX_Period = 14;
input double InpADX_Min = 20.0;

input int InpATR_Period = 14;
input int InpATR_Median_Lookback = 50;
input double InpATR_MinMult = 0.8;
input double InpATR_MaxMult = 2.5;

input double InpSL_ATR_Buffer = 0.5;
input double InpTP_RR = 2.0;

int stg2_handle = INVALID_HANDLE;

int OnInit()
{
   if(!MQLInfoInteger(MQL_OPTIMIZATION) && !MQLInfoInteger(MQL_TESTER))
   {
      stg2_handle = FileOpen("AI_TRADING_Stage2_Footprint.jsonl", FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
      if(stg2_handle != INVALID_HANDLE)
      {
         FileSeek(stg2_handle, 0, SEEK_END);
      }
   }
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(stg2_handle != INVALID_HANDLE)
   {
      FileClose(stg2_handle);
   }
}

void LogStage2Footprint(ulong ticket, string event_type, string direction, double volume, double be_level, double new_sl, double comm_points, double m15_atr, int stops_blocked, int freeze_blocked)
{
   if(stg2_handle == INVALID_HANDLE) return;
   if(MQLInfoInteger(MQL_OPTIMIZATION)) return;
   
   string json = StringFormat(
      "{\"ticket\": %d, \"event_type\": \"%s\", \"direction\": \"%s\", \"volume\": %.2f, \"be_level\": %.5f, \"new_sl\": %.5f, \"commission_points_used\": %.5f, \"m15_atr_used\": %.5f, \"stops_level_blocked_count\": %d, \"freeze_level_blocked_count\": %d}",
      ticket, event_type, direction, volume, be_level, new_sl, comm_points, m15_atr, stops_blocked, freeze_blocked
   );
   
   FileWrite(stg2_handle, json);
   FileFlush(stg2_handle);
}

// =========================================================
// STAGE 2: EXECUTION MANAGEMENT (BE & TRAILING)
// =========================================================

struct PositionState {
   ulong ticket;
   int blocked_stops;
   int blocked_freeze;
};
PositionState g_states[];

int GetStateIndex(ulong ticket) {
   for(int i=0; i<ArraySize(g_states); i++) {
      if(g_states[i].ticket == ticket) return i;
   }
   int size = ArraySize(g_states);
   ArrayResize(g_states, size+1);
   g_states[size].ticket = ticket;
   g_states[size].blocked_stops = 0;
   g_states[size].blocked_freeze = 0;
   return size;
}

double GetM15ATR_Cached()
{
   // Single-symbol cache scope: OK for tester e monopar live.
   static datetime cached_bar = 0;
   static double cached_atr = -1.0;
   
   datetime cur = iTime(_Symbol, PERIOD_M15, 1);
   if(cur != cached_bar || cached_atr < 0.0)
   {
      MqlRates m15[];
      if(CopyRates(_Symbol, PERIOD_M15, 1, 100, m15) == 100)
      {
         cached_atr = PyATR(m15, 14); // Wilder RMA igual ao Python
         cached_bar = cur;
      }
      else
      {
         return -1.0; // Falha na extração, sentinela para abortar proteção
      }
   }
   return cached_atr;
}

void ManageOpenPositions()
{
   if(!PositionsTotal()) return;
   
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double stops_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
   double freeze_level = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * point;
   
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread = (ask - bid);
   
   double m15_atr = GetM15ATR_Cached();
   if(m15_atr < 0.0) return; // Aborta BE silenciosamente, faltam dados
   double buffer_vol = m15_atr * 0.15;
   
   MqlRates m5[];
   if(CopyRates(_Symbol, PERIOD_M5, 1, 100, m5) != 100) return;
   double m5_atr = PyATR(m5, 14);
   
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      
      int state_idx = GetStateIndex(ticket);
      
      long type = PositionGetInteger(POSITION_TYPE);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      double current_profit = PositionGetDouble(POSITION_PROFIT);
      
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double comm_price_offset = 0;
      if(tick_value > 0)
      {
         comm_price_offset = (InpCommissionPerLotRoundTurn * tick_size) / tick_value;
      }
      
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double target_1r_profit = balance * (InpRiskPercent / 100.0);
      
      double new_sl = current_sl;
      bool modify = false;
      string event = "";
      
      if(type == POSITION_TYPE_BUY)
      {
         // FREEZE = Proíbe modificação se o preço estiver perto do SL ou TP atual
         if (MathAbs(bid - current_sl) <= freeze_level || (current_tp > 0 && MathAbs(bid - current_tp) <= freeze_level))
         {
            g_states[state_idx].blocked_freeze++;
            event = "BLOCKED_FREEZE";
         }
         else
         {
            double be_level = open_price + spread + comm_price_offset + buffer_vol;
            
            if(current_sl < be_level && current_profit >= target_1r_profit)
            {
               // STOPS = Proíbe NOVO nível se preço estiver muito perto dele
               if(bid - be_level >= stops_level)
               {
                  new_sl = be_level;
                  modify = true;
                  event = "BE_SET";
               }
               else {
                  g_states[state_idx].blocked_stops++;
                  event = "BLOCKED_STOPS";
               }
            }
            else if (current_sl >= be_level) // In BE, Trailing
            {
               double trail_candidate = bid - (m5_atr * 1.5);
               if(trail_candidate > current_sl)
               {
                  if(bid - trail_candidate >= stops_level)
                  {
                     new_sl = trail_candidate;
                     modify = true;
                     event = "TRAIL_UPDATE";
                  }
                  else {
                     g_states[state_idx].blocked_stops++;
                     event = "BLOCKED_STOPS";
                  }
               }
            }
         }
      }
      else if(type == POSITION_TYPE_SELL)
      {
         if (MathAbs(ask - current_sl) <= freeze_level || (current_tp > 0 && MathAbs(ask - current_tp) <= freeze_level))
         {
            g_states[state_idx].blocked_freeze++;
            event = "BLOCKED_FREEZE";
         }
         else
         {
            double be_level = open_price - spread - comm_price_offset - buffer_vol;
            
            if(current_sl > be_level && current_sl != 0.0 && current_profit >= target_1r_profit)
            {
               if(be_level - ask >= stops_level)
               {
                  new_sl = be_level;
                  modify = true;
                  event = "BE_SET";
               }
               else {
                  g_states[state_idx].blocked_stops++;
                  event = "BLOCKED_STOPS";
               }
            }
            else if (current_sl <= be_level && current_sl != 0.0)
            {
               double trail_candidate = ask + (m5_atr * 1.5);
               if(trail_candidate < current_sl)
               {
                  if(trail_candidate - ask >= stops_level)
                  {
                     new_sl = trail_candidate;
                     modify = true;
                     event = "TRAIL_UPDATE";
                  }
                  else {
                     g_states[state_idx].blocked_stops++;
                     event = "BLOCKED_STOPS";
                  }
               }
            }
         }
      }
      
      double volume = PositionGetDouble(POSITION_VOLUME);
      double absolute_be_level = (type == POSITION_TYPE_BUY) ? (open_price + spread + comm_price_offset + buffer_vol) : (open_price - spread - comm_price_offset - buffer_vol);
      
      if(modify)
      {
         if(trade.PositionModify(ticket, new_sl, current_tp))
         {
            LogStage2Footprint(ticket, event, (type == POSITION_TYPE_BUY) ? "BUY" : "SELL", volume, 
                               absolute_be_level, new_sl, comm_price_offset, m15_atr, 
                               g_states[state_idx].blocked_stops, g_states[state_idx].blocked_freeze);
         }
      }
      else if (event == "BLOCKED_STOPS" || event == "BLOCKED_FREEZE")
      {
         // Log the block if it occurred
         LogStage2Footprint(ticket, event, (type == POSITION_TYPE_BUY) ? "BUY" : "SELL", volume, 
                            absolute_be_level, new_sl, comm_price_offset, m15_atr, 
                            g_states[state_idx].blocked_stops, g_states[state_idx].blocked_freeze);
      }
   }
}

