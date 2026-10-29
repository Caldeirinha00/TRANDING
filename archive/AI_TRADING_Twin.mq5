//+------------------------------------------------------------------+
//|                                              AI_TRADING_Twin.mq5 |
//|                                  Copyright 2026, AI_TRADING Team |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, AI_TRADING Team"
#property link      ""
#property version   "1.20"

#include <Trade\Trade.mqh>
#include "Indicators_Lib.mqh"
#include <AI_TRADING_IPC_Pipe.mqh>

// =========================================================
// INPUTS & GLOBAIS
// =========================================================
input double InpRiskPercent = 1.0;
input double InpCommissionPerLotRoundTurn = 7.0;
input ulong InpMagicNumber = 202609;

// --- ESTRATÉGIA CALIBRADA E OTIMIZADA ---
sinput string WFO_Params = "--- Strategy Parameters ---";
input int InpH1_EMA_Fast = 25;
input int InpH1_EMA_Slow = 70;
input int InpADX_Period = 14;
input double InpADX_Min = 30.0;

input int InpATR_Period = 14;
input int InpATR_Median_Lookback = 50;
input double InpATR_MinMult = 1.4;
input double InpATR_MaxMult = 2.5;

input double InpSL_ATR_Buffer = 1.2;
input double InpTP_RR = 1.25;

// --- FILTROS INSTITUCIONAIS DE RUÍDO ---
sinput string Filter_Params = "--- Execution Filters ---";
input int InpStartHour = 8;
input int InpEndHour = 16;
input bool InpRequireRejection = true;
input double InpMaxSlippageAtrRatio = 0.15; // Máximo drift de preço pós-handshake (15% ATR M5)

// --- HANDSHAKE IA (FASE 4) ---
sinput string LLM_Params = "--- AI Handshake (Phase 4) ---";
input bool InpEnableLLMHandshake = true;
input int InpSlaTimeoutMs = 5000;
input bool InpForceHandshakeInTester = false;

int g_file_handle = INVALID_HANDLE;
int g_stg2_handle = INVALID_HANDLE;
datetime g_last_bar_time = 0;

CTrade trade;

// --- FORWARD DECLARATIONS ---
void EvaluateAndTrade();
void ManageOpenPositions();
void LogStage2Footprint(ulong ticket, string event_type, string direction, double volume, double be_level, double new_sl, double comm_points, double m15_atr, int stops_blocked, int freeze_blocked);
bool RequestUnifiedHandshake(string direction, string setup_type, double adx, double atr, double m5_ema20, const MqlRates &candle, HandshakeResponse &resp);
bool ExecuteInstitutionalOrder(CTrade &trade_obj, const ENUM_ORDER_TYPE order_type, const double volume, const double signal_price, const double sl_dist, const double tp_dist, const double m5_atr, const string comment);

// Semântica temporal do ciclo M5 (imune a trigger falso no arranque)
bool NewM5Bar()
{
   datetime t = iTime(_Symbol, PERIOD_M5, 0);
   if(t == 0 || t == g_last_bar_time) return false;
   g_last_bar_time = t;
   return true;
}

//===================================================================
// EA EVENTS
//===================================================================
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(30);
   trade.SetTypeFillingBySymbol(_Symbol);
   g_last_bar_time = iTime(_Symbol, PERIOD_M5, 0);

   if(!MQLInfoInteger(MQL_OPTIMIZATION) && !MQLInfoInteger(MQL_TESTER))
   {
      g_file_handle = FileOpen("AI_TRADING_Footprint.jsonl", FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
      g_stg2_handle = FileOpen("AI_TRADING_Stage2_Footprint.jsonl", FILE_READ|FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
      if(g_stg2_handle != INVALID_HANDLE)
      {
         FileSeek(g_stg2_handle, 0, SEEK_END);
      }
   }
   PrintFormat("[AI_TRADING_Twin] Inicializado com sucesso. Magic=%I64u, Simbolo=%s", InpMagicNumber, _Symbol);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(g_file_handle != INVALID_HANDLE) FileClose(g_file_handle);
   if(g_stg2_handle != INVALID_HANDLE) FileClose(g_stg2_handle);
}

void OnTick()
{
   // Guarda de Emergência: Bloqueio imediato se o Kill Switch estiver armado (cache 500ms contra micro-picos de I/O)
   static ulong last_fs_check_ms = 0;
   static bool  kill_flag_cached = false;
   ulong now_ms = GetTickCount64();

   if(now_ms - last_fs_check_ms >= 500)
   {
      last_fs_check_ms = now_ms;
      kill_flag_cached = (GlobalVariableGet("AI_KILL_SWITCH_LOCK") > 0.5 ||
                          FileIsExist("AI_KILL_SWITCH_LOCK.flag", FILE_COMMON));
   }
   if(kill_flag_cached) return;

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
   
   if(total_trades < 40 || net_profit <= 0 || max_dd_pct > 20.0 || profit_factor < 1.15)
      return 0.0;
      
   double dd_penalty = MathPow(max_dd_pct / 10.0, 2.0);
   double avg_trade_profit = net_profit / total_trades;
   
   return (avg_trade_profit * MathSqrt(total_trades) * profit_factor) / (1.0 + dd_penalty);
}

void EvaluateAndTrade()
{
   // 1. Guard de Posição Única (Monopar / Sem piramidar - restrito a este bot)
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
      {
         return; // Posição ativa deste EA detetada. Aborta nova entrada.
      }
   }

   MqlRates m5[], h1[];
   if(CopyRates(_Symbol, PERIOD_M5, 1, 100, m5) != 100) return;
   if(CopyRates(_Symbol, PERIOD_H1, 1, 100, h1) != 100) return;
   
   // FIX CRÍTICO: Garantir explicitamente que ArraySetAsSeries é false (m5[99] = vela terminada)
   ArraySetAsSeries(m5, false);
   ArraySetAsSeries(h1, false);
   
   MqlRates last_candle = m5[99];
   
   // 2. Filtro de Janela de Liquidez (Horário Londres + NY Overlap)
   // FIX CRÍTICO: Off-by-one corrigido para InpEndHour (>= fecha novas entradas à hora estipulada)
   MqlDateTime dt;
   TimeToStruct(last_candle.time, dt);
   if(dt.hour < InpStartHour || dt.hour >= InpEndHour)
   {
      return; // Fora da janela de liquidez institucional
   }

   double h1_closes[], m5_closes[];
   ArrayResize(h1_closes, 100);
   ArrayResize(m5_closes, 100);
   for(int i=0; i<100; i++)
   {
      h1_closes[i] = h1[i].close;
      m5_closes[i] = m5[i].close;
   }
   
   double h1_ema_fast = PyEMA(h1_closes, InpH1_EMA_Fast);
   double h1_ema_slow = PyEMA(h1_closes, InpH1_EMA_Slow);
   double h1_adx = PyADX(h1, InpADX_Period);
   
   double m5_atr = PyATR(m5, InpATR_Period);
   double m5_atr_series[];
   PyATRSeries(m5, InpATR_Period, InpATR_Median_Lookback, m5_atr_series);
   double m5_atr_median = ArraySize(m5_atr_series) > 0 ? NpMedian(m5_atr_series) : m5_atr;
   
   double m5_ema20 = PyEMA(m5_closes, 20);
   
   string direction = "null";
   string setup_type = "NONE";
   string reason_code = "VALID";
   
   bool reject = false;
   
   if(h1_adx < InpADX_Min)
   {
      reason_code = "ADX_LOW";
      reject = true;
   }
   
   bool trend_bullish = h1_ema_fast > h1_ema_slow;
   bool trend_bearish = h1_ema_fast < h1_ema_slow;
   
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
   
   if(!reject)
   {
      if(trend_bullish)
      {
         bool rejection = !InpRequireRejection || (last_candle.close > (last_candle.high + last_candle.low) * 0.5);
         if(last_candle.low <= m5_ema20 && last_candle.close > m5_ema20 && rejection)
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
         bool rejection = !InpRequireRejection || (last_candle.close < (last_candle.high + last_candle.low) * 0.5);
         if(last_candle.high >= m5_ema20 && last_candle.close < m5_ema20 && rejection)
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
      
   if(g_file_handle != INVALID_HANDLE)
   {
      FileWrite(g_file_handle, json);
   }
   
   if(!reject && direction != "null")
   {
      MqlTick tick_pre;
      if(!SymbolInfoTick(_Symbol, tick_pre)) return;
      
      double signal_price = (direction == "\"BUY\"") ? tick_pre.ask : tick_pre.bid;
      
      // Handshake com a Inteligência Artificial (Named Pipe primário com fallback atómico)
      HandshakeResponse hs_resp;
      hs_resp.approved = true;
      hs_resp.kelly_risk_pct = InpRiskPercent;
      hs_resp.sl_atr_mult = InpSL_ATR_Buffer;
      hs_resp.tp_atr_mult = InpTP_RR;
      
      if(InpEnableLLMHandshake && (!MQLInfoInteger(MQL_OPTIMIZATION) && (!MQLInfoInteger(MQL_TESTER) || InpForceHandshakeInTester)))
      {
         bool approved = RequestUnifiedHandshake(direction, setup_type, h1_adx, m5_atr, m5_ema20, last_candle, hs_resp);
         if(!approved)
         {
            PrintFormat("[AI_TRADING] Ordem %s abortada por Veto da IA ou Timeout Fail-Closed.", direction);
            return;
         }
      }
      
      // FIX CRÍTICO 1: Refrescar MqlTick imediatamente APÓS o handshake IPC
      MqlTick tick_post;
      if(!SymbolInfoTick(_Symbol, tick_post))
      {
         PrintFormat("[EXEC-ABORT] Falha ao obter tick pós-handshake para %s.", _Symbol);
         return;
      }
      
      double exec_price = (direction == "\"BUY\"") ? tick_post.ask : tick_post.bid;
      double price_drift = MathAbs(exec_price - signal_price);
      double max_drift = InpMaxSlippageAtrRatio * m5_atr;
      
      if(price_drift > max_drift)
      {
         PrintFormat("[EXEC-ABORT] Drift excessivo pós-handshake: %.5f pts (Exec: %.5f vs Sinal: %.5f, MaxTol: %.5f). Ordem abortada.",
                     price_drift / _Point, exec_price, signal_price, max_drift);
         return;
      }
      
      // Recalcular distâncias dinâmicas de SL e TP sobre o exec_price atualizado
      double sl_buffer = (hs_resp.sl_atr_mult > 0.1) ? hs_resp.sl_atr_mult : InpSL_ATR_Buffer;
      double tp_rr = (hs_resp.tp_atr_mult > 0.1) ? hs_resp.tp_atr_mult : InpTP_RR;
      
      double sl_init = 0.0;
      double tp_init = 0.0;
      
      if(direction == "\"BUY\"")
      {
         sl_init = MathMin(last_candle.low, m5_ema20) - (m5_atr * sl_buffer);
         tp_init = exec_price + tp_rr * (exec_price - sl_init);
      }
      else
      {
         sl_init = MathMax(last_candle.high, m5_ema20) + (m5_atr * sl_buffer);
         tp_init = exec_price - tp_rr * (sl_init - exec_price);
      }
      
      double sl_dist = MathAbs(exec_price - sl_init);
      double tp_dist = MathAbs(exec_price - tp_init);
      
      double active_risk_pct = (hs_resp.kelly_risk_pct > 0.05) ? hs_resp.kelly_risk_pct : InpRiskPercent;
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double risk_money = balance * (active_risk_pct / 100.0);
      
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      if(tick_size <= 0.0 || tick_value <= 0.0) return;
      
      double sl_distance_ticks = sl_dist / tick_size;
      double money_lost_per_lot = sl_distance_ticks * tick_value;
      if(money_lost_per_lot <= 0.0) return;
      
      double raw_volume = risk_money / money_lost_per_lot;
      double vol_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
      if(vol_step <= 0.0) vol_step = 0.01;
      double vol_min = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      double vol_max = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
      
      double volume = MathFloor(raw_volume / vol_step) * vol_step;
      volume = NormalizeDouble(volume, (int)MathRound(-MathLog10(vol_step)));
      
      if(volume < vol_min) return; 
      if(volume > vol_max) volume = vol_max;
      
      // Execução Institucional com Retry Idempotente e Proteção contra Requotes
      ENUM_ORDER_TYPE otype = (direction == "\"BUY\"") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
      string comment = (direction == "\"BUY\"") ? "AI_WFO_B" : "AI_WFO_S";
      
      ExecuteInstitutionalOrder(trade, otype, volume, signal_price, sl_dist, tp_dist, m5_atr, comment);
   }
}

//+------------------------------------------------------------------+
//| Execução Institucional com tolerância estrita e retry transiente |
//+------------------------------------------------------------------+
bool ExecuteInstitutionalOrder(
   CTrade &trade_obj,
   const ENUM_ORDER_TYPE order_type,
   const double volume,
   const double signal_price,
   const double sl_dist,
   const double tp_dist,
   const double m5_atr,
   const string comment
) {
   double max_slippage_price = InpMaxSlippageAtrRatio * m5_atr;
   ulong deviation_pts = (ulong)MathMax(2.0, MathRound((0.08 * m5_atr) / _Point));
   trade_obj.SetDeviationInPoints((uint)deviation_pts);
   trade_obj.SetTypeFillingBySymbol(_Symbol);

   int max_attempts = 3;
   for(int attempt = 1; attempt <= max_attempts; attempt++) {
      MqlTick tick;
      if(!SymbolInfoTick(_Symbol, tick)) return false;

      double exec_price = (order_type == ORDER_TYPE_BUY) ? tick.ask : tick.bid;
      if(MathAbs(exec_price - signal_price) > max_slippage_price) {
         PrintFormat("[EXEC-ABORT] Drift excessivo na execução: %.5f vs Sinal %.5f", exec_price, signal_price);
         return false;
      }

      double sl = (order_type == ORDER_TYPE_BUY) ? NormalizeDouble(exec_price - sl_dist, _Digits)
                                                 : NormalizeDouble(exec_price + sl_dist, _Digits);
      double tp = (order_type == ORDER_TYPE_BUY) ? NormalizeDouble(exec_price + tp_dist, _Digits)
                                                 : NormalizeDouble(exec_price - tp_dist, _Digits);

      bool sent = (order_type == ORDER_TYPE_BUY)
                  ? trade_obj.Buy(volume, _Symbol, exec_price, sl, tp, comment)
                  : trade_obj.Sell(volume, _Symbol, exec_price, sl, tp, comment);

      uint retcode = trade_obj.ResultRetcode();
      if(sent && (retcode == TRADE_RETCODE_DONE || retcode == TRADE_RETCODE_DONE_PARTIAL)) {
         PrintFormat("[EXEC-SUCCESS] Ticket=%I64u executado a %.5f (Tentativa %d)", trade_obj.ResultOrder(), exec_price, attempt);
         return true;
      }

      // Retry apenas em requotes ou pequenas variações de cotação
      if(retcode == TRADE_RETCODE_REQUOTE || retcode == TRADE_RETCODE_PRICE_CHANGED || retcode == TRADE_RETCODE_PRICE_OFF) {
         Sleep(25 * attempt);
         continue;
      }
      
      PrintFormat("[EXEC-FAIL] Ordem rejeitada com retcode %d: %s", retcode, trade_obj.ResultComment());
      break;
   }
   return false;
}

void LogStage2Footprint(ulong ticket, string event_type, string direction, double volume, double be_level, double new_sl, double comm_points, double m15_atr, int stops_blocked, int freeze_blocked)
{
   if(g_stg2_handle == INVALID_HANDLE) return;
   if(MQLInfoInteger(MQL_OPTIMIZATION)) return;
   
   string json = StringFormat(
      "{\"ticket\": %d, \"event_type\": \"%s\", \"direction\": \"%s\", \"volume\": %.2f, \"be_level\": %.5f, \"new_sl\": %.5f, \"commission_points_used\": %.5f, \"m15_atr_used\": %.5f, \"stops_level_blocked_count\": %d, \"freeze_level_blocked_count\": %d}",
      ticket, event_type, direction, volume, be_level, new_sl, comm_points, m15_atr, stops_blocked, freeze_blocked
   );
   
   FileWrite(g_stg2_handle, json);
   FileFlush(g_stg2_handle);
}

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
   static datetime cached_bar = 0;
   static double cached_atr = -1.0;
   
   datetime cur = iTime(_Symbol, PERIOD_M15, 1);
   if(cur != cached_bar || cached_atr < 0.0)
   {
      MqlRates m15[];
      if(CopyRates(_Symbol, PERIOD_M15, 1, 100, m15) == 100)
      {
         ArraySetAsSeries(m15, false);
         cached_atr = PyATR(m15, 14);
         cached_bar = cur;
      }
      else return -1.0;
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
   if(m15_atr < 0.0) return;
   double buffer_vol = m15_atr * 0.15;
   
   MqlRates m5[];
   if(CopyRates(_Symbol, PERIOD_M5, 1, 100, m5) != 100) return;
   ArraySetAsSeries(m5, false);
   double m5_atr = PyATR(m5, 14);
   
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC) != InpMagicNumber) continue;
      
      int state_idx = GetStateIndex(ticket);
      
      long type = PositionGetInteger(POSITION_TYPE);
      double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_sl = PositionGetDouble(POSITION_SL);
      double current_tp = PositionGetDouble(POSITION_TP);
      double current_profit = PositionGetDouble(POSITION_PROFIT);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double swap = PositionGetDouble(POSITION_SWAP);
      
      double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      if(tick_value <= 0.0 || volume <= 0.0) continue;
      
      // FIX CRÍTICO: Cálculo exato do Breakeven cobrindo comissão round-turn + swap negativo na moeda da conta
      double total_cost_ccy = (InpCommissionPerLotRoundTurn * volume) + MathMax(0.0, -swap);
      double cost_in_price = (total_cost_ccy / (volume * tick_value)) * tick_size;
      
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double target_1r_profit = balance * (InpRiskPercent / 100.0);
      
      double new_sl = current_sl;
      bool modify = false;
      string event = "";
      
      if(type == POSITION_TYPE_BUY)
      {
         if (MathAbs(bid - current_sl) <= freeze_level || (current_tp > 0 && MathAbs(bid - current_tp) <= freeze_level))
         {
            g_states[state_idx].blocked_freeze++;
            event = "BLOCKED_FREEZE";
         }
         else
         {
            double be_level = open_price + cost_in_price + spread + buffer_vol;
            
            if(current_sl < be_level && current_profit >= target_1r_profit)
            {
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
            else if (current_sl >= be_level)
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
            double be_level = open_price - cost_in_price - spread - buffer_vol;
            
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
      
      if(modify)
      {
         string dir_str = (type == POSITION_TYPE_BUY) ? "BUY" : "SELL";
         double comm_pts = cost_in_price / point;
         
         if(trade.PositionModify(ticket, new_sl, current_tp))
         {
            LogStage2Footprint(ticket, event, dir_str, volume, 
                               (type == POSITION_TYPE_BUY) ? (open_price + cost_in_price + spread + buffer_vol) : (open_price - cost_in_price - spread - buffer_vol),
                               new_sl, comm_pts, m15_atr, g_states[state_idx].blocked_stops, g_states[state_idx].blocked_freeze);
         }
      }
   }
}

//+--------------------------------------------------------------------+
//| Handshake IPC Unificado: Tenta Named Pipe, com fallback para Disco |
//+--------------------------------------------------------------------+
bool RequestUnifiedHandshake(
   string direction,
   string setup_type,
   double adx,
   double atr,
   double m5_ema20,
   const MqlRates &candle,
   HandshakeResponse &resp
) {
   // 1. Tentar Named Pipe de Alta Velocidade (< 1ms)
   uint pipe_timeout = (uint)MathMin(500, InpSlaTimeoutMs);
   if(RequestPipeHandshake(direction, setup_type, adx, atr, m5_ema20, candle, pipe_timeout, resp))
   {
      PrintFormat("[IPC-PIPE] Handshake aprovado via Named Pipe em sub-milissegundo! Kelly=%.2f%%", resp.kelly_risk_pct);
      return true;
   }

   // 2. Fallback para Handshake Atómico em Disco (FILE_COMMON)
   string clean_dir = (direction == "\"BUY\"" || direction == "BUY") ? "BUY" : "SELL";
   string uuid = StringFormat("%I64u_%d", GetTickCount64(), MathRand());
   
   string req_json = StringFormat(
      "{\"uuid\": \"%s\", \"symbol\": \"%s\", \"direction\": \"%s\", \"setup_type\": \"%s\", \"h1_adx\": %.5f, \"m5_atr\": %.5f, \"m5_ema20\": %.5f, \"trigger_close\": %.5f, \"time\": %d}",
      uuid, _Symbol, clean_dir, setup_type, adx, atr, m5_ema20, candle.close, (int)candle.time
   );
   
   string req_tmp = "AI_TRADING_IPC\\requests\\req_" + uuid + ".json.tmp";
   string req_final = "AI_TRADING_IPC\\requests\\req_" + uuid + ".json";
   string res_final = "AI_TRADING_IPC\\responses\\res_" + uuid + ".json";
   
   int h_req = FileOpen(req_tmp, FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON);
   if(h_req == INVALID_HANDLE)
   {
      PrintFormat("[IPC Handshake FAIL-CLOSED] Falha ao criar %s (Erro %d). Ordem cancelada.", req_tmp, GetLastError());
      return false;
   }
   FileWriteString(h_req, req_json);
   FileClose(h_req);
   
   if(!FileMove(req_tmp, FILE_COMMON, req_final, FILE_COMMON))
   {
      PrintFormat("[IPC Handshake FAIL-CLOSED] Falha ao renomear para %s. Ordem cancelada.", req_final);
      return false;
   }
   
   uint start_tick = GetTickCount();
   bool response_found = false;
   bool is_vetoed = true;
   string reason = "TIMEOUT_FAIL_CLOSED";
   
   while(GetTickCount() - start_tick < (uint)InpSlaTimeoutMs)
   {
      if(FileIsExist(res_final, FILE_COMMON))
      {
         Sleep(10);
         int h_res = FileOpen(res_final, FILE_READ|FILE_TXT|FILE_ANSI|FILE_COMMON);
         if(h_res != INVALID_HANDLE)
         {
            string content = "";
            while(!FileIsEnding(h_res))
            {
               content += FileReadString(h_res);
            }
            FileClose(h_res);
            
            string lower_content = content;
            StringToLower(lower_content);
            if(StringFind(lower_content, "\"vetoed\": false") >= 0 || 
               StringFind(lower_content, "\"vetoed\":false") >= 0 ||
               StringFind(lower_content, "\"approved\": true") >= 0 ||
               StringFind(lower_content, "\"approved\":true") >= 0)
            {
               is_vetoed = false;
               reason = "APPROVED_BY_AI";
            }
            else
            {
               is_vetoed = true;
               reason = "VETOED_BY_AI";
            }
            
            FileDelete(res_final, FILE_COMMON);
            response_found = true;
            break;
         }
      }
      Sleep(20);
   }
   
   uint elapsed = GetTickCount() - start_tick;
   
   if(!response_found)
   {
      PrintFormat("[IPC Handshake FAIL-CLOSED] Timeout SLA de %d ms excedido (%d ms decorridos).", InpSlaTimeoutMs, elapsed);
      FileDelete(req_final, FILE_COMMON);
      FileDelete(req_tmp, FILE_COMMON);
      return false;
   }
   
   if(is_vetoed)
   {
      PrintFormat("[IPC Handshake VETOED] Ordem bloqueada pela IA em %d ms. Razao: %s", elapsed, reason);
      return false;
   }
   
   PrintFormat("[IPC Handshake APPROVED] Ordem autorizada pela IA em %d ms (via arquivo).", elapsed);
   return true;
}
