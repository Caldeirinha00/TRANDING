//+------------------------------------------------------------------+
//|                                           AI_TRADING_Twin_V2.mq5 |
//|                         Copyright 2026, Institutional Quant Core |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, Institutional Quant Core"
#property link      ""
#property version   "2.00"

#include <Trade\Trade.mqh>
#include "Indicators_Lib.mqh"
#include "AutonomousDefenseManager.mqh"
#include "IPC_BinaryPipe.mqh"
#include "Regime_Microstructure_Lib.mqh"
#include "SmartOrderRouter.mqh"

//===================================================================
// INPUT PARAMETERS
//===================================================================
input group "--- Parâmetros de Execução & Risco ---"
input ulong  InpMagicNumber            = 202610;    // Magic Number
input double InpRiskPercent            = 1.0;       // Risco Máximo por Trade (%)
input uint   InpIPCTimeoutMS           = 3000;      // SLA Timeout IPC (ms)
input bool   InpUseBinaryPipe          = true;      // Utilizar Windows Named Pipe V2

input group "--- Filtros de Regime & Tendência ---"
input int    InpH1_EMA_Fast            = 20;        // H1 EMA Rápida
input int    InpH1_EMA_Slow            = 50;        // H1 EMA Lenta
input int    InpADX_Period             = 14;        // ADX Período
input double InpADX_Min                = 20.0;      // ADX Mínimo H1
input double InpFDI_MaxChop            = 1.54;      // FDI Limite Máximo de Ruído (Sevcik)
input double InpPE_MaxEntropy          = 0.94;      // Permutation Entropy Máxima (Bandt-Pompe)
input bool   InpUseHawkesOFI           = true;      // Validar Hawkes Tick OFI Imbalance

input group "--- Filtros de Sessão Operacional ---"
input bool   InpUseSessionFilter       = true;      // Ativar Filtro de Sessão (London + NY)
input int    InpSessionStartHourUTC    = 7;         // Hora Início UTC (Abertura Londres)
input int    InpSessionEndHourUTC      = 16;        // Hora Fim UTC (Fecho Londres/NY Overlap)

input group "--- Gatilhos de Alpha da Fase 2 ---"
input bool   InpEnablePullback         = true;      // Ativar Gatilho de Pullback à EMA20
input bool   InpEnableBOS              = false;     // Ativar Break of Structure (BOS - Falso rompimento em M5)
input bool   InpEnableSweep            = true;      // Ativar Liquidity Sweep (Reversão de Extremos)
input double InpD1_MinATRPercentile    = 0.05;      // Mínimo Percentil de Volatilidade D1

input group "--- Volatilidade & Gatilhos M5 ---"
input int    InpATR_Period             = 14;        // ATR Período
input int    InpATR_Median_Lookback    = 50;        // ATR Lookback da Mediana
input double InpATR_MinMult            = 0.8;       // Multiplicador Mínimo do ATR
input double InpATR_MaxMult            = 2.5;       // Multiplicador Máximo do ATR
input double InpSL_ATR_Buffer          = 0.5;       // Buffer do Stop Loss (ATR)
input double InpTP_RR                  = 2.0;       // Relação Risco:Retorno (TP)

//===================================================================
// OBJETOS GLOBAIS
//===================================================================
CBinaryIPCClient          g_ipc;
CSmartOrderRouter         g_sor(InpMagicNumber);
CAutonomousDefenseManager g_adm(InpMagicNumber);
CQuantRegimeEngine        g_regime;

bool NewM5Bar()
{
   static datetime last_time = 0;
   datetime cur_time = iTime(_Symbol, PERIOD_M5, 0);
   if(cur_time == 0 || cur_time == last_time) return false;
   last_time = cur_time;
   return true;
}

//===================================================================
// EVENTOS PRINCIPAIS DO EA
//===================================================================
int OnInit()
{
   PrintFormat("[INIT] AI_TRADING_Twin_V2 carregado no ativo %s. Magic=%d", _Symbol, InpMagicNumber);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   PrintFormat("[DEINIT] AI_TRADING_Twin_V2 finalizado. Razão: %d", reason);
}

void OnTick()
{
   // 1. Obter ATR M5 para gestão contínua de risco
   MqlRates m5_quick[];
   double m5_atr = 0.0010;
   if(CopyRates(_Symbol, PERIOD_M5, 1, 20, m5_quick) == 20)
   {
      m5_atr = PyATR(m5_quick, InpATR_Period);
   }

   // 2. Gestão Autónoma de Posições (BE + Trailing Stop Monotónico + Failover ORANGE/RED)
   g_adm.ManageOpenPositionsTick(_Symbol, m5_atr);

   // 3. Avaliação de Entrada no Fecho Exato da Barra M5
   if(NewM5Bar())
   {
      EvaluateAndTrade();
   }
}

//===================================================================
// AVALIAÇÃO DE ENTRADA & HANDSHAKE
//===================================================================
void EvaluateAndTrade()
{
   MqlRates m5[], h1[];
   if(CopyRates(_Symbol, PERIOD_M5, 1, 100, m5) != 100) return;
   if(CopyRates(_Symbol, PERIOD_H1, 1, 100, h1) != 100) return;

   // 0. Filtro de Sessão Operacional Institucional (London + NY Overlap)
   if(InpUseSessionFilter)
   {
      MqlDateTime dt;
      TimeToStruct(m5[99].time, dt);
      if(dt.hour < InpSessionStartHourUTC || dt.hour >= InpSessionEndHourUTC)
      {
         return;
      }
   }

   double h1_closes[], m5_closes[];
   ArrayResize(h1_closes, 100);
   ArrayResize(m5_closes, 100);
   for(int i = 0; i < 100; i++)
   {
      h1_closes[i] = h1[i].close;
      m5_closes[i] = m5[i].close;
   }

   // 1. Filtros de Tendência H1
   double h1_ema20 = PyEMA(h1_closes, InpH1_EMA_Fast);
   double h1_ema50 = PyEMA(h1_closes, InpH1_EMA_Slow);
   double h1_adx   = PyADX(h1, InpADX_Period);

   if(h1_adx < InpADX_Min) return; // Filtro ADX rejeitado

   bool trend_bullish = h1_ema20 > h1_ema50;
   bool trend_bearish = h1_ema20 < h1_ema50;
   if(!trend_bullish && !trend_bearish) return;

   // 1b. Filtro de Regime de Volatilidade Diária D1
   if(InpD1_MinATRPercentile > 0.0)
   {
      double d1_pct = CQuantRegimeEngine::CalcD1ATRPercentile(_Symbol, 14, 100);
      if(d1_pct < InpD1_MinATRPercentile)
         return; // Mercado comprimido sem momentum suficiente
   }

   // 2. Filtros de Volatilidade & Ruído M5
   double m5_atr = PyATR(m5, InpATR_Period);
   double m5_atr_series[];
   PyATRSeries(m5, InpATR_Period, InpATR_Median_Lookback, m5_atr_series);
   double m5_atr_median = ArraySize(m5_atr_series) > 0 ? NpMedian(m5_atr_series) : m5_atr;

   if(m5_atr < (m5_atr_median * InpATR_MinMult) || m5_atr > (m5_atr_median * InpATR_MaxMult))
      return;

   // 3. Índice de Dimensão Fractal Sevcik (FDI), Entropia Permutação & Velocidade Kalman
   double fdi_m5 = CQuantRegimeEngine::CalcSevcikFDI(m5_closes, 30);
   if(fdi_m5 > InpFDI_MaxChop)
   {
      // Regime altamente caótico / anti-persistente
      return;
   }

   double pe_m5 = CQuantRegimeEngine::CalcPermutationEntropyOrder3(m5_closes, 30);
   if(pe_m5 > InpPE_MaxEntropy)
   {
      // Sequência microestrutural com excesso de ruído branco combinatório
      return;
   }

   double kalman_vel_z = g_regime.UpdateKalmanVelocityZ(m5[99].close, m5_atr);

   // 4. Verificação de Choque de Microestrutura / News Spike sem REST API
   if(CQuantRegimeEngine::IsMicrostructureShock(_Symbol, m5_atr))
   {
      Print("[VETO] Choque de microestrutura/liquidez local detetado. Entrada abortada.");
      return;
   }

   // 5. Gatilhos de Alpha Multimodais (Fase 2)
   MqlRates last_candle = m5[99];
   double m5_ema20 = PyEMA(m5_closes, 20);
   int direction = 0; // 1 = BUY, -1 = SELL

   // 5a. Pullback à EMA(20) com Confirmação Cinemática de Kalman
   if(InpEnablePullback)
   {
      if(trend_bullish && last_candle.low <= m5_ema20 && last_candle.close > m5_ema20 && kalman_vel_z > 0.2)
         direction = 1;
      else if(trend_bearish && last_candle.high >= m5_ema20 && last_candle.close < m5_ema20 && kalman_vel_z < -0.2)
         direction = -1;
   }

   // 5b. Break of Structure (BOS) com Impulso de Kalman
   if(direction == 0 && InpEnableBOS)
   {
      int bos_dir = CQuantRegimeEngine::DetectBOSBreakout(m5, kalman_vel_z, 15);
      if((bos_dir == 1 && trend_bullish) || (bos_dir == -1 && trend_bearish))
         direction = bos_dir;
   }

   // 5c. Liquidity Sweep de Extremos Locais com Rejeição
   if(direction == 0 && InpEnableSweep)
   {
      int sweep_dir = CQuantRegimeEngine::DetectLiquiditySweep(m5, kalman_vel_z, 20);
      if((sweep_dir == 1 && trend_bullish) || (sweep_dir == -1 && trend_bearish))
         direction = sweep_dir;
   }

   if(direction == 0) return;

   // 5b. Confirmação do Fluxo de Ordens Institucional (Hawkes Tick OFI)
   if(InpUseHawkesOFI)
   {
      double ofi_hawkes = CQuantRegimeEngine::CalcHawkesTickOFI(_Symbol, 150, 0.005);
      if((direction == 1 && ofi_hawkes < -0.15) || (direction == -1 && ofi_hawkes > 0.15))
      {
         PrintFormat("[VETO] Hawkes Tick OFI divergente (%+.3f) face à direção %d. Entrada rejeitada.", ofi_hawkes, direction);
         return;
      }
   }

   // 6. Preparação dos Níveis Base de SL/TP
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double exec_price = (direction == 1) ? ask : bid;

   double base_sl = 0.0;
   double base_tp = 0.0;
   if(direction == 1)
   {
      base_sl = MathMin(last_candle.low, m5_ema20) - (m5_atr * InpSL_ATR_Buffer);
      base_tp = exec_price + InpTP_RR * (exec_price - base_sl);
   }
   else
   {
      base_sl = MathMax(last_candle.high, m5_ema20) + (m5_atr * InpSL_ATR_Buffer);
      base_tp = exec_price - InpTP_RR * (base_sl - exec_price);
   }

   // 7. IPC Handshake Binário com o Servidor Python Copilot
   double approved_lots = 0.0;
   double final_sl = base_sl;
   double final_tp = base_tp;

   const bool use_pipe = InpUseBinaryPipe && !MQLInfoInteger(MQL_TESTER) && !MQLInfoInteger(MQL_OPTIMIZATION);
   if(use_pipe)
   {
      IPCResponsePacket res;
      bool ok = g_ipc.QueryCopilot(_Symbol, direction, m5_atr, h1_adx, fdi_m5, kalman_vel_z, InpIPCTimeoutMS, res);

      if(ok && res.veto == 0 && res.approved_lots > 0.0)
      {
         g_adm.RecordIPCSuccess();
         approved_lots = res.approved_lots;

         // Modulação dinâmica de SL e TP via RAG k-NN
         if(res.sl_mult_adj > 0.5 && res.sl_mult_adj < 2.0)
         {
            double sl_dist = MathAbs(exec_price - base_sl) * res.sl_mult_adj;
            final_sl = (direction == 1) ? (exec_price - sl_dist) : (exec_price + sl_dist);
         }
         if(res.tp_mult_adj > 0.5 && res.tp_mult_adj < 3.0)
         {
            double tp_dist = MathAbs(exec_price - base_tp) * res.tp_mult_adj;
            final_tp = (direction == 1) ? (exec_price + tp_dist) : (exec_price - tp_dist);
         }

         PrintFormat("[IPC APROVADO] Lotes=%.2f | MetaProb=%.2f | SL_Adj=%.2f | TP_Adj=%.2f",
                     approved_lots, res.meta_prob, res.sl_mult_adj, res.tp_mult_adj);
      }
      else
      {
         g_adm.RecordIPCFailure();
         PrintFormat("[IPC VETO/REJECT] ReasonCode=%d | Veto=%d | Lots=%.2f", res.reason_code, res.veto, res.approved_lots);
         return; // Fail-Closed: Não executa sem aprovação explícita
      }
   }
   else
   {
      // Fallback de Dimensionamento Local se Pipe estiver desativado
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double risk_money = balance * (InpRiskPercent / 100.0);
      double sl_ticks = MathAbs(exec_price - base_sl) / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      if(sl_ticks > 0 && tick_val > 0)
      {
         approved_lots = NormalizeDouble(MathFloor((risk_money / (sl_ticks * tick_val)) / 0.01) * 0.01, 2);
      }
   }

   // 8. Execução Institucional Inteligente com Smart Order Router (SOR)
   if(approved_lots >= 0.01)
   {
      bool executed = g_sor.ExecuteSmartEntry(_Symbol, direction, approved_lots, m5_atr, final_sl, final_tp, 20);
      if(executed)
      {
         PrintFormat("[EXECUÇÃO SOR OK] %s %s %.2f lotes @ %.5f | SL: %.5f | TP: %.5f",
                     (direction == 1 ? "BUY" : "SELL"), _Symbol, approved_lots, exec_price, final_sl, final_tp);
      }
      else
      {
         Print("[EXECUÇÃO SOR FALHOU] Desvio ou spread excessivo.");
      }
   }
}

//===================================================================
// WFO Custom Fitness Function
//===================================================================
double OnTester()
{
   double net_profit = TesterStatistics(STAT_PROFIT);
   double max_dd_pct = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   int total_trades  = (int)TesterStatistics(STAT_TRADES);
   double profit_fac = TesterStatistics(STAT_PROFIT_FACTOR);

   if(total_trades < 60 || net_profit <= 0 || max_dd_pct > 20.0 || profit_fac < 1.2)
      return 0.0;

   double dd_penalty = MathPow(max_dd_pct / 10.0, 2.0);
   double avg_profit = net_profit / total_trades;

   return (avg_profit * MathSqrt(total_trades) * profit_fac) / (1.0 + dd_penalty);
}
