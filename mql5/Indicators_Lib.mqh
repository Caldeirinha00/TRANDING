//+------------------------------------------------------------------+
//|                                              Indicators_Lib.mqh |
//|                                  Copyright 2026, AI_TRADING Team |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, AI_TRADING Team"
#property link      ""
#property strict

// 1.3 A mediana (np.median) exata para amostra par
double NpMedian(const double &src[])
{
   double tmp[];
   ArrayCopy(tmp, src);
   ArraySort(tmp);
   int n = ArraySize(tmp);
   if(n == 0) return 0.0;
   return ((n % 2) != 0) ? tmp[n/2] : 0.5 * (tmp[n/2 - 1] + tmp[n/2]);
}

// 1.1 EMA idêntica ao Pandas/Numpy
// NOTA: PyEMA deliberadamente exclui testes unitários contra iMA nativo, 
// pois a política do motor estipula truncatura estrita a 100 velas (ignorando o warm-up histórico completo).
double PyEMA(const double &series[], int period)
{
   int n = ArraySize(series);
   if (n < period) return (n > 0 ? series[n-1] : 0.0);
   
   double ema = 0;
   for(int i=0; i<period; i++) ema += series[i];
   ema /= period;
   
   double multiplier = 2.0 / (period + 1.0);
   for(int i=period; i<n; i++)
   {
      ema = (series[i] - ema) * multiplier + ema;
   }
   return ema;
}

// 1.2 ADX idêntico (Wilder Smoothing reescrito, zero native handles)
double PyADX(const MqlRates &candles[], int period)
{
   int n = ArraySize(candles);
   if (n <= period * 2) return 0.0;
   
   double tr_list[];
   double plus_dm_list[];
   double minus_dm_list[];
   ArrayResize(tr_list, n-1);
   ArrayResize(plus_dm_list, n-1);
   ArrayResize(minus_dm_list, n-1);
   
   for(int i=1; i<n; i++)
   {
      double high_low = candles[i].high - candles[i].low;
      double high_close = MathAbs(candles[i].high - candles[i-1].close);
      double low_close = MathAbs(candles[i].low - candles[i-1].close);
      tr_list[i-1] = MathMax(high_low, MathMax(high_close, low_close));
      
      double up_move = candles[i].high - candles[i-1].high;
      double down_move = candles[i-1].low - candles[i].low;
      
      if(up_move > down_move && up_move > 0)
      {
         plus_dm_list[i-1] = up_move;
         minus_dm_list[i-1] = 0.0;
      }
      else if(down_move > up_move && down_move > 0)
      {
         plus_dm_list[i-1] = 0.0;
         minus_dm_list[i-1] = down_move;
      }
      else
      {
         plus_dm_list[i-1] = 0.0;
         minus_dm_list[i-1] = 0.0;
      }
   }
   
   int num_smooth = (n-1) - period + 1;
   double tr_smooth[];
   double plus_dm_smooth[];
   double minus_dm_smooth[];
   ArrayResize(tr_smooth, num_smooth);
   ArrayResize(plus_dm_smooth, num_smooth);
   ArrayResize(minus_dm_smooth, num_smooth);
   
   // Wilder Smoothing Lambda-style
   double sm_tr = 0, sm_plus = 0, sm_minus = 0;
   for(int i=0; i<period; i++)
   {
      sm_tr += tr_list[i];
      sm_plus += plus_dm_list[i];
      sm_minus += minus_dm_list[i];
   }
   tr_smooth[0] = sm_tr;
   plus_dm_smooth[0] = sm_plus;
   minus_dm_smooth[0] = sm_minus;
   
   int out_idx = 1;
   for(int i=period; i<ArraySize(tr_list); i++)
   {
      sm_tr = sm_tr - (sm_tr / period) + tr_list[i];
      tr_smooth[out_idx] = sm_tr;
      
      sm_plus = sm_plus - (sm_plus / period) + plus_dm_list[i];
      plus_dm_smooth[out_idx] = sm_plus;
      
      sm_minus = sm_minus - (sm_minus / period) + minus_dm_list[i];
      minus_dm_smooth[out_idx] = sm_minus;
      
      out_idx++;
   }
   
   double dx_list[];
   ArrayResize(dx_list, num_smooth);
   for(int i=0; i<num_smooth; i++)
   {
      double tr = tr_smooth[i];
      if(tr == 0)
      {
         dx_list[i] = 0.0;
         continue;
      }
      double plus_di = 100.0 * (plus_dm_smooth[i] / tr);
      double minus_di = 100.0 * (minus_dm_smooth[i] / tr);
      if((plus_di + minus_di) == 0)
         dx_list[i] = 0.0;
      else
         dx_list[i] = 100.0 * MathAbs(plus_di - minus_di) / (plus_di + minus_di);
   }
   
   if(ArraySize(dx_list) < period) return 0.0;
   
   double adx = 0;
   for(int i=0; i<period; i++) adx += dx_list[i];
   adx /= period;
   
   for(int i=period; i<ArraySize(dx_list); i++)
   {
      adx = (adx * (period - 1) + dx_list[i]) / period;
   }
   
   return adx;
}

// 1.2 ATR idêntico (Wilder's Smoothing / RMA)
double PyATR(const MqlRates &candles[], int period)
{
   int n = ArraySize(candles);
   if (n <= period) return 0.0;
   double tr_list[];
   ArrayResize(tr_list, n-1);
   
   for(int i=1; i<n; i++)
   {
      double high_low = candles[i].high - candles[i].low;
      double high_close = MathAbs(candles[i].high - candles[i-1].close);
      double low_close = MathAbs(candles[i].low - candles[i-1].close);
      tr_list[i-1] = MathMax(high_low, MathMax(high_close, low_close));
   }
   
   double atr = 0;
   for(int i=0; i<period; i++) atr += tr_list[i];
   atr /= period;
   
   for(int i=period; i<ArraySize(tr_list); i++)
   {
      atr = (atr * (period - 1) + tr_list[i]) / period;
   }
   return atr;
}

// ATR Series idêntico (Wilder's Smoothing / RMA)
void PyATRSeries(const MqlRates &candles[], int period, int lookback, double &out_series[])
{
   int n = ArraySize(candles);
   if (n <= period + lookback)
   {
      ArrayResize(out_series, 1);
      out_series[0] = 0.0;
      return;
   }
   double tr_list[];
   ArrayResize(tr_list, n-1);
   for(int i=1; i<n; i++)
   {
      double high_low = candles[i].high - candles[i].low;
      double high_close = MathAbs(candles[i].high - candles[i-1].close);
      double low_close = MathAbs(candles[i].low - candles[i-1].close);
      tr_list[i-1] = MathMax(high_low, MathMax(high_close, low_close));
   }
   
   int tr_count = ArraySize(tr_list);
   double atr = 0;
   for(int i=0; i<period; i++) atr += tr_list[i];
   atr /= period;
   
   double full_series[];
   ArrayResize(full_series, tr_count - period + 1);
   full_series[0] = atr;
   
   int out_idx = 1;
   for(int i=period; i<tr_count; i++)
   {
      atr = (atr * (period - 1) + tr_list[i]) / period;
      full_series[out_idx++] = atr;
   }
   
   int start_idx = ArraySize(full_series) - lookback;
   if(start_idx < 0) start_idx = 0;
   
   ArrayResize(out_series, lookback);
   for(int i=0; i<lookback; i++)
   {
      out_series[i] = full_series[start_idx + i];
   }
}
