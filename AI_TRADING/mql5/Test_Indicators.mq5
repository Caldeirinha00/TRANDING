//+------------------------------------------------------------------+
//|                                              Test_Indicators.mq5 |
//|                                  Copyright 2026, AI_TRADING Team |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, AI_TRADING Team"
#property link      ""
#property version   "1.00"
#property script_show_inputs

input int InpBarsToTest = 10000;
input int InpPeriod = 14;

#include "Indicators_Lib.mqh"

void TestTimeframe(ENUM_TIMEFRAMES tf, string tf_name)
{
   PrintFormat("--- Testing Timeframe: %s ---", tf_name);
   
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   if(CopyRates(_Symbol, tf, 1, InpBarsToTest, rates) != InpBarsToTest)
   {
      PrintFormat("Not enough bars to test %s.", tf_name);
      return;
   }
   
   // 1. Test ADX
   int adx_handle = iADX(_Symbol, tf, InpPeriod);
   if(adx_handle == INVALID_HANDLE) return;
   
   double adx_buffer[];
   ArraySetAsSeries(adx_buffer, false);
   if(CopyBuffer(adx_handle, 0, 1, InpBarsToTest, adx_buffer) <= 0) return;
   
   // 2. Test ATR
   int atr_handle = iATR(_Symbol, tf, InpPeriod);
   if(atr_handle == INVALID_HANDLE) return;
   
   double atr_buffer[];
   ArraySetAsSeries(atr_buffer, false);
   if(CopyBuffer(atr_handle, 0, 1, InpBarsToTest, atr_buffer) <= 0) return;
   
   int adx_fails = 0, atr_fails = 0;
   int adx_first_fail = -1, adx_last_fail = -1;
   int atr_first_fail = -1, atr_last_fail = -1;
   double adx_max_diff = 0, atr_max_diff = 0;
   
   for(int i = 500; i < InpBarsToTest; i++)
   {
      // Alimentar o histórico COMPLETO até ao índice 'i'
      MqlRates slice[];
      ArrayResize(slice, i + 1);
      ArrayCopy(slice, rates, 0, 0, i + 1);
      
      double py_adx = PyADX(slice, InpPeriod);
      double native_adx = adx_buffer[i];
      
      double py_atr = PyATR(slice, InpPeriod);
      double native_atr = atr_buffer[i];
      
      double adx_diff = MathAbs(py_adx - native_adx);
      if(adx_diff > 1e-9)
      {
         if(adx_first_fail == -1) adx_first_fail = i;
         adx_last_fail = i;
         adx_fails++;
         if(adx_diff > adx_max_diff) adx_max_diff = adx_diff;
      }
      
      double atr_diff = MathAbs(py_atr - native_atr);
      if(atr_diff > 1e-9)
      {
         if(atr_first_fail == -1) atr_first_fail = i;
         atr_last_fail = i;
         atr_fails++;
         if(atr_diff > atr_max_diff) atr_max_diff = atr_diff;
      }
   }
   
   if(adx_fails == 0) Print("ADX Sweep Passed! 100% convergence over 10,000 bars.");
   else PrintFormat("[FAIL] ADX Mismatch: %d fails. First index: %d, Last index: %d, Max Diff: %.12f", adx_fails, adx_first_fail, adx_last_fail, adx_max_diff);
   
   if(atr_fails == 0) Print("ATR Sweep Passed! 100% convergence over 10,000 bars.");
   else PrintFormat("[FAIL] ATR Mismatch: %d fails. First index: %d, Last index: %d, Max Diff: %.12f", atr_fails, atr_first_fail, atr_last_fail, atr_max_diff);
   
   IndicatorRelease(adx_handle);
   IndicatorRelease(atr_handle);
}

void OnStart()
{
   Print("Starting Indicator Unit Test against Native Oracles...");
   TestTimeframe(PERIOD_M5, "M5");
   TestTimeframe(PERIOD_M15, "M15");
   TestTimeframe(PERIOD_H1, "H1");
   Print("Unit Test Complete.");
}
