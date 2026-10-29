//+------------------------------------------------------------------+
//|                                             Test_V2_Includes.mq5 |
//|                                  Copyright 2026, AI_TRADING Team |
//+------------------------------------------------------------------+
#property copyright "Copyright 2026, AI_TRADING Team"
#property link      ""
#property version   "1.00"
#property script_show_inputs

#include "AutonomousDefenseManager.mqh"
#include "IPC_BinaryPipe.mqh"
#include "Regime_Microstructure_Lib.mqh"
#include "SmartOrderRouter.mqh"

void OnStart()
{
   Print("=== TEST V2 INCLUDES SANITY ===");
   CBinaryIPCClient ipc;
   CSmartOrderRouter sor(123456);
   CAutonomousDefenseManager adm(123456);
   CQuantRegimeEngine regime;
   Print("Compilation and instantiation successful!");
}
