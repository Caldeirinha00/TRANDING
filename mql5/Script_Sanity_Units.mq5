//+------------------------------------------------------------------+
//|                                        Script_Sanity_Units.mq5 |
//+------------------------------------------------------------------+
#property script_show_inputs

input double InpCommissionPerLotRoundTurn = 7.0;

void OnStart()
{
   string symbols[] = {"EURUSD", "USDJPY", "XAUUSD", "US30", "NAS100"}; 
   
   Print("=========================================================================");
   Print("Symbol | point | tick_size | tick_value | commission_price_offset | money_per_point_per_lot");
   Print("=========================================================================");
   
   for(int i=0; i<ArraySize(symbols); i++)
   {
      string sym = symbols[i];
      if(!SymbolSelect(sym, true))
      {
         Print(sym, " - Could not select symbol in Market Watch");
         continue;
      }
      
      double point = SymbolInfoDouble(sym, SYMBOL_POINT);
      double tick_size = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
      double tick_value = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
      
      if(tick_value <= 0)
      {
         Print(sym, " - tick_value is zero/invalid. Please check symbol properties.");
         continue;
      }
      
      double commission_price_offset = (InpCommissionPerLotRoundTurn * tick_size) / tick_value;
      double money_per_point_per_lot = tick_value * (point / tick_size);
      
      string output = StringFormat("%s | %f | %f | %f | %f | %f",
                                   sym, point, tick_size, tick_value, commission_price_offset, money_per_point_per_lot);
      Print(output);
   }
   Print("=========================================================================");
}
