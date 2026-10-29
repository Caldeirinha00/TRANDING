//+------------------------------------------------------------------+
//|                                      AI_TRADING_IPC_Pipe.mqh     |
//|                                  Copyright 2026, AI_TRADING Team |
//+------------------------------------------------------------------+
#property strict

#import "kernel32.dll"
   int CallNamedPipeW(
      string lpNamedPipeName,
      uchar &lpInBuffer[],
      uint nInBufferSize,
      uchar &lpOutBuffer[],
      uint nOutBufferSize,
      uint &lpBytesRead,
      uint nTimeOut
   );
   uint GetLastError();
#import

#define PIPE_NAME "\\\\.\\pipe\\AI_TRADING_IPC"
#define IPC_BUFFER_SIZE 4096

struct HandshakeResponse {
   bool   approved;
   double kelly_risk_pct;
   double sl_atr_mult;
   double tp_atr_mult;
};

//+------------------------------------------------------------------+
//| Faz o Handshake IPC via Named Pipe com latência sub-milissegundo |
//+------------------------------------------------------------------+
bool RequestPipeHandshake(
   const string direction,
   const string setup_type,
   const double h1_adx,
   const double m5_atr,
   const double m5_ema20,
   const MqlRates &candle,
   const uint timeout_ms,
   HandshakeResponse &out_resp
) {
   // Default Fail-Closed
   out_resp.approved = false;
   out_resp.kelly_risk_pct = 0.0;
   out_resp.sl_atr_mult = 1.2;
   out_resp.tp_atr_mult = 1.25;

   string clean_dir = (direction == "\"BUY\"" || direction == "BUY") ? "BUY" : "SELL";

   string payload = StringFormat(
      "{\"sym\":\"%s\",\"dir\":\"%s\",\"setup\":\"%s\",\"t\":%I64d,\"o\":%.5f,\"h\":%.5f,\"l\":%.5f,\"c\":%.5f,\"v\":%I64d,\"adx\":%.2f,\"atr\":%.5f,\"ema20\":%.5f}",
      _Symbol, clean_dir, setup_type, (long)candle.time,
      candle.open, candle.high, candle.low, candle.close, (long)candle.tick_volume,
      h1_adx, m5_atr, m5_ema20
   );

   uchar in_buf[];
   int in_len = StringToCharArray(payload, in_buf, 0, WHOLE_ARRAY, CP_UTF8);
   if(in_len <= 1) return false;
   uint send_bytes = (uint)(in_len - 1); // Exclui null terminator para JSON limpo

   uchar out_buf[];
   ArrayResize(out_buf, IPC_BUFFER_SIZE);
   uint bytes_read = 0;

   int res = CallNamedPipeW(
      PIPE_NAME,
      in_buf,
      send_bytes,
      out_buf,
      IPC_BUFFER_SIZE,
      bytes_read,
      timeout_ms
   );

   if(res == 0 || bytes_read == 0) {
      return false; // Pipe não disponível ou timeout
   }

   string resp_str = CharArrayToString(out_buf, 0, (int)bytes_read, CP_UTF8);

   // Parsing leve e determinístico (formato: "1|0.65|1.20|1.35" ou json)
   string parts[];
   if(StringSplit(resp_str, '|', parts) >= 4) {
      out_resp.approved       = (StringToInteger(parts[0]) == 1);
      out_resp.kelly_risk_pct = StringToDouble(parts[1]);
      out_resp.sl_atr_mult    = StringToDouble(parts[2]);
      out_resp.tp_atr_mult    = StringToDouble(parts[3]);
      return out_resp.approved;
   }
   else if(StringFind(resp_str, "\"approved\": true") >= 0 || StringFind(resp_str, "\"approved\":true") >= 0 ||
           StringFind(resp_str, "\"vetoed\": false") >= 0 || StringFind(resp_str, "\"vetoed\":false") >= 0) {
      out_resp.approved = true;
      return true;
   }

   return false;
}
