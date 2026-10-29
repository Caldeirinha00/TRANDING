//+------------------------------------------------------------------+
//|                                             IPC_BinaryPipe.mqh   |
//|                         Copyright 2026, Institutional Quant Core |
//+------------------------------------------------------------------+
#property strict
#include <Files\FilePipe.mqh>

#define PIPE_NAME "\\\\.\\pipe\\AI_TRADING_TWIN_V2"

// Alinhamento natural de 64 bits (88 bytes exatos, zero padding implícito)
struct IPCRequestPacket
{
   ulong             request_id;        // 8 bytes (offset 0)
   long              timestamp_msc;     // 8 bytes (offset 8)
   double            bid;               // 8 bytes (offset 16)
   double            ask;               // 8 bytes (offset 24)
   double            atr_m5;            // 8 bytes (offset 32)
   double            adx_h1;            // 8 bytes (offset 40)
   double            fdi_m5;            // 8 bytes (offset 48) - Fractal Dimension Index
   double            kalman_vel_z;      // 8 bytes (offset 56) - Velocidade Kalman Z-Score
   int               direction;         // 4 bytes (offset 64) - 1 = BUY, -1 = SELL
   int               spread_points;     // 4 bytes (offset 68)
   uchar             symbol[16];        // 16 bytes (offset 72..87)
};

// Alinhamento natural de 64 bits (56 bytes exatos)
struct IPCResponsePacket
{
   ulong             request_id;        // 8 bytes (offset 0)
   long              timestamp_msc;     // 8 bytes (offset 8)
   double            approved_lots;     // 8 bytes (offset 16)
   double            sl_mult_adj;       // 8 bytes (offset 24) - Modulador RAG de SL
   double            tp_mult_adj;       // 8 bytes (offset 32) - Modulador RAG de TP
   double            meta_prob;         // 8 bytes (offset 40) - Probabilidade Meta-Model
   int               veto;              // 4 bytes (offset 48) - 0 = APROVADO, 1 = VETO
   int               reason_code;       // 4 bytes (offset 52)
};

class CBinaryIPCClient
{
private:
   ulong             m_seq_counter;

public:
                     CBinaryIPCClient() : m_seq_counter(0) {}
                    ~CBinaryIPCClient() {}

   bool              QueryCopilot(const string symbol_str,
                                  const int direction,
                                  const double atr_m5,
                                  const double adx_h1,
                                  const double fdi_m5,
                                  const double kalman_vel_z,
                                  const uint timeout_ms,
                                  IPCResponsePacket &out_res)
     {
      // Inicialização fail-closed por defeito
      ZeroMemory(out_res);
      out_res.veto = 1;
      out_res.reason_code = -999;

      IPCRequestPacket req;
      ZeroMemory(req);
      req.request_id    = (++m_seq_counter) ^ (ulong)GetMicrosecondCount();
      req.timestamp_msc = (long)TimeTradeServer() * 1000 + (long)(GetTickCount64() % 1000);
      req.bid           = SymbolInfoDouble(symbol_str, SYMBOL_BID);
      req.ask           = SymbolInfoDouble(symbol_str, SYMBOL_ASK);
      req.atr_m5        = atr_m5;
      req.adx_h1        = adx_h1;
      req.fdi_m5        = fdi_m5;
      req.kalman_vel_z  = kalman_vel_z;
      req.direction     = direction;
      req.spread_points = (int)SymbolInfoInteger(symbol_str, SYMBOL_SPREAD);
      StringToCharArray(symbol_str, req.symbol, 0, 15, CP_UTF8);

      CFilePipe pipe;
      ulong start_us = GetMicrosecondCount();
      ulong timeout_us = (ulong)timeout_ms * 1000ULL;

      // Tenta abrir o Named Pipe dentro do orçamento de tempo
      int handle = INVALID_HANDLE;
      while((GetMicrosecondCount() - start_us) < timeout_us)
        {
         handle = pipe.Open(PIPE_NAME, FILE_READ | FILE_WRITE | FILE_BIN);
         if(handle != INVALID_HANDLE)
            break;
         Sleep(1);
        }

      if(handle == INVALID_HANDLE)
        {
         out_res.reason_code = -101; // Pipe indisponível (Watchdog/Fallback)
         return false;
        }

      // Escrita binária direta em RAM kernel (88 bytes)
      uint bytes_written = pipe.WriteStruct(req);
      if(bytes_written != sizeof(IPCRequestPacket))
        {
         pipe.Close();
         out_res.reason_code = -102;
         return false;
        }

      // Leitura síncrona bloqueante/rápida da resposta (56 bytes)
      uint bytes_read = pipe.ReadStruct(out_res);
      pipe.Close();

      if(bytes_read != sizeof(IPCResponsePacket) || out_res.request_id != req.request_id)
        {
         out_res.veto = 1;
         out_res.reason_code = -103; // Mismatch ou pacote truncado
         return false;
        }

      if((GetMicrosecondCount() - start_us) > timeout_us)
        {
         out_res.veto = 1;
         out_res.reason_code = -104; // SLA Timeout excedido
         return false;
        }

      return (out_res.veto == 0 && out_res.approved_lots > 0.0);
     }
};
