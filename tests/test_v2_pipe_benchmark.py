"""
tests/test_v2_pipe_benchmark.py
Simula o cliente C++ MQL5 e mede a latência real do Named Pipe \\\\.\\pipe\\AI_TRADING_TWIN_V2
"""
import time
import struct
import statistics
import win32file
import win32pipe
import pywintypes

PIPE_NAME = r"\\.\pipe\AI_TRADING_TWIN_V2"

# < = Little-Endian | Q = uint64 | q = int64 | 6d = 6 float64 | 2i = 2 int32 | 16s = char[16] (88 bytes)
REQ_STRUCT = struct.Struct("<Qq6d2i16s")
# < = Little-Endian | Q = uint64 | q = int64 | 4d = 4 float64 | 2i = 2 int32 (56 bytes)
RES_STRUCT = struct.Struct("<Qq4d2i")


def run_pipe_benchmark(n_iterations: int = 30) -> None:
    print("=" * 60)
    print(f"BENCHMARK NAMED PIPE AI_TRADING_TWIN_V2 ({n_iterations} iterações)")
    print("=" * 60)

    latencies_ms = []
    approved_count = 0

    for i in range(n_iterations):
        t0 = time.perf_counter()
        try:
            # Abre conexão ao Named Pipe (emula MQL5 CFilePipe::Open)
            handle = win32file.CreateFile(
                PIPE_NAME,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None,
            )

            req_id = i + 1
            ts_msc = int(time.time() * 1000)
            bid = 1.08520
            ask = 1.08535
            atr_m5 = 0.00095
            adx_h1 = 28.5
            fdi_m5 = 1.38
            kalman_vel_z = 1.65
            direction = 1  # BUY
            spread_pts = 15
            symbol_b = b"EURUSD\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"

            req_payload = REQ_STRUCT.pack(
                req_id, ts_msc, bid, ask, atr_m5, adx_h1,
                fdi_m5, kalman_vel_z, direction, spread_pts, symbol_b
            )

            # Envia pedido de 88 bytes
            win32file.WriteFile(handle, req_payload)

            # Recebe resposta de 56 bytes
            hr, res_payload = win32file.ReadFile(handle, RES_STRUCT.size)
            t1 = time.perf_counter()

            win32file.CloseHandle(handle)

            if len(res_payload) == RES_STRUCT.size:
                unpacked = RES_STRUCT.unpack(res_payload)
                res_req_id = unpacked[0]
                approved_lots = unpacked[2]
                sl_mult = unpacked[3]
                tp_mult = unpacked[4]
                meta_prob = unpacked[5]
                veto = unpacked[6]
                reason = unpacked[7]

                if veto == 0 and approved_lots > 0:
                    approved_count += 1

                lat_ms = (t1 - t0) * 1000.0
                latencies_ms.append(lat_ms)
            else:
                print(f"[Iteração {i+1}] Pacote de resposta inválido: {len(res_payload)} bytes.")

        except pywintypes.error as e:
            print(f"[ERRO DE CONEXÃO PIPE] {e}. Certifique-se de que o v2_copilot_service está a correr.")
            return

    if not latencies_ms:
        print("Nenhuma iteração concluída com sucesso.")
        return

    sorted_lats = sorted(latencies_ms)
    p50 = statistics.median(latencies_ms)
    p95 = sorted_lats[min(len(sorted_lats) - 1, int(len(sorted_lats) * 0.95))]
    p99 = sorted_lats[min(len(sorted_lats) - 1, int(len(sorted_lats) * 0.99))]

    print(f"Iterações Realizadas: {len(latencies_ms)}/{n_iterations}")
    print(f"Aprovados:            {approved_count}/{len(latencies_ms)}")
    print(f"Latência Iteração 1:  {latencies_ms[0]:.3f} ms (Cold-Start Verification)")
    print(f"Latência Mínima:      {min(latencies_ms):.3f} ms")
    print(f"P50 (Mediana):        {p50:.3f} ms")
    print(f"P95:                  {p95:.3f} ms")
    print(f"P99:                  {p99:.3f} ms")
    print(f"Latência Máxima:      {max(latencies_ms):.3f} ms")
    print("=" * 60)
    print("VERIFICAÇÃO DE SLA (Teto = 3000 ms):")
    if p99 < 15.0:
        print(f"-> EXCELENTE: P99 ({p99:.2f}ms) está >200x abaixo do SLA de 3000ms do MQL5!")
    else:
        print(f"-> AVISO: P99 ({p99:.2f}ms) requer otimização.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipe_benchmark(30)
