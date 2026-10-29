import subprocess
import sys
import threading
import time
import os
from dotenv import load_dotenv

def prefix_output(process, prefix):
    """Lê a saída do processo linha a linha e adiciona o prefixo."""
    for line in iter(process.stdout.readline, b''):
        try:
            decoded = line.decode('utf-8', errors='replace').strip()
            if decoded:
                print(f"{prefix} {decoded}")
        except Exception:
            pass

def main():
    print("🚀 A Iniciar o Sistema AI Trading (Headless Engine)...")
    
    # 0. Limpar o lock do Watchdog automaticamente se existir
    print("-> Limpando Watchdog lock antigo (se existir)...")
    try:
        import redis
        load_dotenv()
        host = os.getenv("DRAGONFLY_HOST", "localhost")
        port = int(os.getenv("DRAGONFLY_PORT", 6379))
        
        r = redis.Redis(host=host, port=port, db=0, socket_timeout=2)
        r.delete("system:lock")
        print("[SUCESSO] Lock do sistema limpo.")
    except Exception as e:
        print(f"[AVISO] Não foi possível limpar o lock (Redis pode estar desligado): {e}")
    
    # 1. Start the Core Trading System directly
    print("-> Iniciando AI Trading Core (main.py)...")
    core_process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    t_core = threading.Thread(target=prefix_output, args=(core_process, "[TRADING]"), daemon=True)
    t_core.start()

    print("\n✅ CORE DE TRADING ATIVO! Prime CTRL+C para fechar em segurança.\n" + "="*60)
    
    try:
        while True:
            time.sleep(1)
            if core_process.poll() is not None:
                print("\n[AVISO] O processo TRADING fechou.")
                break
    except KeyboardInterrupt:
        print("\n\n🛑 A encerrar o processo...")
    finally:
        try:
            core_process.terminate()
        except Exception:
            pass
        print("Tudo encerrado com sucesso. Adeus!")

if __name__ == "__main__":
    main()
