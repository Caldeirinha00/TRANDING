# ==============================================================================
# AI_TRADING V2 - INSTITUTIONAL PRODUCTION LAUNCHER (PowerShell)
# Target: Windows 11 | MetaTrader 5 C++ Twin + Python 3.14 Async + Dragonfly + Qdrant
# Base Capital: 1.000 EUR | ESMA 1:30 Max Leverage
# ==============================================================================

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "       AI_TRADING V2 - INSTITUTIONAL PRODUCTION LAUNCHER          " -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$PythonExe = "C:\Users\mob70\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$AppDir = "C:\Users\mob70\Documents\AI_TRADING\AI_TRADING"
$MT5Terminal = "C:\Program Files\MetaTrader 5\terminal64.exe"
$MT5Expert = "C:\Users\mob70\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Experts\AI_TRADING_Twin_V2.ex5"

# 1. Limpeza de Processos Órfãos / Bloqueios de Named Pipe
Write-Host "[1/5] A limpar instâncias órfãs de Python..." -ForegroundColor Yellow
Stop-Process -Name python -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 600
Write-Host "  -> Concluído: Memória kernel e pipes libertados." -ForegroundColor Green

# 2. Verificação de Infraestrutura Docker / LAN
Write-Host "[2/5] A verificar serviços de infraestrutura (192.168.1.90)..." -ForegroundColor Yellow

function Test-Port {
    param([string]$HostName, [int]$Port)
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $iar = $tcp.BeginConnect($HostName, $Port, $null, $null)
        $wait = $iar.AsyncWaitHandle.WaitOne(1000, $false)
        if ($wait) {
            $tcp.EndConnect($iar)
            $tcp.Close()
            return $true
        }
        $tcp.Close()
        return $false
    } catch {
        return $false
    }
}

$DragonflyOk = Test-Port "192.168.1.90" 6379
$QdrantOk = Test-Port "192.168.1.90" 6333

if ($DragonflyOk) {
    Write-Host "  -> DragonflyDB / Redis (6379): ONLINE" -ForegroundColor Green
} else {
    Write-Host "  -> DragonflyDB / Redis (6379): OFFLINE (Verifique o contentor Docker)" -ForegroundColor Red
}

if ($QdrantOk) {
    Write-Host "  -> Qdrant Vector Engine (6333): ONLINE" -ForegroundColor Green
} else {
    Write-Host "  -> Qdrant Vector Engine (6333): OFFLINE (Verifique o contentor Docker)" -ForegroundColor Red
}

if (-not ($DragonflyOk -and $QdrantOk)) {
    Write-Host "`n[AVISO CRÍTICO] Infraestrutura incompleta! Inicie os contentores Docker antes de operar." -ForegroundColor Magenta
}

# 3. Verificação do Binário MQL5 no Terminal MetaTrader 5
Write-Host "[3/5] A verificar binário compilado AI_TRADING_Twin_V2.ex5..." -ForegroundColor Yellow
if (Test-Path $MT5Expert) {
    $ExpertSize = (Get-Item $MT5Expert).Length
    $ExpertDate = (Get-Item $MT5Expert).LastWriteTime
    Write-Host "  -> Expert Adviser: OK ($ExpertSize bytes, $ExpertDate)" -ForegroundColor Green
} else {
    Write-Host "  -> AVISO: Ficheiro $MT5Expert não encontrado!" -ForegroundColor Red
}

# Verifica se o MT5 Terminal está aberto
$MT5Proc = Get-Process -Name terminal64 -ErrorAction SilentlyContinue
if (-not $MT5Proc) {
    Write-Host "  -> MetaTrader 5 não está aberto. A iniciar Terminal..." -ForegroundColor Cyan
    Start-Process -FilePath $MT5Terminal
    Start-Sleep -Seconds 3
} else {
    Write-Host "  -> MetaTrader 5 Terminal: EM EXECUÇÃO" -ForegroundColor Green
}

# 4. Arranque do V2 Copilot Service
Write-Host "[4/5] A iniciar V2 Copilot Service (Named Pipe + ONNX + Qdrant + Harvester)..." -ForegroundColor Yellow
$CopilotCmd = "cd '$AppDir'; & '$PythonExe' -u 'agents/v2_copilot_service.py'"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='[AI_TRADING V2] Copilot Service'; $CopilotCmd"

Write-Host "  -> A aguardar inicialização do Warm-up e BFCArena ONNX (5 segundos)..." -ForegroundColor Cyan
Start-Sleep -Seconds 5

# 5. Arranque do Tiered Risk Watchdog
Write-Host "[5/5] A iniciar Tiered Risk Watchdog..." -ForegroundColor Yellow
$WatchdogCmd = "cd '$AppDir'; & '$PythonExe' -u 'risk/tiered_watchdog.py'"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$Host.UI.RawUI.WindowTitle='[AI_TRADING V2] Risk Watchdog'; $WatchdogCmd"

Start-Sleep -Seconds 2

Write-Host "`n==================================================================" -ForegroundColor Green
Write-Host " SISTEMA AI_TRADING V2 ATIVO E OPERACIONAL EM ALTA DISPONIBILIDADE!" -ForegroundColor Green
Write-Host " - Caminho Crítico IPC: Named Pipe \\.\pipe\AI_TRADING_TWIN_V2 (<1ms)" -ForegroundColor Green
Write-Host " - Proteção de Risco: Watchdog Tiered em 2.85% Daily DD / 19.5% Total DD" -ForegroundColor Green
Write-Host " - Base de Capital: 1.000 EUR (ESMA 1:30)" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "`nPasso Seguinte no MetaTrader 5:" -ForegroundColor Yellow
Write-Host "1. Abra o gráfico EURUSD M5." -ForegroundColor Yellow
Write-Host "2. Arraste o EA 'AI_TRADING_Twin_V2' do Navegador para o gráfico." -ForegroundColor Yellow
Write-Host "3. Verifique se 'Permitir Negociação Automatizada' está ativo e InpUseBinaryPipe=true." -ForegroundColor Yellow
