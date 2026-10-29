# Guia de Execução no Servidor Linux (CLI)

Esta pasta é 100% isolada e adaptada para rodar no seu servidor Linux com o mínimo de consumo de RAM.

## 1. Como iniciar os contentores (Bases de Dados + MT5 Isolado)

Na pasta do projeto no servidor:
`ash
docker compose up -d
`

Isto vai levantar em background:
- **QuestDB** (dados e telemetria)
- **Dragonfly** (Redis ultra-rápido com RAM limitada a 256MB)
- **Qdrant** (base de dados vetorial)
- **mt5** (contentor fechado com Wine + MT5 + ponte RPyC interna na porta 18812, sem VNC nem portas web abertas)

## 2. Instalar dependências Python no Linux

`ash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-linux.txt
`

## 3. Iniciar o Bot

`ash
python main.py
`

O bot vai conectar-se automaticamente ao MT5 através da porta interna 18812 e executar normalmente.\n