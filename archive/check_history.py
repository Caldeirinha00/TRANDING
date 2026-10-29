"""Quick script to pull and analyze all closed trades from MT5."""
import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()

from_date = datetime.now() - timedelta(days=30)
to_date = datetime.now()
deals = mt5.history_deals_get(from_date, to_date)

if not deals:
    print("No deals found in the last 30 days.")
    mt5.shutdown()
    exit()

print(f"Total raw deals: {len(deals)}\n")

wins = 0
losses = 0
total_profit = 0.0
total_loss = 0.0
by_symbol = {}

for d in deals:
    d = d._asdict()
    # entry == 1 means "out" (closing a position)
    if d["entry"] == 1:
        profit = d["profit"] + d["swap"] + d.get("commission", 0.0) + d.get("fee", 0.0)
        symbol = d["symbol"]
        direction = "BUY" if d["type"] == 0 else "SELL"
        t = datetime.fromtimestamp(d["time"])
        print(f"  {t:%Y-%m-%d %H:%M}  {symbol:10}  {direction:4}  Vol:{d['volume']:.2f}  P/L:{profit:+.2f}  Comment:{d.get('comment','')}")

        if symbol not in by_symbol:
            by_symbol[symbol] = {"wins": 0, "losses": 0, "net": 0.0}

        if profit >= 0:
            wins += 1
            total_profit += profit
            by_symbol[symbol]["wins"] += 1
        else:
            losses += 1
            total_loss += profit
            by_symbol[symbol]["losses"] += 1
        by_symbol[symbol]["net"] += profit

n = wins + losses
print("\n" + "=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
if n > 0:
    print(f"  Trades: {n}  |  Wins: {wins}  |  Losses: {losses}  |  WR: {wins/n*100:.1f}%")
    print(f"  Gross Profit: {total_profit:+.2f}")
    print(f"  Gross Loss:   {total_loss:+.2f}")
    print(f"  Net P/L:      {total_profit + total_loss:+.2f}")
    if losses > 0 and total_loss != 0:
        print(f"  Profit Factor: {abs(total_profit / total_loss):.2f}")
    if wins > 0:
        print(f"  Avg Win:  {total_profit / wins:+.2f}")
    if losses > 0:
        print(f"  Avg Loss: {total_loss / losses:+.2f}")
else:
    print("  No closed trades found.")

print("\nPER SYMBOL:")
print("-" * 60)
for sym, data in sorted(by_symbol.items(), key=lambda x: x[1]["net"]):
    sn = data["wins"] + data["losses"]
    wr = data["wins"] / sn * 100 if sn > 0 else 0
    print(f"  {sym:10}  W:{data['wins']}  L:{data['losses']}  WR:{wr:.0f}%  Net:{data['net']:+.2f}")

mt5.shutdown()
