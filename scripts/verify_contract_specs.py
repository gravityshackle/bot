"""Pull verified contract specs + session hours from IB (read-only).

The symbol spec says tick size/value/multiplier are exchange-fixed facts to be
hardcoded as *verified* constants, and that session times must be confirmed
against IB's own settings rather than assumed. This script is that verification
step -- it only calls reqContractDetails, it never places or modifies an order.
"""
import sys
from collections import defaultdict

from ib_async import IB, Future

HOST, PORT, CLIENT_ID = "127.0.0.1", 4002, 17

# (symbol, exchange) for the seven in-scope micros
MICROS = [
    ("MES", "CME"), ("MNQ", "CME"), ("MYM", "CBOT"),
    ("MCL", "NYMEX"), ("MGC", "COMEX"), ("SIL", "COMEX"),
    ("MET", "CME"),
]


def main() -> int:
    ib = IB()
    try:
        ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15, readonly=True)
    except Exception as exc:
        print(f"CONNECT FAILED {HOST}:{PORT} -> {exc}")
        return 1

    print(f"connected: {ib.isConnected()}  server={ib.client.serverVersion()}\n")

    for sym, exch in MICROS:
        try:
            details = ib.reqContractDetails(Future(symbol=sym, exchange=exch))
        except Exception as exc:
            print(f"{sym}: request failed -> {exc}\n")
            continue
        if not details:
            print(f"{sym}: NO CONTRACTS RETURNED (symbol/exchange wrong?)\n")
            continue

        details.sort(key=lambda d: d.contract.lastTradeDateOrContractMonth)
        d0 = details[0]
        c0 = d0.contract

        # min tick and multiplier are constant across expiries; assert that.
        ticks = {d.minTick for d in details}
        mults = {d.contract.multiplier for d in details}

        print(f"=== {sym} ({exch}) ===")
        print(f"  longName        : {d0.longName}")
        print(f"  minTick         : {sorted(ticks)}")
        print(f"  multiplier      : {sorted(mults)}")
        print(f"  tick_value      : {float(c0.multiplier) * d0.minTick:.4f} "
              f"(multiplier x minTick)")
        print(f"  currency        : {c0.currency}")
        print(f"  timeZoneId      : {d0.timeZoneId}")
        print(f"  tradingHours    : {(d0.tradingHours or '')[:110]}")
        print(f"  liquidHours     : {(d0.liquidHours or '')[:110]}")
        print(f"  contractMonth   : {d0.contractMonth}  realExpiry={d0.realExpirationDate}")

        months = [d.contract.lastTradeDateOrContractMonth for d in details[:14]]
        print(f"  n_listed        : {len(details)}")
        print(f"  next_expiries   : {months}")

        # month-code cycle, to check quarterly vs monthly listing
        cycle = sorted({m[4:6] for m in
                        (d.contract.lastTradeDateOrContractMonth for d in details)})
        print(f"  listed_months   : {cycle}")
        print()

    ib.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
