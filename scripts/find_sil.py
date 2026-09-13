"""SIL came back with no security definition. Find the real IB symbol."""
from ib_async import IB, Future

ib = IB()
ib.connect("127.0.0.1", 4002, clientId=19, timeout=15, readonly=True)

print("--- reqMatchingSymbols('silver') ---")
for d in ib.reqMatchingSymbols("silver") or []:
    c = d.contract
    types = {dt for dt in (d.derivativeSecTypes or [])}
    if "FUT" in types or c.secType == "FUT":
        print(f"  symbol={c.symbol:<8} secType={c.secType:<6} exch={c.primaryExchange or c.exchange:<10} "
              f"desc={getattr(d,'description','')} derivs={sorted(types)}")

print("\n--- direct probes ---")
for sym, exch in [("SIL", "COMEX"), ("SIL", "NYMEX"), ("SIL", ""),
                  ("MSI", "COMEX"), ("SI", "COMEX")]:
    try:
        ds = ib.reqContractDetails(Future(symbol=sym, exchange=exch or ""))
    except Exception as exc:
        print(f"  {sym:<4}/{exch or '(any)':<7} -> error {exc}")
        continue
    if not ds:
        print(f"  {sym:<4}/{exch or '(any)':<7} -> none")
        continue
    d0 = ds[0]
    print(f"  {sym:<4}/{exch or '(any)':<7} -> {len(ds):>3} contracts | {d0.longName} | "
          f"minTick={d0.minTick} mult={d0.contract.multiplier} "
          f"tv=${float(d0.contract.multiplier)*d0.minTick:.2f} | tz={d0.timeZoneId} "
          f"| exch={d0.contract.exchange} | localSym={d0.contract.localSymbol}")

ib.disconnect()
