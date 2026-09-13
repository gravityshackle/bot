"""Read-only probe: is the listening IB Gateway socket a PAPER or LIVE account?

Connects with readonly=True (the client refuses to transmit orders) and calls
managedAccounts() only. IB account-number convention: DU/DF prefix = paper,
U prefix = live.
"""
import sys
from ib_async import IB

PORTS = [4002, 4001]

for port in PORTS:
    ib = IB()
    try:
        ib.connect("127.0.0.1", port, clientId=17, timeout=10, readonly=True)
    except Exception as exc:
        print(f"port {port}: no connection ({type(exc).__name__})")
        continue

    accts = list(ib.managedAccounts())
    kinds = {a: ("PAPER" if a.upper().startswith(("DU", "DF")) else "LIVE")
             for a in accts}
    print(f"port {port}: CONNECTED  serverVersion={ib.client.serverVersion()}")
    for a, k in kinds.items():
        print(f"   account {a[:2]}{'*' * (len(a) - 2)}  ->  {k}")
    verdict = "PAPER" if kinds and all(v == "PAPER" for v in kinds.values()) else "LIVE"
    print(f"   VERDICT: {verdict}")
    ib.disconnect()
    sys.exit(0 if verdict == "PAPER" else 2)

print("no IB socket reachable on any tried port")
sys.exit(1)
