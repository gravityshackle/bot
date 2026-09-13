# Symbol config field reference

contract_spec.verified: false until confirmed against IB reqContractDetails
(scripts/verify_contract_specs.py). Values below are CME published specs.

session.rth: null means "no reliable RTH convention -- determine empirically".
Per the symbol spec, GC/SI-family RTH is the weakest post-floor-closure; the
spec directs using the highest-volume window rather than assumed pit hours.
scripts/empirical_rth.py computes it once Phase 1 data is cached.

session.mode: "rth_and_eth" | "continuous" (MET only)
day_boundary: the cutoff that defines "prior day" for level detection (spec S2).
