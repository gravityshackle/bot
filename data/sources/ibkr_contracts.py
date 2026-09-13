"""Contract resolution with hard spec verification.

Why this exists: IB exposes micro silver (1,000oz, $5.00/tick) and full-size
silver (5,000oz, $25.00/tick) under the SAME ib_symbol "SI". A plain
reqContractDetails(Future("SI", "COMEX")) returns both, mixed, and taking
details[0] gets you whichever IB happened to order first. Trading a contract
whose multiplier is 5x what your sizing math assumed turns a $50 risk budget
into a $250 one with nothing anywhere reporting an error.

So nothing here trusts the symbol alone. Every resolved contract is checked
against the config's own numbers before it is handed to any caller:

    minTick               == tick_size
    multiplier            == point_value
    multiplier * minTick  == tick_value

Those three are redundant on purpose -- they are cheap, and they cross-check
each other. If IB ever changes a listing, this raises at startup instead of
silently resizing positions.
"""
from __future__ import annotations

from dataclasses import dataclass

TOL = 1e-9


class ContractSpecMismatch(RuntimeError):
    """A resolved IB contract disagrees with its symbol config."""


class AmbiguousContract(RuntimeError):
    """Multiple distinct contract shapes matched and no disambiguator was set."""


@dataclass(frozen=True)
class ResolvedContract:
    raw_symbol: str          # localSymbol, e.g. "SILU6"
    multiplier: float
    min_tick: float
    tick_value: float
    expiry: str
    detail: object = None    # the underlying ib_async ContractDetails


def _multiplier_of(detail) -> float:
    return float(detail.contract.multiplier)


def verify_contract_details(details, cfg: dict) -> list[ResolvedContract]:
    """Filter and hard-verify IB contract details against a symbol config.

    Pure function -- takes the details list rather than an IB connection, so
    the guard is unit-testable offline. Raises rather than returning a partial
    result: there is no safe "mostly right" contract.
    """
    symbol = cfg.get("symbol", "?")
    spec = cfg["contract_spec"]
    want_mult = cfg.get("ib_multiplier")
    want_root = cfg.get("ib_local_symbol_root")

    if not details:
        raise ContractSpecMismatch(
            f"{symbol}: IB returned no contracts for ib_symbol="
            f"{cfg.get('ib_symbol')!r} on {cfg.get('exchange')!r}"
        )

    found_mults = sorted({_multiplier_of(d) for d in details})

    # 1. Disambiguate by multiplier when the config declares one.
    if want_mult is not None:
        kept = [d for d in details if abs(_multiplier_of(d) - float(want_mult)) < TOL]
        if not kept:
            raise ContractSpecMismatch(
                f"{symbol}: no contract with multiplier {want_mult} under "
                f"ib_symbol={cfg.get('ib_symbol')!r}; IB returned multipliers "
                f"{found_mults}"
            )
        dropped = len(details) - len(kept)
        details = kept
    else:
        # 2. No disambiguator: refuse if the symbol is genuinely ambiguous.
        if len(found_mults) > 1:
            raise AmbiguousContract(
                f"{symbol}: ib_symbol={cfg.get('ib_symbol')!r} returned "
                f"{len(details)} contracts spanning multipliers {found_mults}. "
                "Set ib_multiplier in the symbol config to disambiguate -- "
                "this is the micro-vs-full-size trap (see SIL/SI)."
            )
        dropped = 0

    # 3. localSymbol root, when declared (Globex code may differ from ib_symbol).
    if want_root:
        bad = [d.contract.localSymbol for d in details
               if not str(d.contract.localSymbol).startswith(want_root)]
        if bad:
            raise ContractSpecMismatch(
                f"{symbol}: expected localSymbol root {want_root!r}, got {bad[:5]}"
            )

    # 4. The three redundant numeric checks.
    for d in details:
        mult, tick = _multiplier_of(d), float(d.minTick)
        local = d.contract.localSymbol
        if abs(tick - float(spec["tick_size"])) > TOL:
            raise ContractSpecMismatch(
                f"{symbol}/{local}: IB minTick {tick} != config tick_size "
                f"{spec['tick_size']}"
            )
        if abs(mult - float(spec["point_value"])) > TOL:
            raise ContractSpecMismatch(
                f"{symbol}/{local}: IB multiplier {mult} != config point_value "
                f"{spec['point_value']}"
            )
        implied = mult * tick
        if abs(implied - float(spec["tick_value"])) > TOL:
            raise ContractSpecMismatch(
                f"{symbol}/{local}: multiplier x minTick = {implied} != config "
                f"tick_value {spec['tick_value']} -- position sizing would be "
                f"wrong by {implied / float(spec['tick_value']):.2f}x"
            )

    resolved = [
        ResolvedContract(
            raw_symbol=str(d.contract.localSymbol),
            multiplier=_multiplier_of(d),
            min_tick=float(d.minTick),
            tick_value=_multiplier_of(d) * float(d.minTick),
            expiry=str(d.contract.lastTradeDateOrContractMonth),
            detail=d,
        )
        for d in details
    ]
    resolved.sort(key=lambda r: r.expiry)
    return resolved


def resolve_futures_contracts(ib, cfg: dict) -> list[ResolvedContract]:
    """Query IB and return verified contracts for a symbol config, front first."""
    from ib_async import Future

    details = ib.reqContractDetails(Future(
        symbol=cfg["ib_symbol"],
        exchange=cfg["exchange"],
        currency=cfg["contract_spec"].get("currency", "USD"),
    ))
    return verify_contract_details(details, cfg)


def assert_paper_account(ib) -> str:
    """Refuse to proceed unless the Gateway session is a paper account.

    Paired with config/ibkr.yaml safety.require_paper_account. IB paper
    accounts are DU*/DF*; live accounts are U*.
    """
    accounts = list(ib.managedAccounts())
    if not accounts:
        raise RuntimeError("IB returned no managed accounts; cannot verify paper mode")
    live = [a for a in accounts if not a.upper().startswith(("DU", "DF"))]
    if live:
        raise RuntimeError(
            f"refusing to run: {len(live)} non-paper account(s) on this Gateway "
            f"session (first: {live[0][:2]}{'*' * (len(live[0]) - 2)}). "
            "Port 4002 is paper, 4001 is live."
        )
    return accounts[0]
