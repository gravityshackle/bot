"""Guard tests for contract resolution -- offline, with stubbed IB details.

The scenario these exist for is real and was hit during Phase 1: IB returns
micro and full-size silver together under ib_symbol "SI", differing only by
multiplier. Picking the wrong one is a 5x position-sizing error that raises
no exception anywhere.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from data.sources.ibkr_contracts import (
    AmbiguousContract,
    ContractSpecMismatch,
    verify_contract_details,
)


def detail(local, multiplier, min_tick, expiry="20260925"):
    return SimpleNamespace(
        minTick=min_tick,
        contract=SimpleNamespace(
            localSymbol=local,
            multiplier=str(multiplier),
            lastTradeDateOrContractMonth=expiry,
        ),
    )


def load(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


SIL_CFG = load("config/symbols/SIL.yaml")
SI_CFG = load("config/symbols/full_size/SI.yaml")
MES_CFG = load("config/symbols/MES.yaml")


# Exactly what IB actually returns for SI/COMEX: both sizes, interleaved.
def mixed_silver_details():
    return [
        detail("SIU6", 5000, 0.005, "20260928"),
        detail("SILU6", 1000, 0.005, "20260925"),
        detail("SIZ6", 5000, 0.005, "20261228"),
        detail("SILZ6", 1000, 0.005, "20261224"),
    ]


# --------------------------------------------------------------------------
# the SIL trap
# --------------------------------------------------------------------------

def test_micro_silver_resolves_to_multiplier_1000_only():
    got = verify_contract_details(mixed_silver_details(), SIL_CFG)
    assert [r.raw_symbol for r in got] == ["SILU6", "SILZ6"]
    assert {r.multiplier for r in got} == {1000.0}
    assert {r.tick_value for r in got} == {5.0}


def test_full_size_silver_resolves_to_multiplier_5000_only():
    got = verify_contract_details(mixed_silver_details(), SI_CFG)
    assert [r.raw_symbol for r in got] == ["SIU6", "SIZ6"]
    assert {r.multiplier for r in got} == {5000.0}
    assert {r.tick_value for r in got} == {25.0}


def test_micro_and_full_size_never_overlap():
    micro = {r.raw_symbol for r in verify_contract_details(mixed_silver_details(), SIL_CFG)}
    full = {r.raw_symbol for r in verify_contract_details(mixed_silver_details(), SI_CFG)}
    assert micro.isdisjoint(full)


def test_ambiguous_symbol_without_disambiguator_is_refused():
    """The failure mode this whole module exists to prevent."""
    cfg = dict(SIL_CFG)
    cfg.pop("ib_multiplier")
    cfg.pop("ib_local_symbol_root", None)
    with pytest.raises(AmbiguousContract, match="micro-vs-full-size trap"):
        verify_contract_details(mixed_silver_details(), cfg)


def test_requested_multiplier_absent_raises():
    cfg = dict(SIL_CFG, ib_multiplier=2500)
    with pytest.raises(ContractSpecMismatch, match=r"no contract with multiplier 2500"):
        verify_contract_details(mixed_silver_details(), cfg)


# --------------------------------------------------------------------------
# the three redundant numeric checks
# --------------------------------------------------------------------------

def test_min_tick_disagreement_raises():
    bad = [detail("SILU6", 1000, 0.01)]         # IB tick != config tick_size
    with pytest.raises(ContractSpecMismatch, match="minTick"):
        verify_contract_details(bad, SIL_CFG)


def test_multiplier_not_equal_to_point_value_raises():
    cfg = dict(SIL_CFG)
    cfg["contract_spec"] = dict(SIL_CFG["contract_spec"], point_value=999.0)
    with pytest.raises(ContractSpecMismatch, match="point_value"):
        verify_contract_details([detail("SILU6", 1000, 0.005)], cfg)


def test_tick_value_mismatch_reports_the_sizing_error_factor():
    """A wrong tick_value is a silent position-sizing multiplier -- say so."""
    cfg = dict(SIL_CFG)
    cfg["contract_spec"] = dict(SIL_CFG["contract_spec"], tick_value=1.0)
    with pytest.raises(ContractSpecMismatch, match=r"5\.00x"):
        verify_contract_details([detail("SILU6", 1000, 0.005)], cfg)


def test_local_symbol_root_enforced():
    wrong = [detail("SIU6", 1000, 0.005)]       # right multiplier, wrong root
    with pytest.raises(ContractSpecMismatch, match="localSymbol root"):
        verify_contract_details(wrong, SIL_CFG)


def test_empty_details_raises():
    with pytest.raises(ContractSpecMismatch, match="no contracts"):
        verify_contract_details([], SIL_CFG)


# --------------------------------------------------------------------------
# unambiguous symbols still work without a disambiguator
# --------------------------------------------------------------------------

def test_unambiguous_symbol_needs_no_multiplier():
    cfg = dict(MES_CFG)
    cfg.pop("ib_multiplier", None)
    got = verify_contract_details(
        [detail("MESZ6", 5, 0.25, "20261218"),
         detail("MESH7", 5, 0.25, "20270319")], cfg)
    assert [r.raw_symbol for r in got] == ["MESZ6", "MESH7"]
    assert {r.tick_value for r in got} == {1.25}


def test_results_are_sorted_front_contract_first():
    got = verify_contract_details(
        [detail("MESH7", 5, 0.25, "20270319"),
         detail("MESZ6", 5, 0.25, "20261218")], MES_CFG)
    assert [r.expiry for r in got] == ["20261218", "20270319"]


@pytest.mark.parametrize("name", ["MES", "MNQ", "MYM", "MCL", "MGC", "SIL", "MET"])
def test_every_in_scope_config_is_internally_consistent(name):
    """point_value * tick_size must equal tick_value, or sizing is wrong."""
    cfg = load(f"config/symbols/{name}.yaml")
    cs = cfg["contract_spec"]
    assert cs["point_value"] * cs["tick_size"] == pytest.approx(cs["tick_value"])
    assert cs["verified"] is True
