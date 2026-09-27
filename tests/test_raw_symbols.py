"""CME raw-symbol parsing -- the contract year behind the roll ordering.

ContractMeta.sort_key orders contracts for the roll logic, which only ever
moves forward to a later contract. A contract dated a decade wrong sorts as the
furthest-out one and every roll through it goes wrong, silently.

The original resolver picked the single-digit decade nearest TODAY's year.
That is clock-dependent: in 2026 a 2021 "...Z1" tied between 2021 and 2031 and
happened to resolve right; from 2027 on it resolves to 2031. The 5-year Phase 4
window starts in September 2021, so this would have misdated every 2021
contract on any run from next year on.
"""
from __future__ import annotations

import datetime as real_datetime

import pytest

from data.sources import base
from data.sources.base import SchemaError, is_outright, parse_raw_symbol


@pytest.mark.parametrize("raw,near,expected", [
    ("MESZ1", 2021, ("Z", 2021)),
    ("MESH6", 2026, ("H", 2026)),
    ("MESZ5", 2025, ("Z", 2025)),
    ("MESZ25", 1999, ("Z", 2025)),     # two-digit years ignore the reference
])
def test_single_digit_year_resolves_against_the_reference(raw, near, expected):
    assert parse_raw_symbol(raw, "MES", near_year=near) == expected


def test_a_contract_year_one_off_its_expiration_year_still_resolves():
    """MCL's January contract expires in December of the prior year, so the
    reference (expiration year) can be one below the contract year."""
    assert parse_raw_symbol("MCLF2", "MCL", near_year=2021) == ("F", 2022)
    assert parse_raw_symbol("MCLF0", "MCL", near_year=2029) == ("F", 2030)


def test_resolution_does_not_depend_on_todays_date(monkeypatch):
    """Regression: the old resolver read the clock. With today set to 2027, a
    2021 contract came back as 2031."""
    class Clock2027(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2027, 6, 1)
    # raising=False: the module no longer reads the clock at all; if it ever
    # starts to again, it will see 2027 here
    monkeypatch.setattr(base, "datetime", Clock2027, raising=False)
    assert parse_raw_symbol("MESZ1", "MES", near_year=2021) == ("Z", 2021)


def test_the_reference_year_is_required():
    with pytest.raises(TypeError):
        parse_raw_symbol("MESZ1", "MES")


@pytest.mark.parametrize("raw,ok", [
    ("MESZ1", True), ("MESH26", True), ("MESZ1-MESH2", False),
    ("MES", False), ("MESQX", False),
])
def test_outright_check_needs_no_year(raw, ok):
    """Filtering spreads out of parent symbology only needs to know whether a
    symbol is an outright, not which decade it belongs to."""
    assert is_outright(raw, "MES") is ok


def test_unparseable_symbols_still_raise():
    with pytest.raises(SchemaError):
        parse_raw_symbol("MESQX", "MES", near_year=2026)
