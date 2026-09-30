"""Turkish-aware text normalisation and the dimension/scale/tag parsers."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from archrender.understand.text import (
    fold,
    is_scale_label,
    normalise_tag,
    parse_dimension,
    parse_number,
    parse_scale,
    tr_lower,
    tr_upper,
)


def test_turkish_case_rules() -> None:
    assert tr_lower("IŞIK İÇ") == "ışık iç"
    assert tr_upper("ışık iç") == "IŞIK İÇ"
    assert fold("GENİŞLİK (cm)") == fold("Genişlik cm") == "genislik cm"
    assert fold("YÜKSEKLİK") == "yukseklik" and fold("Çalışma Odası") == "calisma odasi"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("3,50", 3.5),
        ("3.50", 3.5),
        ("350", 350.0),
        ("1.234,5", 1234.5),
        ("1,234.5", 1234.5),
        ("12.400", 12.4),
        ("±0,00", 0.0),
        ("-0,45", -0.45),
        ("abc", None),
        ("3,5,0", None),
    ],
)
def test_parse_number(text: str, value: float | None) -> None:
    assert parse_number(text) == value


def test_dot_grouping_in_turkish_locale() -> None:
    assert parse_number("12.400", locale="tr") == 12400.0
    assert parse_number("1.234.567", locale="tr") == 1234567.0
    assert parse_number("3,50", locale="tr") == 3.5


@given(st.integers(min_value=0, max_value=99_999))
def test_decimal_comma_round_trip(cents: int) -> None:
    v = cents / 100
    s = f"{v:.2f}".replace(".", ",")
    assert parse_number(s) == pytest.approx(v)
    d = parse_dimension(s)
    assert d is not None and d.metres[0] == pytest.approx(v) and d.unit in ("m", "ambiguous")


@pytest.mark.parametrize(
    ("text", "hint", "metres", "unit"),
    [
        ("3,50", None, (3.5,), "m"),
        ("350", None, (3.5, 0.35), "ambiguous"),
        ("3500", None, (3.5, 35.0), "ambiguous"),
        ("350", "cm", (3.5,), "cm"),
        ("2,75 m", None, (2.75,), "m"),
        ("90cm", None, (0.9,), "cm"),
        ("12'-6\"", None, (3.81,), "ft-in"),
        ('6"', None, (0.1524,), "ft-in"),
        ("10' 3 1/2\"", None, (3.1369,), "ft-in"),
    ],
)
def test_parse_dimension(text: str, hint: str | None, metres: tuple[float, ...], unit: str) -> None:
    d = parse_dimension(text, unit_hint=hint)
    assert d is not None and d.unit == unit
    assert d.metres == pytest.approx(metres, abs=1e-4)


@pytest.mark.parametrize(
    ("text", "scale"),
    [
        ("Ölçek 1/50", 50),
        ("ÖLÇEK: 1:100", 100),
        ("M 1:50", 50),
        ("SCALE 1:20 @A1", 20),
        ('1/4" = 1\'-0"', 48),
        ("1/8\" = 1'", 96),
        ("1/2", 2),
        ("Sayfa 1/3", 3),  # pattern only: callers also require a scale label
        ("12.03.2026", None),
        ("K1", None),
    ],
)
def test_parse_scale(text: str, scale: float | None) -> None:
    assert parse_scale(text) == scale


def test_scale_labels() -> None:
    assert is_scale_label("ÖLÇEK") and is_scale_label("Scale 1:50") and is_scale_label("M 1:100")
    assert not is_scale_label("Sayfa 1/3") and not is_scale_label("PAFTA NO")


@pytest.mark.parametrize(
    ("text", "tag"),
    [
        ("K1", "K1"),
        ("k-01", "K1"),
        ("P 12", "P12"),
        ("D3", "D3"),
        ("W10A", "W10A"),
        ("Salon", None),
        ("3,50", None),
    ],
)
def test_normalise_tag(text: str, tag: str | None) -> None:
    assert normalise_tag(text) == tag
