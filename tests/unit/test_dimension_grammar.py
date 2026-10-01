"""Dimension-string grammar (property tests): metric decimals with either separator, unitless
centimetres/millimetres, thousands separators (``1.234,5`` / ``1,234.5`` / ``12 345``), explicit
units, and feet-inches with fractions. The true length is always among the readings; the
scale RANSAC picks the reading that agrees with the drawing."""

from __future__ import annotations

from fractions import Fraction

from hypothesis import given, settings
from hypothesis import strategies as st

from archrender.understand.text import parse_dimension


def _has(text: str, metres: float) -> bool:
    d = parse_dimension(text)
    return d is not None and any(abs(v - metres) <= 1e-9 * max(1.0, metres) for v in d.metres)


@given(cm=st.integers(min_value=5, max_value=9999), sep=st.sampled_from([",", "."]))
@settings(max_examples=300)
def test_metres_with_two_decimals(cm: int, sep: str) -> None:
    text = f"{cm // 100}{sep}{cm % 100:02d}"
    assert _has(text, cm / 100)


@given(cm=st.integers(min_value=10, max_value=999))
@settings(max_examples=200)
def test_unitless_integers_read_as_centimetres_or_millimetres(cm: int) -> None:
    assert _has(str(cm), cm / 100) and _has(str(cm), cm / 1000)


@given(
    mm=st.integers(min_value=1000, max_value=999_999),
    tenths=st.integers(min_value=0, max_value=9),
    style=st.sampled_from(["tr", "en", "space"]),
)
@settings(max_examples=300)
def test_thousands_separated_millimetres(mm: int, tenths: int, style: str) -> None:
    thousands, decimal = {"tr": (".", ","), "en": (",", "."), "space": (" ", ",")}[style]
    grouped = f"{mm:,}".replace(",", thousands)
    text = f"{grouped}{decimal}{tenths}"
    assert _has(text, (mm + tenths / 10) / 1000)
    assert _has(grouped, mm / 1000)


@given(v=st.integers(min_value=1, max_value=99_999), unit=st.sampled_from(["m", "cm", "mm"]))
@settings(max_examples=200)
def test_explicit_units(v: int, unit: str) -> None:
    factor = {"m": 0.01, "cm": 0.01, "mm": 0.001}[unit]
    text = f"{v / 100:.2f} m" if unit == "m" else f"{v} {unit}"
    d = parse_dimension(text)
    assert d is not None and d.unit == unit
    assert abs(d.metres[0] - v * factor) <= 1e-9 * max(1.0, v * factor)


@given(
    ft=st.integers(min_value=0, max_value=200),
    inch=st.integers(min_value=0, max_value=11),
    frac=st.sampled_from([None, (1, 2), (1, 4), (3, 8), (5, 16)]),
    dash=st.booleans(),
)
@settings(max_examples=300)
def test_feet_and_inches(ft: int, inch: int, frac: tuple[int, int] | None, dash: bool) -> None:
    if ft == 0 and inch == 0 and frac is None:
        inch = 1
    inches = Fraction(ft * 12 + inch) + (Fraction(*frac) if frac else 0)
    in_part = f"{inch}" + (f" {frac[0]}/{frac[1]}" if frac else "") + '"'
    text = f"{ft}'" + ("-" if dash else " ") + in_part if ft else in_part
    d = parse_dimension(text)
    assert d is not None and d.unit == "ft-in"
    assert abs(d.metres[0] - float(inches) * 0.0254) < 1e-9


def test_not_dimensions() -> None:
    for text in ["", "K1", "SALON", "m²", "1:50", "-", "abc'"]:
        assert parse_dimension(text) is None, text
