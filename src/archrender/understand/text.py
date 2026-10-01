"""Text normalisation and parsing for drawing sheets (Turkish-aware).

- Case mapping follows Turkish rules (I ↔ ı, İ ↔ i); :func:`fold` gives an ASCII key for matching
  headers and labels across languages ("GENİŞLİK" == "Genişlik" == "genislik").
- Numbers accept a decimal comma ("3,50"), thousands separators ("1.234,5" / "1,234.5") and plain
  integers.
- Dimensions return metres plus the unit reasoning, because "350" on a 1/50 sheet is centimetres
  while "3,50" is metres; ambiguous cases keep every candidate for S2 to reconcile with the scale.
- Scales: "Ölçek 1/50", "ÖLÇEK: 1:100", "M 1:50", "SCALE 1:20", "1/4\" = 1'-0\"" (→ 48).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})
_ASCII = str.maketrans(
    {"ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u", "â": "a", "î": "i", "û": "u"}
)


def tr_lower(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(_TR_LOWER).lower()


def tr_upper(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(_TR_UPPER).upper()


def fold(s: str) -> str:
    """Language-neutral matching key: Turkish lower-case, ASCII letters, single spaces, no punctuation."""
    s = tr_lower(s).replace("i̇", "i").translate(_ASCII)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9a-z]+", " ", s)
    return " ".join(s.split())


_NUM = re.compile(r"^[+-]?(?:\d{1,3}(?:[.,  ]\d{3})+|\d+)(?:[.,]\d+)?$")


def parse_number(s: str, *, locale: str | None = None) -> float | None:
    """'3,50' → 3.5 · '1.234,5' → 1234.5 · '1,234.5' → 1234.5 · '350' → 350.0 · '±0,00' → 0.0.

    A lone dot is a decimal point ('12.400' → 12.4) unless ``locale='tr'``, where the dot groups
    thousands ('12.400' → 12400). A lone comma is always a decimal comma on drawings ('3,50').
    """
    t = s.strip().replace("±", "").replace("−", "-")
    if not t or not _NUM.match(t):
        return None
    if locale == "tr" and "," not in t and re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+", t):
        return float(t.replace(".", ""))
    last_dot, last_comma = t.rfind("."), t.rfind(",")
    if last_dot >= 0 and last_comma >= 0:
        dec = "." if last_dot > last_comma else ","
        grp = "," if dec == "." else "."
        t = t.replace(grp, "").replace(dec, ".")
    elif last_comma >= 0:
        # a single comma followed by exactly three digits and a leading group ≤ 3 digits is
        # ambiguous ("1,250"); on Turkish drawings the comma is the decimal separator
        t = t.replace(",", ".") if t.count(",") == 1 else t.replace(",", "")
    elif t.count(".") > 1:
        t = t.replace(".", "")
    t = t.replace(" ", "").replace(" ", "")
    try:
        return float(t)
    except ValueError:
        return None


@dataclass(frozen=True)
class Dimension:
    text: str
    metres: tuple[float, ...]  # candidate values in metres, most likely first
    unit: str  # m | cm | mm | ft-in | ambiguous


_FT_IN = re.compile(
    r"^\s*(?:(?P<ft>\d+)\s*(?:'|′|ft)\s*-?\s*)?(?:(?P<in>\d+)?(?:\s+(?P<fn>\d+)/(?P<fd>\d+))?\s*(?:\"|″|in)?)?\s*$"
)


def parse_dimension(text: str, *, unit_hint: str | None = None) -> Dimension | None:
    """Parse a dimension string. ``unit_hint`` (m/cm/mm) comes from the sheet's convention."""
    t = text.strip()
    if not t:
        return None
    if any(c in t for c in "'′\"″"):
        m = _FT_IN.match(t)
        if not m or not (m.group("ft") or m.group("in") or m.group("fn")):
            return None
        inches = Fraction(int(m.group("ft") or 0) * 12) + Fraction(int(m.group("in") or 0))
        if m.group("fn"):
            inches += Fraction(int(m.group("fn")), int(m.group("fd")))
        return Dimension(t, (float(inches) * 0.0254,), "ft-in")
    unit = None
    for suffix, u in (("mm", "mm"), ("cm", "cm"), ("m", "m")):
        if t.lower().endswith(suffix):
            unit, t = u, t[: -len(suffix)].strip()
            break
    v = parse_number(t)
    if v is None or v < 0:
        return None
    factor = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    chosen = unit if unit is not None else unit_hint
    if chosen is not None:
        return Dimension(text.strip(), (v * factor[chosen],), chosen)
    if re.fullmatch(r"\d{1,3}[.,]\d{3}", t):
        # "1.234": 1.234 m, or 1234 mm (the same length), or 1234 cm
        return Dimension(text.strip(), (v, v * 10.0), "ambiguous")
    if _THOUSANDS.fullmatch(t):
        # "1.234,5" / "1,234.5" / "12 345": a large count of mm (or cm), never metres
        return Dimension(text.strip(), (v / 1000.0, v / 100.0), "ambiguous")
    has_decimals = bool(re.search(r"[.,]\d{1,2}$", t))
    if has_decimals and v < 100:
        return Dimension(text.strip(), (v,), "m")  # "3,50" → 3.5 m
    if not has_decimals and v >= 1000:
        return Dimension(text.strip(), (v / 1000.0, v / 100.0), "ambiguous")  # mm first
    if not has_decimals:
        return Dimension(text.strip(), (v / 100.0, v / 1000.0), "ambiguous")  # cm first ("350")
    return Dimension(text.strip(), (v,), "m")


_THOUSANDS = re.compile(r"\d{1,3}(?:([.,\s])\d{3})+(?:(?!\1)[.,]\d+)?")


_SCALE_RATIO = re.compile(r"(?<![\d/])1\s*[:/]\s*(\d{1,5})(?![\d/])")
_SCALE_IMPERIAL = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*(?:\"|″|in)\s*=\s*1\s*(?:'|′|ft)\s*-?\s*0?\s*(?:\"|″)?"
)
_SCALE_WORDS = ("olcek", "scale", "massstab", "echelle", "m")


def parse_scale(text: str) -> float | None:
    """Drawing scale denominator from a scale annotation (None if the text is not a scale)."""
    t = text.strip()
    m = _SCALE_IMPERIAL.search(t)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if num:
            return 12.0 * den / num  # 1/4" = 1'-0" → 48
    m = _SCALE_RATIO.search(t)
    if m:
        den = int(m.group(1))
        if 1 <= den <= 10000:
            return float(den)
    return None


def is_scale_label(text: str) -> bool:
    words = fold(text).split()
    return bool(words) and words[0] in _SCALE_WORDS


_TAG = re.compile(r"^(?P<prefix>[A-ZÇĞİÖŞÜ]{1,3})[-_ ]?(?P<num>\d{1,4}[A-Z]?)$")
TAG_KINDS = {
    "K": "door",  # Kapı
    "D": "door",
    "P": "window",  # Pencere
    "W": "window",
    "KP": "door",
    "PN": "window",
}


def normalise_tag(text: str) -> str | None:
    """'K-01' / 'k1' / 'P 12' → 'K1' / 'P12'. Returns None for text that is not a tag."""
    t = tr_upper(text.strip())
    m = _TAG.match(t)
    if not m:
        return None
    num = m.group("num").lstrip("0") or "0"
    return f"{m.group('prefix')}{num}"
