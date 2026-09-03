"""Label space for Vaani noise-event detection.

The dataset card lists 7 top-level categories, but the *actual* strings stored in
the parquet differ from the card (`animal_sound` vs `animal`, `baby_child_noise`
vs `baby_child`).  Everything here is therefore alias-driven: unknown strings are
normalised, then resolved through ALIASES, and anything still unresolved is
reported by `prepare.py` rather than silently dropped.
"""
from __future__ import annotations

import re
from typing import Iterable, List

# Canonical 7 top-level categories (names as they appear in the released parquet).
BASE_CLASSES: List[str] = [
    "animal_sound",
    "vehicle_traffic",
    "baby_child_noise",
    "singing_music",
    "phone_signal_alarm",
    "appliance_machine",
    "human_non_speech",
]

# Maps every spelling we have seen (dataset card, parquet, plausible variants)
# onto a canonical class.
ALIASES = {
    "animal": "animal_sound",
    "animal_sound": "animal_sound",
    "animal_sounds": "animal_sound",
    "animals": "animal_sound",
    "vehicle": "vehicle_traffic",
    "vehicle_traffic": "vehicle_traffic",
    "traffic": "vehicle_traffic",
    "vehicle_and_traffic": "vehicle_traffic",
    "baby_child": "baby_child_noise",
    "baby_child_noise": "baby_child_noise",
    "child": "baby_child_noise",
    "baby": "baby_child_noise",
    "singing_music": "singing_music",
    "music": "singing_music",
    "singing": "singing_music",
    "music_singing": "singing_music",
    "phone_signal_alarm": "phone_signal_alarm",
    "phone": "phone_signal_alarm",
    "alarm": "phone_signal_alarm",
    "signal": "phone_signal_alarm",
    "phone_alarm": "phone_signal_alarm",
    "appliance_machine": "appliance_machine",
    "appliance": "appliance_machine",
    "machine": "appliance_machine",
    "machinery": "appliance_machine",
    "human_non_speech": "human_non_speech",
    "human_nonspeech": "human_non_speech",
    "non_speech": "human_non_speech",
    "human": "human_non_speech",
}

# --- vehicle_traffic subtype split -------------------------------------------
# Rationale: frequency-dynamic convolution helps impulsive/harmonic events
# (horns) and can hurt stationary broadband ones (engines).  `vehicle_traffic`
# straddles both regimes, so we optionally train them as separate classes.
_HORN_KW = ("horn", "honk", "hooter", "hoot", "beep of vehicle")
_SIREN_KW = ("siren", "ambulance", "police")
VEHICLE_SPLIT = ["vehicle_horn", "vehicle_engine"]


def normalise_token(s: str) -> str:
    """Lowercase, strip <>/[]/() wrappers and collapse separators."""
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"^[<\[\(\{]+|[>\]\)\}]+$", "", s)
    s = re.sub(r"[\s\-/]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def canon_category(cat: str) -> str | None:
    """Resolve a raw category string to a canonical class, or None if unknown."""
    t = normalise_token(cat)
    if t in ALIASES:
        return ALIASES[t]
    # Fall back to substring matching so future spellings still land somewhere.
    for key, val in ALIASES.items():
        if key and key in t:
            return val
    return None


def vehicle_subtype(tag: str) -> str:
    """Split a vehicle_traffic event into horn-like vs engine-like by its tag."""
    t = normalise_token(tag).replace("_", " ")
    if any(k in t for k in _HORN_KW) or any(k in t for k in _SIREN_KW):
        return "vehicle_horn"
    return "vehicle_engine"


def build_class_list(expand_vehicle: bool = True) -> List[str]:
    if not expand_vehicle:
        return list(BASE_CLASSES)
    out: List[str] = []
    for c in BASE_CLASSES:
        if c == "vehicle_traffic":
            out.extend(VEHICLE_SPLIT)
        else:
            out.append(c)
    return out


def resolve_event_class(category: str, tag: str, expand_vehicle: bool) -> str | None:
    """Map one (category, tag) annotation onto a training class name."""
    base = canon_category(category)
    if base is None:
        return None
    if expand_vehicle and base == "vehicle_traffic":
        return vehicle_subtype(tag)
    return base


class LabelEncoder:
    def __init__(self, expand_vehicle: bool = True):
        self.expand_vehicle = expand_vehicle
        self.classes = build_class_list(expand_vehicle)
        self.idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.classes)

    def encode(self, category: str, tag: str = "") -> int | None:
        c = resolve_event_class(category, tag, self.expand_vehicle)
        return self.idx.get(c) if c is not None else None

    def encode_clip_categories(self, cats: Iterable[str]) -> List[int]:
        """Clip-level (bronze) tags carry no `tag` field, so a bare
        `vehicle_traffic` activates *both* vehicle subclasses (it is genuinely
        ambiguous which one is present)."""
        out: List[int] = []
        for c in cats or []:
            base = canon_category(c)
            if base is None:
                continue
            if self.expand_vehicle and base == "vehicle_traffic":
                out.extend(self.idx[v] for v in VEHICLE_SPLIT)
            elif base in self.idx:
                out.append(self.idx[base])
        return sorted(set(out))
