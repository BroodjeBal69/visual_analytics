"""Shared color palette for all dashboard charts.

The three primary colors are luminance-matched so accents have consistent
visual weight across chart types.
"""

from __future__ import annotations

PALETTES = {
    "pastel": {
        "positive": "#E7A4A4",
        "negative": "#95C1DB",
        "patient": "#C3A8E5",
        "grey": "#B8B8B8",
    },
    "colorblind": {
        "positive": "#66C2A5",
        "negative": "#8DA0CB",
        "patient": "#FC8D62",
        "grey": "#B8B8B8",
    },
}

DEFAULT_PALETTE_MODE = "colorblind"
ACTIVE_PALETTE_MODE = DEFAULT_PALETTE_MODE

POSITIVE_COLOR = ""
NEGATIVE_COLOR = ""
PATIENT_COLOR = ""
DEEMPHASIS_GREY = ""
NEUTRAL_LINE_COLOR = "#9AA0A6"
NEUTRAL_TEXT_COLOR = "#5F6368"

DISEASE_COLOR_MAP = {}
PROFILE_DIRECTION_COLOR_MAP = {}
CLUSTER_COLOR_MAP = {}
DIVERGING_COLOR_SCALE = []


def _normalize_mode(mode: str | None) -> str:
    return mode if mode in PALETTES else DEFAULT_PALETTE_MODE


def get_palette_tokens(mode: str | None = None) -> dict[str, str]:
    return PALETTES[_normalize_mode(mode)].copy()


def apply_palette_mode(mode: str | None = None) -> str:
    global ACTIVE_PALETTE_MODE
    global POSITIVE_COLOR, NEGATIVE_COLOR, PATIENT_COLOR, DEEMPHASIS_GREY
    global DISEASE_COLOR_MAP, PROFILE_DIRECTION_COLOR_MAP, CLUSTER_COLOR_MAP, DIVERGING_COLOR_SCALE

    resolved_mode = _normalize_mode(mode)
    tokens = get_palette_tokens(resolved_mode)

    POSITIVE_COLOR = tokens["positive"]
    NEGATIVE_COLOR = tokens["negative"]
    PATIENT_COLOR = tokens["patient"]
    DEEMPHASIS_GREY = tokens["grey"]

    DISEASE_COLOR_MAP = {
        "No disease": NEGATIVE_COLOR,
        "Disease": POSITIVE_COLOR,
        "No Disease": NEGATIVE_COLOR,
        "Heart disease": POSITIVE_COLOR,
        "No heart disease": NEGATIVE_COLOR,
    }

    PROFILE_DIRECTION_COLOR_MAP = {
        "Higher than population": POSITIVE_COLOR,
        "Lower than population": NEGATIVE_COLOR,
    }

    CLUSTER_COLOR_MAP = {
        "Cluster 1": NEGATIVE_COLOR,
        "Cluster 2": POSITIVE_COLOR,
        "Cluster 3": PATIENT_COLOR,
    }

    DIVERGING_COLOR_SCALE = [
        [0.0, NEGATIVE_COLOR],
        [0.5, PATIENT_COLOR],
        [1.0, POSITIVE_COLOR],
    ]

    ACTIVE_PALETTE_MODE = resolved_mode
    return resolved_mode


apply_palette_mode(DEFAULT_PALETTE_MODE)
