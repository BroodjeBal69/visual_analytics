"""Shared color palette for all dashboard charts.

The three primary colors are luminance-matched so accents have consistent
visual weight across chart types.
"""

from __future__ import annotations

# Initialize color palettes, normal and color blind 
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
SEQUENTIAL_COLOR_SCALES = {}
SEQUENTIAL_COLOR_SCALE = []


def _normalize_mode(mode: str | None) -> str:
    return mode if mode in PALETTES else DEFAULT_PALETTE_MODE


def get_palette_tokens(mode: str | None = None) -> dict[str, str]:
    return PALETTES[_normalize_mode(mode)].copy()


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    raw = hex_color.strip().lstrip("#")
    if len(raw) != 6:
        return (0, 0, 0)
    return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def _mix_with_white(hex_color: str, t: float) -> str:
    """Blend a color toward white. t=0 keeps original color, t=1 gives white."""
    r, g, b = _hex_to_rgb(hex_color)
    t_clamped = max(0.0, min(1.0, float(t)))
    mixed = (
        int(round(r + (255 - r) * t_clamped)),
        int(round(g + (255 - g) * t_clamped)),
        int(round(b + (255 - b) * t_clamped)),
    )
    return _rgb_to_hex(mixed)


def _build_sequential_scale(base_color: str) -> list[list[float | str]]:
    # Light-to-strong single-hue progression for sequential encodings.
    return [
        [0.00, _mix_with_white(base_color, 0.92)],
        [0.20, _mix_with_white(base_color, 0.75)],
        [0.40, _mix_with_white(base_color, 0.55)],
        [0.60, _mix_with_white(base_color, 0.35)],
        [0.80, _mix_with_white(base_color, 0.18)],
        [1.00, base_color],
    ]

# ==== Application logic to apply palette tokens and generate color scales =====
def apply_palette_mode(mode: str | None = None) -> str:
    global ACTIVE_PALETTE_MODE
    global POSITIVE_COLOR, NEGATIVE_COLOR, PATIENT_COLOR, DEEMPHASIS_GREY
    global DISEASE_COLOR_MAP, PROFILE_DIRECTION_COLOR_MAP, CLUSTER_COLOR_MAP, DIVERGING_COLOR_SCALE
    global SEQUENTIAL_COLOR_SCALES, SEQUENTIAL_COLOR_SCALE

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

    SEQUENTIAL_COLOR_SCALES = {
        "positive": _build_sequential_scale(POSITIVE_COLOR),
        "negative": _build_sequential_scale(NEGATIVE_COLOR),
        "patient": _build_sequential_scale(PATIENT_COLOR),
    }
    # Default sequential scale uses the positive hue.
    SEQUENTIAL_COLOR_SCALE = SEQUENTIAL_COLOR_SCALES["positive"]

    ACTIVE_PALETTE_MODE = resolved_mode
    return resolved_mode


def risk_color(probability: float) -> str:
    probability = max(0.0, min(1.0, float(probability)))
    red = int(255 * probability)
    green = int(160 * (1 - probability))
    return f"rgb({red}, {green}, 60)"


apply_palette_mode(DEFAULT_PALETTE_MODE)
