"""Shared color palette for all dashboard charts.

The three primary colors are luminance-matched so accents have consistent
visual weight across chart types.
"""

from __future__ import annotations

POSITIVE_COLOR = "#E7A4A4"
NEGATIVE_COLOR = "#95C1DB"
PATIENT_COLOR = "#C3A8E5"

DEEMPHASIS_GREY = "#B8B8B8"

NEUTRAL_LINE_COLOR = "#9AA0A6"
NEUTRAL_TEXT_COLOR = "#5F6368"

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
