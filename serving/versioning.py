"""
Active version selection and A/B traffic splitting.

Reads from config/serving.yaml. When A/B testing is disabled the active_version
field is returned directly. When enabled, a weighted random draw picks from the
configured versions list.
"""

from __future__ import annotations

import random
from typing import Any, Dict


def pick_version(cfg: Dict[str, Any]) -> str:
    """
    Return the collection version to use for this request.

    serving.yaml shape:
        vector_store:
          active_version: "v1"
          collection_prefix: "docs"
          ab_test:
            enabled: false
            versions: []          # [{version: "v1", weight: 0.7}, ...]

    When ab_test.enabled is true and versions is non-empty, a weighted
    random choice is made among the listed versions.
    Falls back to active_version in all other cases.
    """
    vs = cfg.get("vector_store", {})
    active = vs.get("active_version", "v1")
    ab = vs.get("ab_test", {})

    if not ab.get("enabled"):
        return active

    versions = ab.get("versions") or []
    if not versions:
        return active

    names = [v["version"] for v in versions]
    weights = [float(v.get("weight", 1.0)) for v in versions]
    return random.choices(names, weights=weights, k=1)[0]
