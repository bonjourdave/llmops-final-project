from serving.versioning import pick_version


def _cfg(active: str = "v1", ab_enabled: bool = False, versions: list = None) -> dict:
    return {
        "vector_store": {
            "active_version": active,
            "collection_prefix": "docs",
            "ab_test": {
                "enabled": ab_enabled,
                "versions": versions or [],
            },
        }
    }


def test_returns_active_version_when_ab_disabled():
    assert pick_version(_cfg(active="v1")) == "v1"


def test_returns_active_version_when_ab_enabled_but_no_versions():
    cfg = _cfg(active="v2", ab_enabled=True, versions=[])
    assert pick_version(cfg) == "v2"


def test_ab_single_version_always_returns_it():
    cfg = _cfg(ab_enabled=True, versions=[{"version": "v1", "weight": 1.0}])
    assert pick_version(cfg) == "v1"


def test_ab_two_versions_returns_one_of_them():
    cfg = _cfg(
        ab_enabled=True,
        versions=[{"version": "v1", "weight": 0.5}, {"version": "v2", "weight": 0.5}],
    )
    for _ in range(20):
        assert pick_version(cfg) in ("v1", "v2")


def test_ab_deterministic_with_zero_weight():
    # v1 weight=0 → should never be picked
    cfg = _cfg(
        ab_enabled=True,
        versions=[{"version": "v1", "weight": 0.0}, {"version": "v2", "weight": 1.0}],
    )
    for _ in range(20):
        assert pick_version(cfg) == "v2"


def test_missing_ab_test_key_falls_back_to_active():
    cfg = {"vector_store": {"active_version": "v1", "collection_prefix": "docs"}}
    assert pick_version(cfg) == "v1"


def test_missing_vector_store_key_uses_defaults():
    assert pick_version({}) == "v1"
