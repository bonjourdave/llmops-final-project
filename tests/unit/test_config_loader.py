import os
import textwrap
from pathlib import Path

import pytest

from shared.config_loader import load_config


@pytest.fixture
def tmp_yaml(tmp_path):
    """Helper: write a YAML file to a temp directory and return its path."""
    def _write(content: str) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(textwrap.dedent(content))
        return p
    return _write


def test_load_config_returns_dict(tmp_yaml):
    path = tmp_yaml("""
        pipeline:
          embedding_model: text-embedding-3-small
          chunk_size: 512
    """)
    cfg = load_config(path)
    assert isinstance(cfg, dict)
    assert cfg["pipeline"]["embedding_model"] == "text-embedding-3-small"
    assert cfg["pipeline"]["chunk_size"] == 512


def test_env_var_substitution(tmp_yaml, monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "supersecret")
    path = tmp_yaml("""
        store:
          token: "${TEST_SECRET}"
    """)
    cfg = load_config(path)
    assert cfg["store"]["token"] == "supersecret"


def test_missing_env_var_returns_empty_string(tmp_yaml, monkeypatch):
    monkeypatch.delenv("UNSET_VAR", raising=False)
    path = tmp_yaml("""
        store:
          uri: "${UNSET_VAR}"
    """)
    cfg = load_config(path)
    assert cfg["store"]["uri"] == ""


def test_nested_substitution(tmp_yaml, monkeypatch):
    monkeypatch.setenv("HOST", "cloud.langfuse.com")
    path = tmp_yaml("""
        langfuse:
          host: "https://${HOST}"
    """)
    cfg = load_config(path)
    assert cfg["langfuse"]["host"] == "https://cloud.langfuse.com"


def test_real_pipeline_yaml_loads():
    cfg = load_config("config/pipeline.yaml")
    assert "pipeline" in cfg
    assert "vector_store" in cfg
    assert cfg["pipeline"]["chunk_size"] == 512
