from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_json_files_are_valid():
    for path in ROOT.rglob("*.json"):
        if any(part in {".git", ".pytest_cache", ".ruff_cache"} for part in path.parts):
            continue
        with path.open(encoding="utf-8") as f:
            json.load(f)


def test_public_configs_use_reference_fields():
    depth_cfg = json.loads((ROOT / "llama_config" / "depth_attention_410m" / "config.json").read_text())
    assert depth_cfg["recurrent_model"] is True
    assert depth_cfg["cross_layer_pattern"] == "depth_softmax"
    assert depth_cfg["cross_layer_mode"] == "depth_softmax"
    assert depth_cfg["depth_softmax_stride"] == 4
    assert depth_cfg["num_attention_heads"] // depth_cfg["num_key_value_heads"] == 4

    expected_baselines = {
        "baseline_attnres_410m": "attnres",
        "baseline_denseformer_410m": "denseformer",
        "baseline_mhc_410m": "mhc",
    }
    for dirname, baseline_mode in expected_baselines.items():
        data = json.loads((ROOT / "llama_config" / dirname / "config.json").read_text())
        assert data["baseline_mode"] == baseline_mode
        assert data["recurrent_model"] is True
        assert data["num_attention_heads"] // data["num_key_value_heads"] == 4


def test_public_modeling_files_are_one_per_method():
    modeling_dir = ROOT / "src" / "llamafactory" / "model" / "modeling"
    expected = {
        "modeling_llama_depth_attention.py",
        "modeling_llama_attnres.py",
        "modeling_llama_denseformer.py",
        "modeling_llama_mhc.py",
    }
    actual = {path.name for path in modeling_dir.glob("modeling_llama_*.py")}
    assert expected.issubset(actual)
    assert "modeling_llama_baselines.py" not in actual


def test_no_private_paths_or_secrets():
    forbidden_literals = [
        "/" + "mnt/",
        "/" + "data0/",
    ]
    hardcoded_secret = re.compile(
        r"""(?i)\b(api[_-]?key|password|secret|token)\b\s*[:=]\s*["'][^"'<>{}$\s][^"']{7,}["']"""
    )
    allowed_suffixes = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".pdf"}
    for path in ROOT.rglob("*"):
        if path.is_dir():
            continue
        if any(part in {".git", ".pytest_cache", ".ruff_cache", "outputs", "saves"} for part in path.parts):
            continue
        if path.suffix.lower() in allowed_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for needle in forbidden_literals:
            assert needle not in text, f"{needle!r} found in {path.relative_to(ROOT)}"
        assert hardcoded_secret.search(text) is None, f"hard-coded credential-like value found in {path.relative_to(ROOT)}"


def test_patch_method_strings_are_depth_attention_specific():
    patch_file = (ROOT / "src" / "llamafactory" / "model" / "llama_patch.py").read_text()
    for method in ["depth_attention", "attnres", "denseformer", "mhc"]:
        assert method in patch_file
    assert "ponderlm" not in patch_file.lower()
