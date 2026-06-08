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
    assert depth_cfg["use_depth_attention"] is True
    assert depth_cfg["depth_attention_stride"] == depth_cfg["num_hidden_layers"] // 2
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

    expected_released = {
        "1.5b_denseformer_qknorm_gqa4x": ("denseformer", 1536),
        "1.5b_mhc_qknorm_gqa4x": ("mhc", 1536),
        "3b_denseformer_qknorm_gqa4x": ("denseformer", 2048),
        "3b_mhc_qknorm_gqa4x": ("mhc", 2048),
    }
    for dirname, (baseline_mode, hidden_size) in expected_released.items():
        data = json.loads((ROOT / "llama_config" / "released" / dirname / "config.json").read_text())
        assert data["baseline_mode"] == baseline_mode
        assert data["hidden_size"] == hidden_size
        assert data["num_hidden_layers"] == 48
        assert data["num_attention_heads"] // data["num_key_value_heads"] == 4
        assert data["use_qk_norm"] is True


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


def test_depth_attention_release_exposes_single_public_mode():
    forbidden_modes = [
        "depth_" + "softmax",
        "cross_layer_pattern",
        "cross_layer_mode",
        "depth_" + "softmax_" + "1head",
        "cross_attn_" + "lse",
        '"' + "g" + "ate" + '"',
    ]
    paths = [
        ROOT / "README.md",
        ROOT / "README_zh.md",
        ROOT / "src" / "llamafactory" / "hparams" / "model_args.py",
        ROOT / "src" / "llamafactory" / "model" / "loader.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for mode in forbidden_modes:
            assert mode not in text, f"{mode!r} found in {path.relative_to(ROOT)}"

    loader_text = (ROOT / "src" / "llamafactory" / "model" / "loader.py").read_text(encoding="utf-8")
    assert "default_" + "depth_stride" not in loader_text
