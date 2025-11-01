import json
import subprocess
import sys
from pathlib import Path

import yaml


def run_cli(args):
    result = subprocess.run([sys.executable, "-m", "neutryx.cli", *args], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def test_price_vanilla_cli(tmp_path: Path):
    config = {
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.01,
        "dividend": 0.0,
        "volatility": 0.2,
        "mc": {"steps": 4, "paths": 128},
        "seed": 0,
    }
    cfg_path = tmp_path / "vanilla.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    output = run_cli(["price", "vanilla", "--config", str(cfg_path)])
    assert "price" in output
    assert output["price"] > 0.0


def test_cva_cli(tmp_path: Path):
    config = {
        "epe": [1.0, 0.8, 0.6],
        "discount": [0.99, 0.97, 0.95],
        "default_probability": [0.0, 0.02, 0.05],
        "lgd": 0.6,
    }
    cfg_path = tmp_path / "cva.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    output = run_cli(["xva", "cva", "--config", str(cfg_path)])
    assert "cva" in output
    assert output["cva"] >= 0.0
