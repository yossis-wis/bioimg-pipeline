from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from drivers.run_slice0 import run_slice0
from src.simulate_phantom import PhantomParams, write_phantom_tiff


def test_end_to_end_sim_tiff(tmp_path, monkeypatch):
    # Create a minimal data bench
    data_root = tmp_path / "bench"
    for name in ["raw_staging", "runs", "cache", "models"]:
        (data_root / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BIOIMG_DATA_ROOT", str(data_root))

    # Write a config
    cfg = {
        "input_relpath": "raw_staging/phantom_slice0.tif",
        "output_runs_dir": "runs",
        "threshold": 200,
        "min_distance": 3,
        "smooth_sigma": 1.0,
        "channel": 1,
        "sim_seed": 1,
        "sim_height": 64,
        "sim_width": 64,
        "sim_num_spots": 10,
        "sim_sigma_px": 1.5,
        "sim_min_separation_px": 6.0,
        "sim_background_level": 100.0,
        "sim_noise_sigma": 5.0,
        "sim_intensity_min": 800.0,
        "sim_intensity_max": 1500.0,
    }
    cfg_path = tmp_path / "sim.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Generate TIFF (using the same code as the generator script)
    out_tif = write_phantom_tiff(
        data_root / cfg["input_relpath"],
        PhantomParams(
            height=cfg["sim_height"],
            width=cfg["sim_width"],
            num_spots=cfg["sim_num_spots"],
            sigma_px=cfg["sim_sigma_px"],
            min_separation_px=cfg["sim_min_separation_px"],
            background_level=cfg["sim_background_level"],
            noise_sigma=cfg["sim_noise_sigma"],
            intensity_min=cfg["sim_intensity_min"],
            intensity_max=cfg["sim_intensity_max"],
            seed=cfg["sim_seed"],
        ),
        overwrite=True,
    )
    assert out_tif.exists()

    # Run slice0
    out_dir = run_slice0(cfg_path)
    assert out_dir.exists()

    spots_path = out_dir / "spots.parquet"
    manifest_path = out_dir / "run_manifest.yaml"
    qc_path = out_dir / "qc_overlay.png"

    assert spots_path.exists()
    assert manifest_path.exists()
    assert qc_path.exists()

    spots = pd.read_parquet(spots_path)
    for col in ["frame", "y_px", "x_px", "intensity", "background", "snr"]:
        assert col in spots.columns
