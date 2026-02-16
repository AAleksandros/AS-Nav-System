#!/usr/bin/env python3
"""Run all 5 new scenarios and generate output videos."""

import subprocess
import sys
from pathlib import Path

scenarios = ["slalom", "narrow_gap", "u_turn", "crossing", "dense"]
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

print("Running all 5 scenarios...\n")

for scenario in scenarios:
    output_file = output_dir / f"{scenario}.mp4"
    print(f"[{scenario}] Running...")

    result = subprocess.run([
        sys.executable, "-m", "src.main",
        "--scenario", scenario,
        "-o", str(output_file),
    ])

    if result.returncode != 0:
        print(f"[{scenario}] FAILED")
        sys.exit(1)

    print(f"[{scenario}] ✓ {output_file}\n")

print("All scenarios completed!")
