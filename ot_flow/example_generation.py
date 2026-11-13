from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

if __package__ in (None, ""):
    # Allow running this file directly as a script: add the parent directory
    # to sys.path so the package containing FMTabularGenerator can be
    # imported by name. Adjust "ot_flow" below to match your package's
    # directory name if different.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from ot_flow import FMTabularGenerator
else:
    from . import FMTabularGenerator


def build_fake_dataset() -> pd.DataFrame:
    """Small synthetic tabular dataset with two categorical columns."""

    return pd.DataFrame(
        {
            "city": pd.Categorical(
                [
                    "rome",
                    "milan",
                    "turin",
                    "rome",
                    "milan",
                    "rome",
                    "turin",
                    "milan",
                    "rome",
                    "turin",
                    "rome",
                    "milan",
                ]
            ),
            "device": pd.Categorical(
                [
                    "mobile",
                    "desktop",
                    "tablet",
                    "mobile",
                    "desktop",
                    "mobile",
                    "tablet",
                    "desktop",
                    "mobile",
                    "tablet",
                    "desktop",
                    "mobile",
                ]
            ),
            "age": [23, 35, 41, 29, 52, 38, 47, 31, 26, 44, 33, 39],
            "income": [32000, 54000, 61000, 38000, 72000, 59000, 68000, 45000, 34000, 64000, 50000, 56000],
        }
    )


def main() -> None:
    df = build_fake_dataset()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = FMTabularGenerator(
        device=device,
        hidden_dim=128,
        time_dim=32,
        depth=3,
        dropout=0.0,
    )

    generator.fit(
        df,
        num_epochs=20,
        batch_size=4,
        lr=1e-3,
        epsilon=0.1,
        categorical_cols=["city", "device"],
        seed=42,
    )

    generated = generator.generate(100, n_steps=50, seed=123)

    print("Original dataset:\n", df)
    print("\nGenerated samples:\n", generated)
    print("\nLoss history (last 5):", generator.loss_history_[-5:])
    print(f"Real data stats:\n{df.describe(include='all')}")
    print(f"\nGenerated data stats:\n{generated.describe(include='all')}")

    # Persistence round-trip
    save_path = Path("Results/generated/fm_tabular_generator.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save(save_path)
    loaded = FMTabularGenerator.load(save_path, device=device)
    reloaded_samples = loaded.generate(5, n_steps=50, seed=123)
    print("\nReloaded model samples:\n", reloaded_samples)


if __name__ == "__main__":
    main()
