#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/workspaces/macro-regime-monte-carlo"

echo "Running data pipeline..."
python "$PROJECT_ROOT/src/data_pipeline.py"

echo "Running HMM regime engine..."
python "$PROJECT_ROOT/src/hmm_regime_engine.py"

echo "Generating regime plots..."
python "$PROJECT_ROOT/src/plot_regimes.py"

echo "Done."