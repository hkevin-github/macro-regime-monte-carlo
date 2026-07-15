#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="/workspaces/macro-regime-monte-carlo"

echo "Running data pipeline..."
python "$PROJECT_ROOT/src/data_pipeline.py"

echo "Optimizing regime count..."
python "$PROJECT_ROOT/src/model_selection.py"

echo "Running HMM regime engine..."
python "$PROJECT_ROOT/src/hmm_regime_engine.py"

echo "Done."