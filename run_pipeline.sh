#!/bin/bash
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_section() {
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}  $1${NC}"
    echo "================================================================================"
    echo ""
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[*]${NC} $1"
}

echo "================================================================================"
echo "  REGIME-AWARE MONTE CARLO PORTFOLIO SIMULATION"
echo "  Full Pipeline Execution"
echo "================================================================================"
echo ""
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found."
    exit 1
fi

print_status "Python3 found: $(python3 --version)"

mkdir -p data/processed data/models notebook output config
print_status "Directories ready"

if [ -z "$FRED_API_KEY" ] && [ ! -f .env ]; then
    print_error "FRED API key not found in environment or .env file"
    exit 1
fi

print_section "STEP 1: DATA INGESTION & ALIGNMENT"
python3 src/data_pipeline.py
print_status "Data pipeline completed successfully"

print_section "STEP 2: HMM MODEL SELECTION (BIC/AIC)"
python3 src/model_selection.py
print_status "Model selection completed successfully"

print_section "STEP 3: HMM REGIME TRAINING"
python3 src/hmm_regime_engine.py
print_status "HMM training completed successfully"

print_section "STEP 4: PORTFOLIO ANALYSIS + MONTE CARLO"

PORTFOLIO_FILE="config/my_portfolio.csv"
CONFIG_FILE="config/info.json"

if [ ! -f "$PORTFOLIO_FILE" ]; then
    print_error "Portfolio file not found: $PORTFOLIO_FILE"
    echo "Create a CSV with columns: ticker,shares"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    print_info "Config file not found. Will create default: $CONFIG_FILE"
fi

# Run simulation using config file
python3 src/analyze_and_simulate.py "$PORTFOLIO_FILE" --config "$CONFIG_FILE"
print_status "Portfolio analysis + Monte Carlo completed successfully"

print_section "PIPELINE COMPLETE"
echo ""
print_status "All steps completed successfully!"
echo ""
echo "Output files:"
echo "  • Data: data/processed/aligned_macro_dataset.csv"
echo "  • Regime labels: data/processed/regime_labeled_dataset.csv"
echo "  • Model selection: data/models/model_selection_results.csv"
echo "  • HMM model: data/models/hmm_*state.pkl"
echo "  • Regime stats: data/models/regime_market_assumptions.json"
echo "  • Visualizations: notebook/macro_regime_map.png"
echo "  • Portfolio analysis: output/portfolio_analysis.json"
echo "  • Simulation results: output/simulation_results.json"
echo "  • Visualization: output/simulation_results.png"
echo ""
echo "Configuration used: $CONFIG_FILE"
echo ""
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""