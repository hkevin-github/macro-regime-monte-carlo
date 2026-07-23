#!/bin/bash
# run_pipeline.sh
# Master execution script for Regime-Aware Monte Carlo Portfolio Simulation
#
# This script orchestrates the complete pipeline:
# 1. Data ingestion and alignment (with today's data)
# 2. Model selection (optimal number of HMM states)
# 3. HMM regime training
# 4. Regime visualization
# 5. Monte Carlo portfolio simulation with 80/20 portfolio
#
# Usage:
#   chmod +x run_pipeline.sh
#   ./run_pipeline.sh

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo "================================================================================"
    echo -e "${BLUE}  $1${NC}"
    echo "================================================================================"
    echo ""
}

# Function to print status messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[*]${NC} $1"
}

# Start pipeline
echo "================================================================================"
echo "  REGIME-AWARE MONTE CARLO PORTFOLIO SIMULATION"
echo "  Full Pipeline Execution"
echo "================================================================================"
echo ""
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

print_status "Python3 found: $(python3 --version)"

# Check required directories exist
print_info "Creating required directories..."
mkdir -p data/processed
mkdir -p data/models
mkdir -p notebook
mkdir -p output
print_status "Directories ready"

# Check for FRED API key
if [ -z "$FRED_API_KEY" ] && [ ! -f .env ]; then
    print_error "FRED API key not found in environment or .env file"
    echo "Please set FRED_API_KEY environment variable or create .env file"
    exit 1
fi

# Step 1: Data Pipeline
print_section "STEP 1: DATA INGESTION & ALIGNMENT"
print_info "Fetching macro data from 1953 to $(date '+%Y-%m-%d')"
print_info "Sources: FRED (macro), Yahoo Finance (equity), Fama-French (factors)"

if python3 src/data_pipeline.py; then
    print_status "Data pipeline completed successfully"
else
    print_error "Data pipeline failed"
    exit 1
fi

# Step 2: Model Selection
print_section "STEP 2: HMM MODEL SELECTION (BIC/AIC)"
print_info "Testing 2-6 state models to find optimal complexity"

if python3 src/model_selection.py; then
    print_status "Model selection completed successfully"
else
    print_error "Model selection failed"
    exit 1
fi

# Step 3: HMM Training
print_section "STEP 3: HMM REGIME TRAINING"
print_info "Training Hidden Markov Model with optimal number of states"

if python3 src/hmm_regime_engine.py; then
    print_status "HMM training completed successfully"
else
    print_error "HMM training failed"
    exit 1
fi

# Step 4: Regime Visualization
print_section "STEP 4: REGIME VISUALIZATION"
print_info "Generating historical regime map"

if python3 src/plot_regimes.py; then
    print_status "Regime visualization completed successfully"
else
    print_error "Regime visualization failed"
    exit 1
fi

# Step 5: Monte Carlo Simulation
print_section "STEP 5: MONTE CARLO PORTFOLIO SIMULATION"
print_info "Running simulation with 80/20 equity/bond portfolio"
print_info "Configuration: config/80_20_portfolio.json"

if python3 src/run_portfolio_mc.py config/80_20_portfolio.json; then
    print_status "Monte Carlo simulation completed successfully"
else
    print_error "Monte Carlo simulation failed"
    exit 1
fi

# Summary
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
echo "  • Simulation results: output/portfolio_80equ_20bon.png"
echo ""
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""