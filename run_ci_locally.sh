#!/usr/bin/env bash

# ML-Sandbox Local CI runner script
# Simulates the GitHub Actions checks locally

set -eo pipefail

# ANSI color codes for premium formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BLUE='\033[0;34m'
BOLD='\033[1m'

log_info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} $1"
}

log_info "Starting Local CI pipeline check..."
echo "=================================================="

# Step 1: Run Ruff Linter
log_info "Step 1: Running Ruff Linter..."
if python3 -m ruff check .; then
    log_success "Ruff linter passed!"
else
    log_error "Ruff linter failed!"
    exit 1
fi

echo "--------------------------------------------------"

# Step 2: Run Mypy Type Checker
log_info "Step 2: Running mypy Type Checker..."
if python3 -m mypy --ignore-missing-imports projects/; then
    log_success "Mypy type checking passed!"
else
    log_error "Mypy type checking failed!"
    exit 1
fi

echo "--------------------------------------------------"

# Step 3: Run sub-project tests
log_info "Step 3: Running Test Suites..."

# 3.1 Cultural Enrichment
log_info "Running tests for: cultural-enrichment-pipeline"
if PYTHONPATH=projects/cultural-enrichment-pipeline python3 -m pytest projects/cultural-enrichment-pipeline/tests/ -v; then
    log_success "cultural-enrichment-pipeline tests passed!"
else
    log_error "cultural-enrichment-pipeline tests failed!"
    exit 1
fi

echo "--------------------------------------------------"

# 3.2 Ad Optimization
log_info "Running tests for: ad-optimization-engine"
if PYTHONPATH=projects/ad-optimization-engine python3 -m pytest projects/ad-optimization-engine/tests/ -v; then
    log_success "ad-optimization-engine tests passed!"
else
    log_error "ad-optimization-engine tests failed!"
    exit 1
fi

echo "--------------------------------------------------"

# 3.3 Causal Uplift
log_info "Running tests for: causal-uplift-experimenter"
if PYTHONPATH=projects/causal-uplift-experimenter python3 -m pytest projects/causal-uplift-experimenter/tests/ -v; then
    log_success "causal-uplift-experimenter tests passed!"
else
    log_error "causal-uplift-experimenter tests failed!"
    exit 1
fi

echo "--------------------------------------------------"

# 3.4 MLOps Serving Infrastructure
log_info "Running tests for: MLOps-serving-infrastructure"
if PYTHONPATH=projects/MLOps-serving-infrastructure python3 -m pytest projects/MLOps-serving-infrastructure/tests/ -v; then
    log_success "MLOps-serving-infrastructure tests passed!"
else
    log_error "MLOps-serving-infrastructure tests failed!"
    exit 1
fi

echo "=================================================="
log_success "Local CI pipeline completed successfully! All checks passed!"
