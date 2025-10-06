#!/bin/bash
# Quick start script for FedGAN prototype

echo "==========================================="
echo "FedGAN Prototype - Quick Start"
echo "==========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run FedGAN with default parameters
echo ""
echo "==========================================="
echo "Starting FedGAN training..."
echo "Configuration:"
echo "  - Agents: 3"
echo "  - Epochs: 50"
echo "  - Sync Interval K: 5"
echo "==========================================="
echo ""

python main.py \
  --num_agents 3 \
  --epochs 50 \
  --sync_interval 5 \
  --output_dir tmp_outputs

echo ""
echo "==========================================="
echo "Training complete!"
echo "Results saved to: tmp_outputs/"
echo "==========================================="
