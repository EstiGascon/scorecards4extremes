#!/bin/bash
#
# Setup script for scorecards4extremes on ECMWF systems
# Run this before executing the tool:
#   source setup_ecmwf.sh
#

echo "=========================================="
echo "Setting up scorecards4extremes environment"
echo "=========================================="

# 1. Set up TMPDIR
export TMPDIR=/ec/res4/scratch/$USER/tmp
mkdir -p $TMPDIR
echo "✓ TMPDIR set to: $TMPDIR"

# 2. Try to load Metview module
echo ""
echo "Loading Metview module..."

# Try different possible module names
if module load metview 2>/dev/null; then
    echo "✓ Loaded: metview"
elif module load metview-bundle 2>/dev/null; then
    echo "✓ Loaded: metview-bundle"
elif module load metview-python 2>/dev/null; then
    echo "✓ Loaded: metview-python"
else
    echo "⚠ Warning: Could not load metview module"
    echo "  Available metview modules:"
    module avail metview 2>&1 | grep metview
    echo ""
    echo "  Try manually: module load <metview_module_name>"
fi

# 3. Check if metview command is available
echo ""
if command -v metview &> /dev/null; then
    echo "✓ metview command found: $(which metview)"
    metview -v 2>/dev/null || echo "  (version info not available)"
else
    echo "⚠ Warning: 'metview' command not found in PATH"
    echo "  This will cause import errors"
fi

# 4. Set Metview timeout (in case it's slow)
export METVIEW_PYTHON_START_TIMEOUT=30
echo "✓ METVIEW_PYTHON_START_TIMEOUT set to 30 seconds"

# 5. Check Python
echo ""
echo "Python version: $(python3 --version)"
echo "Python location: $(which python3)"

# 6. Check required packages
echo ""
echo "Checking Python packages..."
python3 -c "import metview; print('✓ metview-python installed')" 2>/dev/null || echo "✗ metview-python NOT installed"
python3 -c "import pandas; print('✓ pandas installed')" 2>/dev/null || echo "✗ pandas NOT installed"
python3 -c "import numpy; print('✓ numpy installed')" 2>/dev/null || echo "✗ numpy NOT installed"
python3 -c "import matplotlib; print('✓ matplotlib installed')" 2>/dev/null || echo "✗ matplotlib NOT installed"
python3 -c "import yaml; print('✓ pyyaml installed')" 2>/dev/null || echo "✗ pyyaml NOT installed"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Now you can run: python3 run.py"
echo ""
