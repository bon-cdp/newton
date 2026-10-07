#!/bin/bash
# Start the ellipsoid tuner detached (see optimize.py); arguments pass through, e.g.
#   setsid nohup bash calibration/soybean_bucket/run_optimize.sh --calls 24 > runs/logs/optimize.log 2>&1 &
#   setsid nohup bash calibration/soybean_bucket/run_optimize.sh --tag opt2 --fix-shape 1.165 1.18 \
#       --calls 16 > runs/logs/optimize2.log 2>&1 &
cd "$(dirname "$0")/../.."
exec /home/s/Documents/newton-dem/runs/perf/vtkcheck/bin/python calibration/soybean_bucket/optimize.py "$@"
