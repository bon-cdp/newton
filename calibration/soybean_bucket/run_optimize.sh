#!/bin/bash
# Start the ellipsoid tuner detached (see optimize.py):
#   setsid nohup bash calibration/soybean_bucket/run_optimize.sh 24 > runs/logs/optimize.log 2>&1 &
cd "$(dirname "$0")/../.."
exec /home/s/Documents/newton-dem/runs/perf/vtkcheck/bin/python calibration/soybean_bucket/optimize.py --calls "${1:-24}"
