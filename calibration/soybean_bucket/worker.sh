#!/bin/bash
# Long-running job worker for calibration sweeps (one GPU job at a time).
#
#   runs/calib/queue/jobs.txt   one job per line:  TAG|RUN [RUN ...]|key=value key=value ...
#   runs/calib/queue/done.txt   line numbers finished (appended by this worker)
#
# Append lines to jobs.txt at any time.  Start detached so it outlives the session:
#   setsid nohup bash calibration/soybean_bucket/worker.sh > runs/logs/worker.log 2>&1 &
cd "$(dirname "$0")/../.."
Q=runs/calib/queue
mkdir -p $Q
touch $Q/jobs.txt $Q/done.txt
PY=/home/s/Documents/newton-dem/.venv/bin/python
CV=/home/s/Documents/newton-dem/runs/perf/vtkcheck/bin/python
VID=/home/s/Documents/newton-dem/video-runs
while true; do
  [ -f $Q/stop ] && { echo "$(date) stop file found"; exit 0; }
  n=0; job=""
  while IFS= read -r line; do
    n=$((n + 1))
    [ -z "$line" ] && continue
    case "$line" in \#*) continue;; esac
    if ! grep -qx "$n" $Q/done.txt; then job="$line"; break; fi
  done < $Q/jobs.txt
  if [ -z "$job" ]; then sleep 60; continue; fi
  # one GPU job at a time: wait for anything else (e.g. an orphaned sim) to finish
  while pgrep -f "dem_run.py|dem_clumps.py" > /dev/null; do sleep 20; done
  IFS='|' read -r tag runs sets <<< "$job"
  echo "$(date) job $n: $tag | $runs | $sets"
  $PY calibration/soybean_bucket/calibrate.py --tag $tag --runs $runs --jobs 1 --set $sets 2>&1 | grep -v -i deprecat
  for r in $runs; do
    for d in runs/calib/$tag/*/$r; do
      if [ -d $d/out ] && ls $d/out/frame_0000_particles.vtk > /dev/null 2>&1 && [ ! -f $d/frames.json ]; then
        (cd calibration/soybean_bucket && $CV render.py $VID ../../$d --video | tail -1)
      fi
    done
  done
  echo "$n" >> $Q/done.txt
  echo "$(date) job $n done"
done
