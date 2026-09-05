#!/usr/bin/env bash
# T35: re-run the topology and event-location sweeps at the ship configuration C.
#
# These were optional invariance checks under the published controller. They are
# not optional any more. T34 shows the dP_max boundary moves from
# frequency-limited to VOLTAGE-limited at ship C: at the boundary the first
# insecure point fails on `v_min` alone (0.8996 < 0.90) with RoCoF still at 1.918
# against a 2.0 threshold.
#
# That removes the premise both sweeps concluded on. T22/T23 argued that
# topology and disturbance location cannot move the boundary because the binding
# quantity is a bulk one (RoCoF, nadir) rather than a local one -- T23 measured
# the relative spread across event locations as 0.0012% for nadir and 0.070% for
# RoCoF, against 0.053% for V_min and 0.63% for mu_I. If V_min binds instead,
# the boundary is set by a quantity ~44x more location-sensitive than nadir, and
# the invariance conclusion has to be re-derived rather than reused.
#
# Same seed (7) and the same counts as the originals, so the topology and event
# sets are identical and only the controller differs.
#
#   sweep        published dP_max at KPplim = 5
#   topology     1.185059 for all four topologies (G0, G1, G2, G3)
#   event_loc    1.185059 for five of six locations; 1.173535 at bus 102
#
# Usage:  bash experiments/t35_rerun_sweeps_ship_c.sh
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=artifacts/T35_sweeps_shipC
COMMON="--seed 7 --event gen_loss --tol 0.02 --t-end 8.0 --dt 0.002 \
        --q-max 0.6 --x-f 0.15 --droop-r 0.05 --i-max 2.00 --dp-lo 0.05 --dp-hi 3.0"

run () {
  name=$1; shift
  if [ -f "$OUT/$name/boundaries.csv" ]; then
    echo "=== $name  already done, skipping"; echo; return
  fi
  echo "=== $name  $(date +%H:%M:%S)"
  $PY experiments/t22_topology_sweep.py --out "$OUT/$name" $COMMON "$@" 2>&1 \
    | grep -vE "near-zero impedance|^$" | tail -25
  echo
}

# Four topologies: G0 plus three alternatives, as in T22.
run topology   --n 3
# Six event locations on G0, as in T23. `--event-buses` selects the location
# sweep; the buses are the same percentile-spaced set the original picked.
run event_loc  --n 0 --event-buses 1,76,102,41,88,33

echo "=== done $(date +%H:%M:%S)"
