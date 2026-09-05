#!/usr/bin/env bash
# T41: re-run the remaining boundaries at the settled security band.
#
# The band change (+/-1.0 -> +/-0.5 Hz, v_min 0.90 -> 0.88) moved dP_max from
# 1.450 to 0.724 MW -- a 50% shift, larger than the KPplim correction that
# preceded it -- so every number measured at the old band is provisional.
#
# All flags now come from the CLI defaults, which carry the sourced band
# (`--f-band 0.5`, `--v-min 0.88`, `--rocof-max 2.0`); see
# reference/security_band_provenance.md. Only what differs per campaign is
# passed explicitly.
#
#   kappa_dp0p6   P_head_min at dP = 0.6   negative control: `run_dp_max`'s own
#                                          docstring says P_head_min is the
#                                          feasibility bound `P_head >= dP` and
#                                          carries no dynamics, so it should be
#                                          unchanged by a band change. Published
#                                          at the old band: 0.595336, kappa 1.007834.
#   kappa_dp0p7   P_head_min at dP = 0.7   same, close to the new boundary 0.724.
#                 dP = 1.1 (the other published point) is not run: it now exceeds
#                 dP_max, so no headroom survives it and the bisection has nothing
#                 to bracket. That is a consequence of the band, not a failure.
#   pdgoff        P_DG_off_max             published 1.296094 at the old band.
#   topology      dP_max x 4 topologies    published 1.438574 x3 + 1.450098 (G2).
#   event_loc     dP_max x 6 locations     published 1.438574 x6.
#
# T24's I_maxF_crit needs no run here: it is a refit of I_dev(dP) over the T40
# metrics, done separately.
#
# Usage:  bash experiments/t41_rerun_final_band.sh
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=artifacts/T41_final_band
COMMON="--event gen_loss --tol 0.02 --t-end 8.0 --dt 0.002 --q-max 0.6 \
        --x-f 0.15 --droop-r 0.05 --i-max 2.00"

run () {
  script=$1; name=$2; shift 2
  if [ -f "$OUT/$name/boundaries.csv" ]; then
    echo "=== $name  already done, skipping"; echo; return
  fi
  echo "=== $name  $(date +%H:%M:%S)"
  $PY "experiments/$script" --out "$OUT/$name" $COMMON "$@" 2>&1 \
    | grep -vE "near-zero impedance|^$" | tail -3
  echo
}

run t20_andes_bisect.py kappa_dp0p6 --what p_head --dp 0.6 --head-lo 0.05 --head-hi 3.414
run t20_andes_bisect.py kappa_dp0p7 --what p_head --dp 0.7 --head-lo 0.05 --head-hi 3.414
run t20_andes_bisect.py pdgoff      --what p_dg_off --diesel-mva 1.5 --diesel-h 1.0 \
                                    --governor TGOV1 --dg-lo 0.0 --dg-hi 1.4
run t22_topology_sweep.py topology  --seed 7 --n 3 --dp-lo 0.05 --dp-hi 3.0
run t22_topology_sweep.py event_loc --seed 7 --n 0 --event-buses 1,76,102,41,88,33 \
                                    --dp-lo 0.05 --dp-hi 3.0

echo "=== done $(date +%H:%M:%S)"
