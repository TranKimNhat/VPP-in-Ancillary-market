#!/usr/bin/env bash
# T34: re-run every security boundary at the ship configuration C.
#
# T32/T33 moved the defaults in `src/phasor/build_case.py`:
#     kp_plim 5.0 -> 1.0   (kppmax_eff = m_p*KPplim = 0.050, inside REGFM_A1's
#                           0.005-0.05; the old 5.0 was 5x above it)
#     kp_i    0.20 -> 0.10
#     ki_i    5.0  -> 3.0
#
# None of those three is a CLI argument, so every campaign below picks them up
# from `CaseSpec` automatically. Every other flag reproduces the original run
# byte for byte, recovered from each campaign's own `config.yaml`, so the only
# difference between these artifacts and the published ones is the controller.
#
# Originals are NOT overwritten: they are the published numbers and the
# comparison is the deliverable.
#
#   campaign            boundary            published value at KPplim = 5
#   dpmax_q060          dP_max_mw           1.185059   <- the headline number
#   dpmax_imaxf15       dP_max_mw           1.185059   (ImaxF = 1.5)
#   phead_dp0p6         P_head_min_mw       0.595336   (kappa 1.007834)
#   phead_dp1p1         P_head_min_mw       1.094680   (kappa 1.004860)
#   pdgoff_h1p0         P_DG_off_max_mw     1.208594   (diesel H = 1.0)
#   pdgoff_h0p1         P_DG_off_max_mw     1.208594   (diesel H = 0.1)
#   pdgoff_gast         P_DG_off_max_mw     1.208594   (GAST governor)
#
# Usage:  bash experiments/t34_rerun_ship_c.sh
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=artifacts/T34_rerun_shipC
COMMON="--tol 0.02 --t-end 8.0 --dt 0.002 --q-max 0.6 --x-f 0.15 --droop-r 0.05"

run () {
  name=$1; shift
  if [ -f "$OUT/$name/boundaries.csv" ]; then
    echo "=== $name  already done, skipping"; echo; return
  fi
  echo "=== $name  $(date +%H:%M:%S)"
  $PY experiments/t20_andes_bisect.py --out "$OUT/$name" $COMMON "$@" 2>&1 \
    | grep -vE "near-zero impedance|^$" | tail -20
  echo
}

run dpmax_q060     --what dp_max --event gen_loss --i-max 2.00 --dp-lo 0.05 --dp-hi 3.0
run dpmax_imaxf15  --what dp_max --event gen_loss --i-max 1.50 --dp-lo 0.05 --dp-hi 3.0
run phead_dp1p1    --what p_head --event gen_loss --i-max 2.00 --dp 1.1 --head-lo 0.05 --head-hi 3.414
run phead_dp0p6    --what p_head --event gen_loss --i-max 2.00 --dp 0.6 --head-lo 0.05 --head-hi 3.414
run pdgoff_h1p0    --what p_dg_off --i-max 2.00 --diesel-mva 1.5 --diesel-h 1.0 --governor TGOV1 --dg-lo 0.0 --dg-hi 1.4
run pdgoff_h0p1    --what p_dg_off --i-max 2.00 --diesel-mva 1.5 --diesel-h 0.1 --governor TGOV1 --dg-lo 0.0 --dg-hi 1.4
run pdgoff_gast    --what p_dg_off --i-max 2.00 --diesel-mva 1.5 --diesel-h 1.0 --governor GAST  --dg-lo 0.0 --dg-hi 1.4

echo "=== done $(date +%H:%M:%S)"
