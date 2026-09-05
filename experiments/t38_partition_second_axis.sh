#!/usr/bin/env bash
# T38: does the robust/fragile partition hold on a second perturbation axis?
#
# T34-T37 found that across one configuration change (KPplim 5 -> 1 plus the
# inner-loop pair), the security quantities split cleanly:
#
#   survived unchanged   P_head_min and its kappa envelope (to six decimals),
#                        df_ss, topology/location invariance, ImaxF-does-not-bind
#   moved                dP_max (+21.4%), kappa_os, I_maxF_crit, diesel-off ranking
#
# and that the survivors are the quantities set by energy balance and network
# topology while the movers are the ones set by converter transient shaping.
#
# One configuration change is one point, and one point is not a partition. This
# sweeps a second axis and asks whether the same quantities sit on the same
# sides. Two representatives, one from each side:
#
#   dP_max        the fragile side's headline
#   P_head_min    the robust side's headline (published: 1.094680 at dP = 1.1,
#                 unchanged to six decimals by the KPplim change)
#
# Two axes, because they test different things:
#
#   x_f    a PHYSICAL coupling reactance. Both quantities may legitimately move
#          with it; what the partition predicts is that P_head_min moves only as
#          far as the power flow forces and dP_max moves more.
#   inner  the CONTROL axis, and the more direct test: the partition claims
#          quantities set by converter control are the fragile ones, so moving
#          only the inner-loop gains -- within the region T33 verified stable at
#          KPplim = 1.0 -- should move dP_max and leave P_head_min alone.
#
# Both perturbations stay inside T33's stable box; nothing here re-opens the
# stability question.
#
# Usage:  bash experiments/t38_partition_second_axis.sh
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
OUT=artifacts/T38_partition
COMMON="--event gen_loss --tol 0.02 --t-end 8.0 --dt 0.002 --q-max 0.6 \
        --droop-r 0.05 --i-max 2.00"

run () {
  name=$1; shift
  if [ -f "$OUT/$name/boundaries.csv" ]; then
    echo "=== $name  already done, skipping"; echo; return
  fi
  echo "=== $name  $(date +%H:%M:%S)"
  $PY experiments/t20_andes_bisect.py --out "$OUT/$name" $COMMON "$@" 2>&1 \
    | grep -vE "near-zero impedance|^$" | tail -4
  echo
}

# --- axis 1: x_f, physical. Nominal 0.15; both inside T33's stable region. ---
for XF in 0.10 0.20; do
  run "xf${XF}_dpmax"  --what dp_max --x-f $XF --dp-lo 0.05 --dp-hi 3.0
  run "xf${XF}_phead"  --what p_head --x-f $XF --dp 1.1 --head-lo 0.05 --head-hi 3.414
done

# --- axis 2: inner loops, control. Requires the CaseSpec defaults to be
# overridden, which t20 has no flag for, so these run through a tiny wrapper. ---
echo "=== inner-loop axis: see t38_inner_axis.py"

echo "=== done $(date +%H:%M:%S)"
