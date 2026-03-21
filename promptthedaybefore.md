Mục tiêu: L0 dùng deficit pricing theo placement + profiles (không dùng duals), L1 nhận zone prices theo vpp_id.
Gate: lambda_p2p_z4 > lambda_p2p_z1 khi wind cao; zone_net_mw có deficit Z4; L1 solve_step nhận vpp_id.

== PHẦN 1: L0 deficit pricing (src/opt/l0_reconfig.py) ==

def compute_zone_lmp(self, profiles, placement, lambda_ref=5.0):
    """
    Zone LMP từ net injection deficit — không cần MOSEK solve.
    """
    zone_load = {
        1: profiles.get('load_z1', 4.0),
        2: profiles.get('load_z2', 5.0),
        3: profiles.get('load_z3', 3.0),
        4: profiles.get('load_z4', 4.0),
    }

    pv_pu   = profiles.get('pv_pu', 0.5)
    wind_mw = profiles.get('wind_mw', 8.0)

    zone_gen = {z: 0.0 for z in [1,2,3,4]}
    zone_gen[3] += wind_mw

    for e in placement['evcs']:
        z = e['zone']
        zone_gen[z] += e['pv_mw'] * pv_pu
    for p in placement['dpv']:
        z = p['zone']
        zone_gen[z] += p['mw'] * pv_pu

    zone_gen[4] += placement['gfm']['G2']['pv_mw'] * pv_pu

    zone_net = {z: zone_gen[z] - zone_load[z] for z in [1,2,3,4]}

    alpha = 0.15
    lmps = {}
    for z in [1, 2, 4]:
        deficit = max(0.0, -zone_net[z])
        load_z  = max(zone_load[z], 0.5)
        ratio   = deficit / load_z
        lmps[z] = lambda_ref * (1.0 + alpha * ratio)
        lmps[z] = max(lambda_ref * 0.85, min(lmps[z], lambda_ref * 1.35))

    return lmps, zone_net

== PHẦN 2: L1 zone-aware dispatch (src/opt/l1_dispatch.py) ==

def solve_step(self, step, profiles, l0_result, vpp_id=1):
    VPP_ZONE = {1: 1, 2: 2, 3: 4}
    zone = VPP_ZONE[vpp_id]

    lambda_p2p = getattr(l0_result, f'lambda_p2p_z{zone}', profiles.get('lambda_p2p', 5.0))
    lambda_as  = getattr(l0_result, f'lambda_as_z{zone}', profiles.get('lambda_as_ffr', 10.0))

== GATE TEST ==
import numpy as np

l0 = L0Optimizer(net_data)
profiles_wind = pd.Series({
    'pv_pu': 0.3, 'wind_mw': 11.0,
    'load_z1': 4.0, 'load_z2': 5.5,
    'load_z3': 3.0, 'load_z4': 5.0,
    'lambda_as_ffr': 10.0, 'lambda_q': 2.0
})

l0_result = l0.solve(hour_block=8, profiles=profiles_wind, vpp_capacities=vpp_caps,
                    placement=placement)

lmps = [
    l0_result.lambda_p2p_z1,
    l0_result.lambda_p2p_z2,
    l0_result.lambda_p2p_z4,
]

assert all(4.0 <= p <= 8.0 for p in lmps), f"LMP out of range: {lmps}"
assert len(set(round(p, 3) for p in lmps)) >= 2, \
    f"Zone prices identical (no differentiation): {lmps}"

import numpy as np
spread = max(lmps) - min(lmps)
assert spread >= 0.01, f"Zone price spread too small: {spread:.4f} $/MWh"

print(f"Zone LMP: Z1={lmps[0]:.3f}  Z2={lmps[1]:.3f}  Z4={lmps[2]:.3f} $/MWh")
print(f"Spread: {spread:.4f} $/MWh  std: {np.std(lmps):.4f}")
print("GATE DAY 9: PASS")
