import sys; sys.path.insert(0,'.')
from src.env.microgrid_env_dual import MicrogridEnvDual
env = MicrogridEnvDual(placement_path='artifacts/placement/official_placement_v3.json', mpc_path='data/grid_IEEE123_complete.m', seed=42, ffr_mode='mappo_dual')
inv = {v:k for k,v in env._bus_map.items()}
st = sorted(int(b) for b in env.net.storage['bus']) if not env.net.storage.empty else []
print('STORAGE/BESS ext ids:', sorted(inv.get(b,'?') for b in st))
for ext in [27,36,67,101]:
    pp = env._bus_map.get(ext)
    has = pp is not None and (env.net.sgen['bus']==pp).any()
    n = int((env.net.sgen['bus']==pp).sum()) if pp is not None else 0
    print('bus', ext, '-> sgen count:', n)
