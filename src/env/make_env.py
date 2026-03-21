from __future__ import annotations

from typing import Any, Callable

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.env.microgrid_env import MicrogridEnv


class SeededDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns: list[Callable[[], MicrogridEnv]], base_seed: int | None) -> None:
        super().__init__(env_fns)
        self._base_seed = base_seed
        self._reset_count = 0

    def reset(self, seed: int | list[int] | None = None, options: dict | list[dict] | None = None):
        if seed is None:
            if self._base_seed is None:
                seeds = [None for _ in range(self.num_envs)]
            else:
                seeds = [self._base_seed + self._reset_count * self.num_envs + i for i in range(self.num_envs)]
        elif isinstance(seed, (list, tuple, np.ndarray)):
            seeds = list(seed)
        else:
            seeds = [int(seed) + i for i in range(self.num_envs)]

        self._reset_count += 1

        if options is None:
            options_list = [None for _ in range(self.num_envs)]
        elif isinstance(options, (list, tuple)):
            options_list = list(options)
        else:
            options_list = [options for _ in range(self.num_envs)]

        for env_idx in range(self.num_envs):
            maybe_options = {"options": options_list[env_idx]} if options_list[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=seeds[env_idx], **maybe_options)
            self._save_obs(env_idx, obs)

        return self._obs_from_buf()


def make_env(env_kwargs: dict[str, Any], seed: int, rank: int) -> Callable[[], MicrogridEnv]:
    def _init() -> MicrogridEnv:
        env = MicrogridEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_envs(
    n_envs: int = 4,
    env_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    start_method: str | None = None,
    use_dummy: bool = False,
) -> SubprocVecEnv | DummyVecEnv:
    if env_kwargs is None:
        env_kwargs = {}
    else:
        env_kwargs = dict(env_kwargs)

    if "mpc_path" in env_kwargs:
        env_kwargs["mpc_path"] = MicrogridEnv.pre_fix_mpc(env_kwargs["mpc_path"])

    env_fns = [make_env(env_kwargs, seed, i) for i in range(n_envs)]
    if use_dummy or n_envs == 1:
        vec_env = SeededDummyVecEnv(env_fns, seed)
        vec_env.reset()
        return vec_env

    if start_method is None:
        import platform

        start_method = "fork" if platform.system() != "Windows" else "spawn"

    return SubprocVecEnv(env_fns, start_method=start_method)


def make_vec_envs_normalized(
    n_envs: int = 4,
    env_kwargs: dict[str, Any] | None = None,
    seed: int = 42,
    training: bool = True,
    start_method: str | None = None,
    use_dummy: bool = False,
) -> VecNormalize:
    vec_env = make_vec_envs(
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        seed=seed,
        start_method=start_method,
        use_dummy=use_dummy,
    )
    return VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        training=training,
    )
