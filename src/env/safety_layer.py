from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def apply_safety_layer(
    actions_dict: Dict[str, List[Any]],
    env_state: Dict[str, Any],
    placement_config: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    safe_actions = {
        "evcs_pv": [list(item) for item in actions_dict.get("evcs_pv", [])],
        "evcs_bess": [list(item) for item in actions_dict.get("evcs_bess", [])],
        "evcs_v2g": [float(item) for item in actions_dict.get("evcs_v2g", [])],
        "dpv": [list(item) for item in actions_dict.get("dpv", [])],
    }

    stage1 = stage2 = stage3 = stage4 = stage5 = 0

    evcs_cfgs = placement_config or []
    n_evcs = len(evcs_cfgs)
    dpv_p_rated = env_state.get("dpv_p_rated", [])
    dpv_s_rated = env_state.get("dpv_s_rated", dpv_p_rated)
    n_dpv = len(dpv_p_rated)

    s_margin_pen = 0.0

    # Stage 1: type clip
    for i, (p_curt, q) in enumerate(safe_actions["evcs_pv"]):
        cfg = evcs_cfgs[i]
        p_curt_new = float(np.clip(p_curt, 0.0, float(cfg["pv_mw"])))
        s_rated = float(cfg.get("inverter_mva", cfg["pv_mw"]))
        q_max = 0.436 * s_rated
        q_new = float(np.clip(q, -q_max, q_max))
        if p_curt_new != p_curt or q_new != q:
            stage1 += 1
        safe_actions["evcs_pv"][i] = [p_curt_new, q_new]

    for i, (p, q) in enumerate(safe_actions["evcs_bess"]):
        cfg = evcs_cfgs[i]
        p_new = float(np.clip(p, -float(cfg["bess_mw"]), float(cfg["bess_mw"])))
        s_rated = float(cfg.get("inverter_mva", cfg["bess_mw"]))
        q_max = 0.436 * s_rated
        q_new = float(np.clip(q, -q_max, q_max))
        if p_new != p or q_new != q:
            stage1 += 1
        safe_actions["evcs_bess"][i] = [p_new, q_new]

    for i, p in enumerate(safe_actions["evcs_v2g"]):
        cfg = evcs_cfgs[i]
        p_cap = min(float(cfg["v2g_mw"]), 0.75)
        p_new = float(np.clip(p, 0.0, p_cap))
        if p_new != p:
            stage1 += 1
        safe_actions["evcs_v2g"][i] = p_new

    for i, (p_curt, q) in enumerate(safe_actions["dpv"]):
        p_rated = float(dpv_p_rated[i])
        p_curt_new = float(np.clip(p_curt, 0.0, p_rated))
        s_rated = float(dpv_s_rated[i])
        q_max = 0.436 * s_rated
        q_new = float(np.clip(q, -q_max, q_max))
        if p_curt_new != p_curt or q_new != q:
            stage1 += 1
        safe_actions["dpv"][i] = [p_curt_new, q_new]

    # Stage 4: voltage droop
    v_bus = env_state.get("v_bus", np.ones(123))
    agent_buses = env_state.get("agent_buses", list(range(n_evcs * 3 + n_dpv)))
    k_q = 0.1

    for i, (p_curt, q) in enumerate(safe_actions["evcs_pv"]):
        cfg = evcs_cfgs[i]
        s_rated = float(cfg.get("inverter_mva", cfg["pv_mw"]))
        q_max = 0.436 * s_rated
        bus_idx = int(agent_buses[i])
        v = float(v_bus[bus_idx]) if bus_idx < len(v_bus) else 1.0
        if v < 0.94 or v > 1.06:
            delta_q = k_q * (1.0 - v) * s_rated
            q_new = float(np.clip(q + delta_q, -q_max, q_max))
            if q_new != q:
                stage4 += 1
            safe_actions["evcs_pv"][i][1] = q_new

    for i, (p, q) in enumerate(safe_actions["evcs_bess"]):
        cfg = evcs_cfgs[i]
        s_rated = float(cfg.get("inverter_mva", cfg["bess_mw"]))
        q_max = 0.436 * s_rated
        bus_idx = int(agent_buses[i + n_evcs])
        v = float(v_bus[bus_idx]) if bus_idx < len(v_bus) else 1.0
        if v < 0.94 or v > 1.06:
            delta_q = k_q * (1.0 - v) * s_rated
            q_new = float(np.clip(q + delta_q, -q_max, q_max))
            if q_new != q:
                stage4 += 1
            safe_actions["evcs_bess"][i][1] = q_new

    for i, (p_curt, q) in enumerate(safe_actions["dpv"]):
        s_rated = float(dpv_s_rated[i])
        q_max = 0.436 * s_rated
        bus_idx = int(agent_buses[i + 3 * n_evcs])
        v = float(v_bus[bus_idx]) if bus_idx < len(v_bus) else 1.0
        if v < 0.94 or v > 1.06:
            delta_q = k_q * (1.0 - v) * s_rated
            q_new = float(np.clip(q + delta_q, -q_max, q_max))
            if q_new != q:
                stage4 += 1
            safe_actions["dpv"][i][1] = q_new

    # Stage 3: frequency droop
    delta_f = float(env_state.get("delta_f", 0.0))
    if abs(delta_f) > 0.2:
        k_droop = 0.05
        p_flex_up = env_state.get("p_flex_up", np.zeros(n_evcs * 3 + n_dpv))
        p_flex_down = env_state.get("p_flex_down", np.zeros(n_evcs * 3 + n_dpv))

        for i, (p_curt, q) in enumerate(safe_actions["evcs_pv"]):
            cfg = evcs_cfgs[i]
            p_rated = float(cfg["pv_mw"])
            p_set = p_rated - p_curt
            delta_p = -k_droop * delta_f * float(p_flex_up[i])
            p_new = float(np.clip(p_set + delta_p, -float(p_flex_down[i]), float(p_flex_up[i])))
            if p_new != p_set:
                stage3 += 1
            safe_actions["evcs_pv"][i][0] = float(np.clip(p_rated - p_new, 0.0, p_rated))

        for i, (p, q) in enumerate(safe_actions["evcs_bess"]):
            delta_p = -k_droop * delta_f * float(p_flex_up[i + n_evcs])
            p_new = float(
                np.clip(p + delta_p, -float(p_flex_down[i + n_evcs]), float(p_flex_up[i + n_evcs]))
            )
            if p_new != p:
                stage3 += 1
            safe_actions["evcs_bess"][i][0] = p_new

        for i, p in enumerate(safe_actions["evcs_v2g"]):
            delta_p = -k_droop * delta_f * float(p_flex_up[i + 2 * n_evcs])
            p_new = float(
                np.clip(p + delta_p, -float(p_flex_down[i + 2 * n_evcs]), float(p_flex_up[i + 2 * n_evcs]))
            )
            if p_new != p:
                stage3 += 1
            safe_actions["evcs_v2g"][i] = p_new

        for i, (p_curt, q) in enumerate(safe_actions["dpv"]):
            p_rated = float(dpv_p_rated[i])
            p_set = p_rated - p_curt
            delta_p = -k_droop * delta_f * float(p_flex_up[i + 3 * n_evcs])
            p_new = float(
                np.clip(p_set + delta_p, -float(p_flex_down[i + 3 * n_evcs]), float(p_flex_up[i + 3 * n_evcs]))
            )
            if p_new != p_set:
                stage3 += 1
            safe_actions["dpv"][i][0] = float(np.clip(p_rated - p_new, 0.0, p_rated))

    # Stage 2: EV obligation
    evcs_p_ch_min = env_state.get("evcs_p_ch_min", [])
    for i, cfg in enumerate(evcs_cfgs):
        if i >= len(evcs_p_ch_min):
            break
        p_ch_min = float(evcs_p_ch_min[i])
        p_bess = float(safe_actions["evcs_bess"][i][0])
        p_v2g = float(safe_actions["evcs_v2g"][i])
        current_ch = max(0.0, -p_bess)
        if current_ch + 1e-6 < p_ch_min:
            need = p_ch_min - current_ch
            if p_v2g > 0:
                reduce_v2g = min(need, p_v2g)
                p_v2g -= reduce_v2g
                need -= reduce_v2g
            if need > 0:
                p_bess = max(-float(cfg["bess_mw"]), p_bess - need)
            safe_actions["evcs_bess"][i][0] = p_bess
            safe_actions["evcs_v2g"][i] = p_v2g
            stage2 += 1

    # Stage 5: inverter S^2 limit
    for i, (p_curt, q) in enumerate(safe_actions["evcs_pv"]):
        cfg = evcs_cfgs[i]
        s_rated = float(cfg.get("inverter_mva", cfg["pv_mw"]))
        p_set = float(cfg["pv_mw"]) - p_curt
        s_sq = p_set * p_set + q * q
        if s_sq > s_rated * s_rated:
            q_allowed = float(np.sqrt(max(0.0, s_rated * s_rated - p_set * p_set)))
            q_new = float(np.clip(q, -q_allowed, q_allowed))
            s_margin_pen += max(0.0, (s_sq - s_rated * s_rated) / (s_rated * s_rated))
            if q_new != q:
                stage5 += 1
            safe_actions["evcs_pv"][i][1] = q_new

    for i, (p, q) in enumerate(safe_actions["evcs_bess"]):
        cfg = evcs_cfgs[i]
        s_rated = float(cfg.get("inverter_mva", cfg["bess_mw"]))
        s_sq = p * p + q * q
        if s_sq > s_rated * s_rated:
            q_allowed = float(np.sqrt(max(0.0, s_rated * s_rated - p * p)))
            q_new = float(np.clip(q, -q_allowed, q_allowed))
            s_margin_pen += max(0.0, (s_sq - s_rated * s_rated) / (s_rated * s_rated))
            if q_new != q:
                stage5 += 1
            safe_actions["evcs_bess"][i][1] = q_new

    for i, (p_curt, q) in enumerate(safe_actions["dpv"]):
        s_rated = float(dpv_s_rated[i])
        p_set = float(dpv_p_rated[i]) - p_curt
        s_sq = p_set * p_set + q * q
        if s_sq > s_rated * s_rated:
            q_allowed = float(np.sqrt(max(0.0, s_rated * s_rated - p_set * p_set)))
            q_new = float(np.clip(q, -q_allowed, q_allowed))
            s_margin_pen += max(0.0, (s_sq - s_rated * s_rated) / (s_rated * s_rated))
            if q_new != q:
                stage5 += 1
            safe_actions["dpv"][i][1] = q_new

    max_v2g_mw = max(safe_actions["evcs_v2g"], default=0.0)
    safety_info = {
        "stage1_activations": int(stage1),
        "stage2_activations": int(stage2),
        "stage3_activations": int(stage3),
        "stage4_activations": int(stage4),
        "stage5_activations": int(stage5),
        "s_margin_pen": float(s_margin_pen),
        "max_v2g_mw": float(max_v2g_mw),
    }
    safety_info["total_safety_activations"] = int(
        stage1 + stage2 + stage3 + stage4 + stage5
    )

    return safe_actions, safety_info
