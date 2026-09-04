from __future__ import annotations

import math

from src.layer2_control.reward import MarketPrices, compute_reward


def test_market_reward_em_only_term() -> None:
    prices = MarketPrices(lambda_e=50.0, lambda_q=0.0, lambda_cap=100.0, lambda_act=200.0)
    reward = compute_reward(
        p_actual=1.0,
        q_actual=0.0,
        r_commit=0.0,
        r_delivered=0.0,
        voltage_violation=0.0,
        scenario_type="em_normal",
        prices=prices,
    )
    assert math.isclose(reward, 50.0, rel_tol=0.0, abs_tol=1e-9)


def test_market_reward_am_capacity_and_activation() -> None:
    prices = MarketPrices(lambda_e=0.0, lambda_q=0.0, lambda_cap=100.0, lambda_act=200.0)
    reward = compute_reward(
        p_actual=0.0,
        q_actual=0.0,
        r_commit=2.0,
        r_delivered=2.0,
        voltage_violation=0.0,
        scenario_type="am_ffr_mid",
        prices=prices,
    )
    assert math.isclose(reward, 600.0, rel_tol=0.0, abs_tol=1e-9)


def test_market_reward_undersupply_penalty() -> None:
    prices = MarketPrices(lambda_e=0.0, lambda_q=0.0, lambda_cap=0.0, lambda_act=200.0, c_undersupply=3.0)
    reward = compute_reward(
        p_actual=0.0,
        q_actual=0.0,
        r_commit=2.0,
        r_delivered=0.5,
        voltage_violation=0.0,
        scenario_type="am_ffr_mid",
        prices=prices,
    )
    assert math.isclose(reward, -800.0, rel_tol=0.0, abs_tol=1e-9)


def test_market_reward_voltage_penalty() -> None:
    prices = MarketPrices(lambda_e=0.0, lambda_q=0.0, lambda_cap=0.0, lambda_act=0.0)
    reward = compute_reward(
        p_actual=0.0,
        q_actual=0.0,
        r_commit=0.0,
        r_delivered=0.0,
        voltage_violation=0.05,
        scenario_type="em_normal",
        prices=prices,
    )
    assert math.isclose(reward, -0.1, rel_tol=0.0, abs_tol=1e-9)
