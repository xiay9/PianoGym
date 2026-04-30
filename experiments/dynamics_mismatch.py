"""
Experiment A5: Dynamics misspecification robustness (Tornado-plot sensitivity battery)

Tolerance test for misspecification in environment parameters
(forgetting rate eta_forget, fatigue coefficient gamma_f, and smoothness kappa).
Scan offsets around the baseline and record resulting changes in
TTM / Feasible / Avg Fatigue for each policy.

Output format:
  param_name, scale, metric, value, seed
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import PianoGymEnv  # noqa: E402
from src.agents import get_agent  # noqa: E402
from src.data_logging import ensure_data_dirs, resolve_data_path  # noqa: E402
from src.metrics import compute_safety_stats  # noqa: E402
from src.safety import ExternalSafetyGuard  # noqa: E402


# ========== Configuration ==========

PROFILE = {
    'name': 'balanced',
    'left_weakness': 0.15,        # v2: increase left-hand weakness to make the task harder
    'enforce_constraint': True    # v2.1: activate the fatigue constraint to separate feasible_rate values
}
AGENTS = ('linucb', 'thompson', 'pianoMPC', 'dqn', 'bayesianmab', 'ccb_df', 'safe_ac', 'autocurriculum')

NUM_SEEDS = 10  # v2.1: increase to 10 seeds to reduce variance
SEED_OFFSET = 97

# Metrics to track.
METRICS = ['ttm', 'feasible_rate', 'avg_fatigue', 'learn_slope']

# Key point: define the actual runtime baseline for all scanned parameters.
# This matches A1/A2 and is not the synthetic baseline stored in each sweep.
# It keeps the reference point consistent across eta_forget / gamma_f / kappa scans.
BASELINE = {
    'eta_forget': 0.02,  # A1/A2 project default
    'gamma_f': 0.4,      # A1/A2 project default
    'kappa': 5.0,        # A1/A2 project default
}

# Relative perturbation ranges around the baseline.
PARAM_SWEEPS = {
    'eta_forget': {
        'baseline': BASELINE['eta_forget'],
        'scales': [-0.3, -0.15, 0.0, 0.15, 0.3],
        'bounds': (0.01, 0.05),
    },
    'gamma_f': {
        'baseline': BASELINE['gamma_f'],
        'scales': [-0.5, -0.25, 0.0, 0.25, 0.5],
        'bounds': (0.1, 0.8),
    },
    'kappa': {
        'baseline': BASELINE['kappa'],
        'scales': [-0.5, -0.25, 0.0, 0.25, 0.5],
        'bounds': (2.5, 7.5),
    },
}

AGENT_GUARD_PARAMS = {
    "guard_delta": 0.08,
    "guard_horizon": 1,
    "guard_safety_margin": 0.05,
}

EXTERNAL_GUARD_CONFIG = {
    "guard_delta": 0.08,
    "safety_margin": 0.05,
}


# ========== Helper functions ==========

def _set_all_to_baseline(env: PianoGymEnv) -> None:
    """
    Reset all scanned parameters back to the shared baseline.

    This keeps parameter groups consistent. For example, when scanning gamma_f,
    eta_forget and kappa are reset to their baseline values, and vice versa.
    The goal is to prevent leftover non-baseline values from a previous sweep
    from contaminating the next one.
    """
    for param_name, baseline_val in BASELINE.items():
        setattr(env.cfg, param_name, baseline_val)

    # Rebind so all changes take effect.
    if hasattr(env, "rebind_from_cfg") and callable(env.rebind_from_cfg):
        env.rebind_from_cfg()


def _make_agent(
    agent_name: str,
    env: PianoGymEnv,
    context_dim: int,
    seed: int,
) -> object:
    """Create an agent with the standard experiment configuration."""
    num_actions = env.cfg.num_actions
    state_dim = env.cfg.K  # skill dimension as state

    if agent_name == 'linucb':
        return get_agent(
            'linucb',
            num_actions=num_actions,
            context_dim=context_dim,
            alpha=2.0,
            ridge=1e-3,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )
    elif agent_name == 'thompson':
        return get_agent(
            'thompson',
            num_actions=num_actions,
            context_dim=context_dim,
            v=0.8,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )
    elif agent_name == 'pianoMPC':
        return get_agent(
            'pianoMPC',
            num_actions=num_actions,
            reward_weights=env.cfg.w_reward,
            reward_norm=env.cfg.reward_norm,
            fatigue_limit=env.cfg.fatigue_threshold,
            rest_recovery=env.cfg.rest_recovery,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )
    elif agent_name == 'dqn':
        return get_agent(
            'dqn',
            num_actions=num_actions,
            state_dim=state_dim,
            hidden_dim=64,
            epsilon=0.2,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )
    elif agent_name == 'bayesianmab':
        return get_agent(
            'bayesianmab',
            num_actions=num_actions,
            discount=0.995,
            prior_mu=0.0,
            prior_sigma=1.0,
            sigma_noise=0.5,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )
    elif agent_name == 'ccb_df':
        return get_agent(
            'ccb_df',
            num_actions=num_actions,
            context_dim=context_dim,
            delay_window=1,
            weight_clip=10.0,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )
    elif agent_name == 'safe_ac':
        return get_agent(
            'safe_ac',
            num_actions=num_actions,
            state_dim=state_dim,
            hidden_dim=32,
            cost_limit=env.cfg.fatigue_threshold,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )
    elif agent_name == 'autocurriculum':
        return get_agent(
            'autocurriculum',
            num_actions=num_actions,
            advance_threshold=0.7,
            regress_threshold=0.3,
            seed=seed,
            **AGENT_GUARD_PARAMS,
        )

    raise ValueError(f"Unsupported agent: {agent_name}")


def _apply_param_perturbation(env: PianoGymEnv, param_name: str, scale: float) -> Tuple[float, float]:
    """
    Apply a parameter perturbation to the environment configuration.

    Returns ``(base_val, eff_val)``.
    The key fix is to apply a relative perturbation with boundary clipping,
    then synchronize submodules and rebind so the change actually takes effect.

    Parameters:
    - param_name: 'eta_forget', 'gamma_f', 'kappa'
    - scale: relative offset (-0.3 to +0.3 means ±30%)

    Returns:
    - (base_val, eff_val): baseline value and effective value
    """
    cfg = env.cfg
    sweep_info = PARAM_SWEEPS[param_name]

    # Key fix #1: use the environment's true current default as the baseline.
    try:
        base_val = float(getattr(cfg, param_name))
    except (AttributeError, TypeError):
        base_val = sweep_info['baseline']

    # Key fix #2: use the experiment-defined relative perturbation plus dynamic bounds.
    # This avoids flattening the sweep when static bounds do not match the runtime baseline.
    eff_val = base_val * (1.0 + scale)

    # Dynamic bounds that exactly cover the current relative sweep range.
    scales = sweep_info['scales']
    dyn_lo = base_val * (1.0 + min(scales))
    dyn_hi = base_val * (1.0 + max(scales))

    # Static bounds from the configuration, if provided.
    static_lo, static_hi = sweep_info.get('bounds', (None, None))
    if static_lo is not None:
        dyn_lo = max(dyn_lo, static_lo)
    if static_hi is not None:
        dyn_hi = min(dyn_hi, static_hi)

    # Extra safety check: enforce non-negativity.
    dyn_lo = max(0.0, dyn_lo)
    dyn_hi = max(0.0, dyn_hi)

    # Clip to the intersection of dynamic and static bounds.
    eff_val = float(np.clip(eff_val, dyn_lo, dyn_hi))

    # Key fix #3: write back to cfg (main path).
    setattr(cfg, param_name, eff_val)

    # Key fix #4: synchronize possible submodule fields (fatigue_model / dynamics_model).
    field_variants = {
        'gamma_f': ['gamma_f', 'fatigue_gamma'],
        'kappa': ['kappa', 'sigmoid_kappa', 'gate_kappa'],
        'eta_forget': ['eta_forget', 'forget_rate', 'forgetting_rate'],
    }
    variants = field_variants.get(param_name, [param_name])

    for variant in variants:
        if hasattr(env, 'fatigue_model') and hasattr(env.fatigue_model, variant):
            setattr(env.fatigue_model, variant, eff_val)
        if hasattr(env, 'dynamics_model') and hasattr(env.dynamics_model, variant):
            setattr(env.dynamics_model, variant, eff_val)

    # Key fix #5: rebind after writing so the new value enters the dynamics.
    if hasattr(env, "rebind_from_cfg") and callable(env.rebind_from_cfg):
        env.rebind_from_cfg()

    # Key fix #6: assert that the write took effect.
    written_val = float(getattr(cfg, param_name))
    assert abs(written_val - eff_val) < 1e-9, \
        f"Parameter write failed: {param_name} expected {eff_val}, got {written_val}"

    # Key fix #7: confirm that ``eff_val`` actually changes when clipping is not active.
    if abs(scale) >= 1e-9 and abs(eff_val - base_val) < 1e-9:
        clamped_lower = math.isclose(base_val, dyn_lo, rel_tol=1e-9, abs_tol=1e-9) and scale < 0
        clamped_upper = math.isclose(base_val, dyn_hi, rel_tol=1e-9, abs_tol=1e-9) and scale > 0
        if not (clamped_lower or clamped_upper):
            raise AssertionError(
                f"Parameter did not change: {param_name} base={base_val} eff={eff_val} scale={scale}"
            )

    return base_val, eff_val


def _compute_learning_slope(infos: List[dict], mastery_window: int) -> float:
    """
    Compute the learning slope: the change rate of the mastery indicator over time.

    The idea is that the subtle effect of gamma_f / kappa on learning gain may
    be hidden when looking only at TTM. A slope-style signal can capture changes
    in learning speed more sensitively.

    Parameters:
    - infos: trajectory-info list
    - mastery_window: mastery window size

    Returns:
    - the slope of mastery probability per step; NaN if data is insufficient
    """
    if not infos or len(infos) < 5:
        return float('nan')

    # Build the 0/1 mastery-indicator sequence.
    mastery_indicators = np.array([
        1.0 if info.get('mastery_count', 0) >= mastery_window else 0.0
        for info in infos
    ], dtype=float)

    # Time axis (1-indexed).
    time_steps = np.arange(1.0, len(infos) + 1, dtype=float)

    # Least-squares slope fit.
    # For y = a*x + b, only ``a`` is needed.
    mean_x = np.mean(time_steps)
    mean_y = np.mean(mastery_indicators)

    numerator = np.sum((time_steps - mean_x) * (mastery_indicators - mean_y))
    denominator = np.sum((time_steps - mean_x) ** 2)

    if denominator < 1e-12:
        return float('nan')

    slope = numerator / denominator
    return float(slope)


def _run_one_config(
    agent_name: str,
    param_name: str,
    scale: float,
    seed: int,
) -> Dict[str, object]:
    """
    Run a single configuration (agent × param × scale × seed) and collect metrics.

    Key design points:
    1. Create the environment from the project defaults (PianoGymEnv with cfg=None).
    2. Apply parameter perturbations only after reset(), otherwise internal rebinds may overwrite them.
    3. Reset the global cfg to baseline before every call to avoid cross-run contamination.
    4. Read ``base_val`` from the config of the environment created for this call.

    Returns:
    - ttm, feasible_rate, avg_fatigue, plus the base / effective parameter values
    """
    from configs import cfg as global_cfg

    # Key fix #-1: reset the global cfg before every call.
    # A previous scale may have modified cfg through ``_apply_param_perturbation``.
    for param_name_tmp, baseline_val in BASELINE.items():
        setattr(global_cfg, param_name_tmp, baseline_val)

    env = PianoGymEnv(seed=seed)
    obs = env.reset(profile=PROFILE)

    # Key fix #0: align every parameter to the common baseline.
    # For example, when scanning gamma_f, eta_forget and kappa must be reset too.
    _set_all_to_baseline(env)

    # Key fix #1: apply parameter perturbations only after reset().
    # reset() may rebuild internal state, so the perturbation must happen later.
    base_val, eff_val = _apply_param_perturbation(env, param_name, scale)

    # Extra rebind as a safeguard when the implementation exposes that hook.
    if hasattr(env, "rebind_from_cfg") and callable(env.rebind_from_cfg):
        env.rebind_from_cfg()

    context_dim = len(obs['context']) + len(obs['action_features'][0])
    agent = _make_agent(agent_name, env, context_dim, seed)
    agent.reset()
    guard = ExternalSafetyGuard(
        env,
        enabled=True,
        **EXTERNAL_GUARD_CONFIG,
    )
    guard.reset()

    trajectory = {
        'rewards': [],
        'obs_history': [obs],
        'info_history': [],
        'actions': [],
        'proposed_actions': [],
        'guard_history': [],
        'steps': 0,
        'done': False,
    }

    while not env.done:
        action = agent.select_action(obs)
        executed_action, guard_decision = guard.enforce(int(action), obs)
        next_obs, reward, done, info = env.step(executed_action)
        guard.annotate_info(info)

        trajectory['rewards'].append(reward)
        trajectory['obs_history'].append(next_obs)
        trajectory['info_history'].append(info)
        trajectory['actions'].append(executed_action)
        trajectory['proposed_actions'].append(action)
        trajectory['guard_history'].append(guard_decision.as_dict())
        trajectory['steps'] += 1

        next_obs_with_done = dict(next_obs)
        next_obs_with_done['done'] = done
        next_obs_with_done['_info'] = info
        if hasattr(agent, "update") and callable(agent.update):
            agent.update(obs, executed_action, reward, next_obs_with_done)

        obs = next_obs

        if done:
            break

    # Key fixes #2 and #3: compute metrics locally with one consistent definition.
    infos = trajectory['info_history']

    # TTM: first step at which mastery is reached.
    ttm = next(
        (i for i, info in enumerate(infos, start=1)
         if info.get('mastery_count', 0) >= env.cfg.mastery_window
            or info.get('mastery_flag', False)),
        len(infos)
    )
    ttm = float(ttm)

    safety_stats = compute_safety_stats(infos)
    feasible_rate = safety_stats.feasible_rate
    avg_fatigue = safety_stats.avg_fatigue

    # Learning slope: mastery probability over time, used as a more sensitive learning-speed metric.
    learn_slope = _compute_learning_slope(infos, env.cfg.mastery_window)

    return {
        'agent': agent_name,
        'profile': PROFILE['name'],
        'param_name': param_name,
        'scale': scale,
        'seed': seed,
        'ttm': ttm,
        'feasible_rate': feasible_rate,
        'avg_fatigue': avg_fatigue,
        'learn_slope': learn_slope,
        f'{param_name}_base': base_val,
        f'{param_name}_eff': eff_val,
    }


def smoke_test_env_sensitivity(
    param_name: str,
    scales: List[float] = None,
    steps: int = 100,
    seed: int = 123,
) -> Dict[str, object]:
    """
    Smoke test v2: use an actual learning agent to check whether parameter changes
    alter TTM / fatigue behavior.

    Key observation: gamma_f and kappa do not directly change fatigue accumulation;
    they change learning dynamics instead.
    - gamma_f reduces learning gain under fatigue
    - kappa controls the steepness of the difficulty gate

    That means a fixed-action test is misleading because no learning happens.
    Instead, this test uses a LinUCB agent and looks for sensitivity through
    differences in learning speed.

    Parameters:
    - param_name: 'gamma_f' or 'kappa'
    - scales: list of relative perturbations to test
    - steps: number of steps per configuration
    - seed: random seed

    Returns:
    - a dictionary containing scales, TTM values, feasible rates, and related fields
    """
    if scales is None:
        scales = PARAM_SWEEPS[param_name]['scales']

    print(f"\n{'=' * 80}")
    print(f"Smoke test #{param_name}: parameter sensitivity of TTM/feasible_rate via LinUCB learning")
    print(f"{'=' * 80}")

    results_list = []

    for scale in scales:
        from configs import cfg as global_cfg

        # Reset cfg.
        for pname, bval in BASELINE.items():
            setattr(global_cfg, pname, bval)

        env = PianoGymEnv(seed=seed)
        obs = env.reset(profile=PROFILE)
        _set_all_to_baseline(env)

        # Apply the parameter perturbation.
        base_val, eff_val = _apply_param_perturbation(env, param_name, scale)
        if hasattr(env, "rebind_from_cfg") and callable(env.rebind_from_cfg):
            env.rebind_from_cfg()

        # Create the agent.
        context_dim = len(obs['context']) + len(obs['action_features'][0])
        agent = get_agent('linucb', num_actions=env.cfg.num_actions, context_dim=context_dim, alpha=2.0, ridge=1e-3, seed=seed)
        agent.reset()
        guard = ExternalSafetyGuard(
            env,
            enabled=True,
            **EXTERNAL_GUARD_CONFIG,
        )
        guard.reset()

        # Run the learning trajectory.
        infos = []
        mastery_count = 0

        for step_idx in range(steps):
            if env.done:
                break

            # Agent selects an action.
            action = agent.select_action(obs)
            executed_action, _ = guard.enforce(int(action), obs)
            next_obs, reward, done, info = env.step(executed_action)
            guard.annotate_info(info)

            # Collect trajectory info.
            infos.append(info)

            # Update mastery status using ``mastery_count`` from info.
            mastery_count = info.get('mastery_count', 0)

            # Update the agent.
            next_obs_with_done = dict(next_obs)
            next_obs_with_done['done'] = done
            next_obs_with_done['_info'] = info
            agent.update(obs, executed_action, reward, next_obs_with_done)

            obs = next_obs

        # Compute metrics.
        ttm = next(
            (i for i, info in enumerate(infos, start=1)
             if info.get('mastery_count', 0) >= env.cfg.mastery_window),
            len(infos)
        )

        feasible_rate = compute_safety_stats(infos).feasible_rate

        results_list.append({
            'scale': scale,
            'eff_val': eff_val,
            'ttm': ttm,
            'feasible_rate': feasible_rate,
        })

        print(f"  Scale {scale:+.2f}: eff={eff_val:.4f}, TTM={ttm:3.0f}, feasible_rate={feasible_rate:.4f}")

    # Compute the range of observed changes.
    ttms = [r['ttm'] for r in results_list]
    feasibles = [r['feasible_rate'] for r in results_list]

    ttm_range = float(np.max(ttms) - np.min(ttms)) if ttms else 0.0
    feasible_range = float(np.max(feasibles) - np.min(feasibles)) if feasibles else 0.0

    # Sensitivity decision.
    is_sensitive = ttm_range > 5.0 or feasible_range > 0.05
    sensitivity_verdict = "Parameter is sensitive to learning dynamics" if is_sensitive else "Parameter sensitivity is insufficient"

    print(f"\n  [Diagnosis]")
    print(f"    TTM range: {min(ttms):.0f} ~ {max(ttms):.0f} (Δ={ttm_range:.0f})")
    print(f"    feasible_rate range: {min(feasibles):.4f} ~ {max(feasibles):.4f} (Δ={feasible_range:.4f})")
    print(f"    {sensitivity_verdict}")

    return {
        'param_name': param_name,
        'scales': scales,
        'results': results_list,
        'ttm_range': ttm_range,
        'feasible_range': feasible_range,
        'is_sensitive': is_sensitive,
    }


def run_dynamics_mismatch(lock_baseline: bool = True) -> Dict[str, object]:
    """
    Main procedure: scan the parameter space and collect sensitivity data.

    Parameters:
    - lock_baseline: if True, force a unified baseline consistent with A1/A2 defaults

    Note:
    The 'baseline' stored in PARAM_SWEEPS is only used to define the sweep space.
    At runtime we instead use the A1/A2 baseline (gamma_f=0.4, eta_forget=0.02,
    kappa=5.0), then apply relative perturbations around that real baseline.
    The configured bounds in PARAM_SWEEPS are still enforced.

    Output format:
    - param_name, scale, agent, metric, value, seed
    """
    ensure_data_dirs()

    # Key fix: use the A1/A2 baseline rather than the nominal sweep baseline.
    # v2 also adjusts difficulty and constraints to magnify sensitivity.
    if lock_baseline:
        from configs import cfg
        # Actual A1/A2 default values, used to keep all scans aligned to one baseline.
        cfg.eta_forget = 0.02          # Project default
        cfg.gamma_f = 0.4              # Project default
        cfg.kappa = 5.0                # Project default
        # v2.1: lower the fatigue threshold further to activate the constraint mechanism.
        cfg.fatigue_threshold = 0.42   # Lowered threshold to trigger the constraint
        cfg.mastery_window = 5         # Stronger mastery requirement to amplify slowdown effects
        cfg.max_steps = 600            # More steps so parameter effects have time to emerge

    print("\n" + "=" * 100)
    print("A5: Dynamics misspecification robustness (Tornado-plot sensitivity battery)")
    print("=" * 100)

    # Collect all results.
    all_results = []

    for param_name in PARAM_SWEEPS.keys():
        print(f"\n{'=' * 100}")
        print(f"Parameter: {param_name}")
        print(f"{'=' * 100}")

        sweep_info = PARAM_SWEEPS[param_name]
        scales = sweep_info['scales']

        for agent_name in AGENTS:
            print(f"\n  Agent: {agent_name}")

            # Verification print: make sure the parameter truly changes.
            # Print the scan range only for the first agent.
            if agent_name == AGENTS[0]:
                # Create a temporary environment to verify the parameter range.
                env_test = PianoGymEnv(seed=1)
                env_test.reset(profile=PROFILE)
                _set_all_to_baseline(env_test)

                eff_vals = []
                for s in scales:
                    _set_all_to_baseline(env_test)  # Reset before each attempt.
                    base_tmp, eff_tmp = _apply_param_perturbation(env_test, param_name, s)
                    eff_vals.append(eff_tmp)

                unique_effs = len(set(f"{e:.6g}" for e in eff_vals))
                print(f"    [verify] {param_name}: base={base_tmp:.6g}, eff_range={min(eff_vals):.6g}~{max(eff_vals):.6g}, distinct_values={unique_effs}/{len(scales)}")
                if unique_effs == 1:
                    print("    Warning: all scales were clipped to the same value. Check whether the bounds are too tight.")

            for scale in scales:
                print(f"    Scale {scale:+.2f}:", end=' ', flush=True)

                for seed_idx in range(NUM_SEEDS):
                    seed = seed_idx * SEED_OFFSET + 1

                    try:
                        result = _run_one_config(agent_name, param_name, scale, seed)
                        all_results.append(result)
                    except Exception as e:
                        print(f"\n      [ERROR] {agent_name} | {param_name}={scale:+.2f} | seed={seed}: {e!r}")
                        continue

                print("✓")

    # Convert to Tornado-plot row format:
    # param_name, scale, agent, metric, value, seed
    tornado_rows = []

    for result in all_results:
        param_name = result['param_name']
        scale = result['scale']
        agent = result['agent']
        seed = result['seed']
        profile = result.get('profile', 'unknown')
        base_val = result.get(f'{param_name}_base')
        eff_val = result.get(f'{param_name}_eff')

        for metric in METRICS:
            value = result[metric]
            tornado_rows.append({
                'param_name': param_name,
                'scale': scale,
                'agent': agent,
                'metric': metric,
                'value': value,
                'seed': seed,
                'profile': profile,
                f'{param_name}_base': base_val,
                f'{param_name}_eff': eff_val,
            })

    # Output data.
    data_dir = "p1_misspec"

    # Detailed per-seed data.
    tornado_csv = resolve_data_path(data_dir, "dynamics_mismatch_per_seed.csv")
    df_tornado = pd.DataFrame(tornado_rows)
    df_tornado.to_csv(tornado_csv, index=False, na_rep="NA")
    print(f"\n✓ Tornado plot data: {tornado_csv}")

    # Summary data: mean ± std for each param × scale × agent × metric slice.
    summary_rows = []
    for (param_name, scale, agent, metric), group in df_tornado.groupby(
        ['param_name', 'scale', 'agent', 'metric']
    ):
        values = group['value'].values
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if metric == 'learn_slope':
            mean_val = round(mean_val, 4)
            std_val = round(std_val, 4)
        summary_rows.append({
            'param_name': param_name,
            'scale': scale,
            'agent': agent,
            'metric': metric,
            'value_mean': mean_val,
            'value_std': std_val,
            'n_seeds': len(values),
        })

    summary_csv = resolve_data_path(data_dir, "dynamics_mismatch_summary.csv")
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(summary_csv, index=False, na_rep="NA")
    print(f"✓ Summary CSV: {summary_csv}")

    # Parameter sweep configuration for visualization annotations.
    config_json = resolve_data_path(data_dir, "param_sweep_config.json")
    sweep_dump = {}
    for pname, sweep in PARAM_SWEEPS.items():
        sweep_dump[pname] = dict(sweep)
        if pname in BASELINE:
            sweep_dump[pname]["baseline"] = BASELINE[pname]
    with open(config_json, 'w') as f:
        json.dump(sweep_dump, f, indent=2, default=str)
    print(f"✓ Parameter config: {config_json}")

    print(f"\n{'=' * 100}")
    print("A5: Dynamics misspecification robustness - data collection complete")
    print(f"{'=' * 100}\n")

    return {
        'tornado': tornado_rows,
        'summary': summary_rows,
        'param_config': PARAM_SWEEPS,
    }


if __name__ == "__main__":
    # ===== Smoke tests: diagnose parameter sensitivity =====
    print("\n" + "=" * 100)
    print("SMOKE TESTS: diagnose environment sensitivity to parameters")
    print("=" * 100)

    smoke_results_gamma_f = smoke_test_env_sensitivity('gamma_f')
    smoke_results_kappa = smoke_test_env_sensitivity('kappa')

    print("\n" + "=" * 100)
    print("SMOKE TEST summary")
    print("=" * 100)
    print(f"\n[gamma_f sensitivity]")
    print(f"  TTM range: Δ={smoke_results_gamma_f['ttm_range']:.0f}")
    print(f"  feasible_rate range: Δ={smoke_results_gamma_f['feasible_range']:.4f}")
    print(f"  sensitive: {smoke_results_gamma_f['is_sensitive']}")

    print(f"\n[kappa sensitivity]")
    print(f"  TTM range: Δ={smoke_results_kappa['ttm_range']:.0f}")
    print(f"  feasible_rate range: Δ={smoke_results_kappa['feasible_range']:.4f}")
    print(f"  sensitive: {smoke_results_kappa['is_sensitive']}")

    print("\n" + "=" * 100)
    print("Starting the main experiment: dynamics misspecification parameter sweep")
    print("=" * 100)

    # ===== Main experiment =====
    run_dynamics_mismatch()
