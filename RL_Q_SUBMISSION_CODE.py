import math
import time
import numpy as np
import pandas as pd


# ============================================================
# GLOBAL SETTINGS
# ============================================================

RISK_FREE_R = 0.02
ALPHA = 2.0
W0 = 1.0

# step size requested
WEIGHT_STEP = 0.05
WEALTH_MIN = 0.50
WEALTH_MAX = 2.00
WEALTH_STEP = 0.05

ACTION_STEP_SMALL = 0.05
ACTION_STEP_BIG = 0.10
MAX_TURNOVER = 0.10

RETURN_FLOOR = -0.99

# learning settings
N_EPISODES = 30000
LEARNING_RATE = 0.05
EPSILON_START = 0.60
EPSILON_END = 0.05
EPSILON_DECAY = 0.9999
N_TRAIN_RUNS = 3
BASE_TRAIN_SEED = 42

# convergence logging
CONVERGENCE_LOG_INTERVAL = 1000

# evaluation settings
EVAL_EPISODES = 2000
FEASIBILITY_TESTS = 300
BELLMAN_MC_SAMPLES = 20

# DP is expensive with step=0.05, especially for n=4
RUN_DP_TEST = False
DP_MC_SAMPLES = 20
DP_ALLOWED_CASES = {(3, 1), (3, 2), (3, 3)}   # only used if RUN_DP_TEST = True

# cases to run
N_VALUES = [3, 4]
T_VALUES = list(range(1, 10))

# parameters for n=3 and n=4
PARAMS_BY_N = {
    3: {
        "means": np.array([0.08, 0.10, 0.12], dtype=float),
        "variances": np.array([0.04, 0.09, 0.16], dtype=float),
    },
    4: {
        "means": np.array([0.08, 0.10, 0.12, 0.09], dtype=float),
        "variances": np.array([0.04, 0.09, 0.16, 0.05], dtype=float),
    },
}


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def cara_utility(wealth, alpha=ALPHA):
    return (1.0 - np.exp(-alpha * wealth)) / alpha


def equal_weight_portfolio(n_risky):
    n_assets = n_risky + 1
    return np.ones(n_assets, dtype=float) / n_assets


def validate_inputs(T, n_risky, means, variances, p_init):
    n_assets = n_risky + 1
    assert n_risky in [3, 4], "n_risky must be 3 or 4"
    assert 1 <= T <= 9, "T must be between 1 and 9"
    assert len(means) == n_risky, "means length must equal n_risky"
    assert len(variances) == n_risky, "variances length must equal n_risky"
    assert len(p_init) == n_assets, "p_init length must equal n_assets"
    assert abs(np.sum(p_init) - 1.0) < 1e-10, "p_init must sum to 1"
    assert np.all(p_init >= 0), "p_init must be nonnegative"


# ============================================================
# GRID CONSTRUCTION
# ============================================================

def generate_weight_grid(num_assets, step):
    m = int(round(1.0 / step))
    states = []

    def rec_build(prefix, remaining, k_left):
        if k_left == 1:
            states.append(tuple(prefix + [remaining / m]))
            return
        for x in range(remaining + 1):
            rec_build(prefix + [x / m], remaining - x, k_left - 1)

    rec_build([], m, num_assets)
    return [np.array(s, dtype=float) for s in states]


def generate_action_list(num_assets):
    actions = [np.zeros(num_assets, dtype=float)]

    for amt in [ACTION_STEP_SMALL, ACTION_STEP_BIG]:
        for i in range(num_assets):
            for j in range(num_assets):
                if i == j:
                    continue
                u = np.zeros(num_assets, dtype=float)
                u[i] = -amt
                u[j] = amt
                actions.append(u)

    return actions


# ============================================================
# STATE / ACTION HELPERS
# ============================================================

def nearest_wealth_index(w, wealth_grid):
    wealth_min = wealth_grid[0]
    wealth_max = wealth_grid[-1]
    wealth_step = wealth_grid[1] - wealth_grid[0]

    w_clipped = min(max(w, wealth_min), wealth_max)
    idx = int(round((w_clipped - wealth_min) / wealth_step))
    return max(0, min(idx, len(wealth_grid) - 1))


def nearest_weight_index(w_vec, weight_grid):
    dists = [np.sum((w_vec - g) ** 2) for g in weight_grid]
    return int(np.argmin(dists))


def discretize_state(wealth, weights, wealth_grid, weight_grid):
    return (
        nearest_wealth_index(wealth, wealth_grid),
        nearest_weight_index(weights, weight_grid),
    )


def feasible_action_indices(current_weights, action_list):
    feasible = []
    for a_idx, u in enumerate(action_list):
        new_w = current_weights + u

        if np.any(new_w < -1e-12):
            continue
        if np.any(new_w > 1.0 + 1e-12):
            continue
        if abs(np.sum(new_w) - 1.0) > 1e-10:
            continue

        turnover = 0.5 * np.sum(np.abs(u))
        if turnover <= MAX_TURNOVER + 1e-12:
            feasible.append(a_idx)

    return feasible


def check_weights_valid(w, tol=1e-8):
    return np.all(w >= -tol) and abs(np.sum(w) - 1.0) <= tol


def check_turnover(action_idx, action_list, tol=1e-8):
    u = action_list[action_idx]
    turnover = 0.5 * np.sum(np.abs(u))
    return turnover <= MAX_TURNOVER + tol


# ============================================================
# ENVIRONMENT
# ============================================================

def sample_risky_returns(rng, means, stds):
    r = rng.normal(loc=means, scale=stds)
    r = np.maximum(r, RETURN_FLOOR)
    return r


def step_environment(wealth, weights, action_idx, action_list, rng, means, stds):
    u = action_list[action_idx]
    post_trade_weights = weights + u

    risky_returns = sample_risky_returns(rng, means, stds)

    gross_cash = post_trade_weights[0] * (1.0 + RISK_FREE_R)
    gross_risky = post_trade_weights[1:] * (1.0 + risky_returns)
    gross_total = gross_cash + np.sum(gross_risky)
    gross_total = max(gross_total, 1e-12)

    next_wealth = wealth * gross_total

    next_weights = np.zeros_like(weights)
    next_weights[0] = gross_cash / gross_total
    next_weights[1:] = gross_risky / gross_total

    next_weights = np.maximum(next_weights, 0.0)
    s = np.sum(next_weights)
    if s <= 0:
        next_weights = np.zeros_like(weights)
        next_weights[0] = 1.0
    else:
        next_weights = next_weights / s

    return next_wealth, next_weights


# ============================================================
# Q-LEARNING
# ============================================================

def epsilon_greedy_action(Q_t, wealth_idx, weight_idx, feasible_idxs, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.choice(feasible_idxs))

    qvals = Q_t[wealth_idx, weight_idx, feasible_idxs]
    best_local = int(np.argmax(qvals))
    return int(feasible_idxs[best_local])


def best_action_for_state(Q, t, wealth, weights, wealth_grid, weight_grid, action_list):
    wealth_idx, weight_idx = discretize_state(wealth, weights, wealth_grid, weight_grid)
    feasible = feasible_action_indices(weights, action_list)
    qvals = Q[t, wealth_idx, weight_idx, feasible]
    best_local = int(np.argmax(qvals))
    return int(feasible[best_local])


def train_q_learning_with_seed(
    T,
    p_init,
    means,
    stds,
    wealth_grid,
    weight_grid,
    action_list,
    seed,
    log_interval=CONVERGENCE_LOG_INTERVAL
):
    rng = np.random.default_rng(seed)

    q_shape = (T, len(wealth_grid), len(weight_grid), len(action_list))
    Q = np.zeros(q_shape, dtype=float)

    epsilon = EPSILON_START
    convergence_history = []

    running_update_sum = 0.0
    running_td_error_sum = 0.0
    running_steps = 0

    for episode in range(N_EPISODES):
        wealth = W0
        weights = p_init.copy()

        for t in range(T):
            wealth_idx, weight_idx = discretize_state(wealth, weights, wealth_grid, weight_grid)
            feasible = feasible_action_indices(weights, action_list)

            action_idx = epsilon_greedy_action(
                Q[t], wealth_idx, weight_idx, feasible, epsilon, rng
            )

            next_wealth, next_weights = step_environment(
                wealth, weights, action_idx, action_list, rng, means, stds
            )

            if t == T - 1:
                td_target = cara_utility(next_wealth)
            else:
                next_wealth_idx, next_weight_idx = discretize_state(
                    next_wealth, next_weights, wealth_grid, weight_grid
                )
                next_feasible = feasible_action_indices(next_weights, action_list)
                td_target = np.max(Q[t + 1, next_wealth_idx, next_weight_idx, next_feasible])

            old_q = Q[t, wealth_idx, weight_idx, action_idx]
            td_error = td_target - old_q

            new_q = (1.0 - LEARNING_RATE) * old_q + LEARNING_RATE * td_target
            q_update_size = abs(new_q - old_q)

            Q[t, wealth_idx, weight_idx, action_idx] = new_q

            running_update_sum += q_update_size
            running_td_error_sum += abs(td_error)
            running_steps += 1

            wealth, weights = next_wealth, next_weights

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (episode + 1) % log_interval == 0:
            avg_q_update = running_update_sum / max(running_steps, 1)
            avg_td_error = running_td_error_sum / max(running_steps, 1)

            convergence_history.append({
                "episode": episode + 1,
                "avg_q_update": avg_q_update,
                "avg_td_error": avg_td_error,
                "epsilon": epsilon,
            })

            print(
                f"Seed {seed} | Episode {episode + 1}/{N_EPISODES} | "
                f"avg_q_update={avg_q_update:.6f} | "
                f"avg_td_error={avg_td_error:.6f} | "
                f"epsilon={epsilon:.4f}"
            )

            running_update_sum = 0.0
            running_td_error_sum = 0.0
            running_steps = 0

    return Q, convergence_history


def evaluate_policy(
    Q,
    T,
    p_init,
    means,
    stds,
    wealth_grid,
    weight_grid,
    action_list,
    n_eval=EVAL_EPISODES,
    seed=999
):
    rng = np.random.default_rng(seed)

    terminal_wealths = []
    terminal_utilities = []

    for _ in range(n_eval):
        wealth = W0
        weights = p_init.copy()

        for t in range(T):
            action_idx = best_action_for_state(
                Q, t, wealth, weights, wealth_grid, weight_grid, action_list
            )
            wealth, weights = step_environment(
                wealth, weights, action_idx, action_list, rng, means, stds
            )

        terminal_wealths.append(wealth)
        terminal_utilities.append(cara_utility(wealth))

    return {
        "avg_terminal_wealth": float(np.mean(terminal_wealths)),
        "std_terminal_wealth": float(np.std(terminal_wealths)),
        "avg_terminal_utility": float(np.mean(terminal_utilities)),
        "min_terminal_wealth": float(np.min(terminal_wealths)),
        "max_terminal_wealth": float(np.max(terminal_wealths)),
    }


def train_best_of_n_runs(
    T,
    p_init,
    means,
    stds,
    wealth_grid,
    weight_grid,
    action_list,
    n_runs=N_TRAIN_RUNS
):
    best_Q = None
    best_score = -np.inf
    best_history = None

    for run in range(n_runs):
        seed = BASE_TRAIN_SEED + run
        Q, history = train_q_learning_with_seed(
            T=T,
            p_init=p_init,
            means=means,
            stds=stds,
            wealth_grid=wealth_grid,
            weight_grid=weight_grid,
            action_list=action_list,
            seed=seed,
        )

        stats = evaluate_policy(
            Q=Q,
            T=T,
            p_init=p_init,
            means=means,
            stds=stds,
            wealth_grid=wealth_grid,
            weight_grid=weight_grid,
            action_list=action_list,
            n_eval=800,
            seed=1000 + run,
        )

        score = stats["avg_terminal_utility"]
        if score > best_score:
            best_score = score
            best_Q = Q.copy()
            best_history = history

    return best_Q, best_score, best_history


# ============================================================
# BENCHMARK POLICIES
# ============================================================

def do_nothing_policy(t, wealth, weights, rng, action_list, means):
    return 0


def random_policy(t, wealth, weights, rng, action_list, means):
    feasible = feasible_action_indices(weights, action_list)
    return int(rng.choice(feasible))


def greedy_mean_policy(t, wealth, weights, rng, action_list, means):
    best_risky = 1 + int(np.argmax(means))
    feasible = feasible_action_indices(weights, action_list)

    candidate = None
    candidate_amt = -1.0
    for a_idx in feasible:
        u = action_list[a_idx]
        if u[best_risky] > candidate_amt:
            candidate = a_idx
            candidate_amt = u[best_risky]

    return int(candidate if candidate is not None else 0)


def learned_policy_factory(Q, T, wealth_grid, weight_grid, action_list):
    def learned_policy(t, wealth, weights, rng, _action_list=None, means=None):
        return best_action_for_state(Q, t, wealth, weights, wealth_grid, weight_grid, action_list)
    return learned_policy


def evaluate_named_policy(
    policy_fn,
    T,
    p_init,
    means,
    stds,
    action_list,
    n_eval=EVAL_EPISODES,
    seed=999
):
    rng = np.random.default_rng(seed)

    terminal_wealths = []
    terminal_utilities = []

    for _ in range(n_eval):
        wealth = W0
        weights = p_init.copy()

        for t in range(T):
            action_idx = policy_fn(t, wealth, weights, rng, action_list, means)
            wealth, weights = step_environment(
                wealth, weights, action_idx, action_list, rng, means, stds
            )

        terminal_wealths.append(wealth)
        terminal_utilities.append(cara_utility(wealth))

    return {
        "avg_terminal_wealth": float(np.mean(terminal_wealths)),
        "std_terminal_wealth": float(np.std(terminal_wealths)),
        "avg_terminal_utility": float(np.mean(terminal_utilities)),
    }


def compare_against_benchmarks(
    Q,
    T,
    p_init,
    means,
    stds,
    wealth_grid,
    weight_grid,
    action_list,
    tolerance=1e-3,
):
    learned = learned_policy_factory(Q, T, wealth_grid, weight_grid, action_list)

    results = {
        "learned_policy": evaluate_named_policy(learned, T, p_init, means, stds, action_list),
        "do_nothing": evaluate_named_policy(do_nothing_policy, T, p_init, means, stds, action_list),
        "random_policy": evaluate_named_policy(random_policy, T, p_init, means, stds, action_list),
        "greedy_mean_policy": evaluate_named_policy(greedy_mean_policy, T, p_init, means, stds, action_list),
    }

    learned_u = results["learned_policy"]["avg_terminal_utility"]
    do_nothing_u = results["do_nothing"]["avg_terminal_utility"]
    random_u = results["random_policy"]["avg_terminal_utility"]
    greedy_u = results["greedy_mean_policy"]["avg_terminal_utility"]

    summary = {
        "learned_minus_do_nothing": learned_u - do_nothing_u,
        "learned_minus_random": learned_u - random_u,
        "learned_minus_greedy_mean": learned_u - greedy_u,
        "beats_do_nothing": learned_u + tolerance >= do_nothing_u,
        "beats_random": learned_u + tolerance >= random_u,
    }

    return results, summary


# ============================================================
# TESTS
# ============================================================

def test_policy_feasibility(
    Q,
    T,
    means,
    stds,
    wealth_grid,
    weight_grid,
    action_list,
    n_tests=FEASIBILITY_TESTS,
    seed=1234
):
    rng = np.random.default_rng(seed)

    for _ in range(n_tests):
        wealth = rng.uniform(wealth_grid[0], wealth_grid[-1])
        weight_idx = int(rng.integers(0, len(weight_grid)))
        weights = weight_grid[weight_idx].copy()

        for t in range(T):
            a = best_action_for_state(Q, t, wealth, weights, wealth_grid, weight_grid, action_list)

            if not check_turnover(a, action_list):
                return False, f"Turnover violated at t={t}"

            post_trade = weights + action_list[a]
            if not check_weights_valid(post_trade):
                return False, f"Invalid post-trade weights at t={t}"

            next_wealth, next_weights = step_environment(
                wealth, weights, a, action_list, rng, means, stds
            )

            if next_wealth <= 0:
                return False, f"Nonpositive wealth at t={t}"

            if not check_weights_valid(next_weights):
                return False, f"Invalid next weights at t={t}"

            wealth, weights = next_wealth, next_weights

    return True, "passed"


def bellman_residual_test(
    Q,
    T,
    means,
    stds,
    wealth_grid,
    weight_grid,
    action_list,
    mc_samples=BELLMAN_MC_SAMPLES,
    seed=77
):
    rng = np.random.default_rng(seed)
    residuals = []

    for t in range(T):
        for wealth_idx in range(len(wealth_grid)):
            wealth = wealth_grid[wealth_idx]

            sample_weight_indices = np.linspace(
                0, len(weight_grid) - 1, min(30, len(weight_grid)), dtype=int
            )

            for weight_idx in sample_weight_indices:
                weights = weight_grid[weight_idx]
                feasible = feasible_action_indices(weights, action_list)

                for a_idx in feasible[:min(8, len(feasible))]:
                    targets = []

                    for _ in range(mc_samples):
                        next_wealth, next_weights = step_environment(
                            wealth, weights, a_idx, action_list, rng, means, stds
                        )

                        if t == T - 1:
                            targets.append(cara_utility(next_wealth))
                        else:
                            nw_idx, nwt_idx = discretize_state(
                                next_wealth, next_weights, wealth_grid, weight_grid
                            )
                            next_feasible = feasible_action_indices(next_weights, action_list)
                            targets.append(np.max(Q[t + 1, nw_idx, nwt_idx, next_feasible]))

                    bellman_target = np.mean(targets)
                    q_val = Q[t, wealth_idx, weight_idx, a_idx]
                    residuals.append(abs(q_val - bellman_target))

    return {
        "avg_residual": float(np.mean(residuals)),
        "max_residual": float(np.max(residuals)),
    }


# ============================================================
# OPTIONAL DP REFERENCE
# ============================================================

def compute_dp_reference(
    T,
    means,
    stds,
    wealth_grid,
    weight_grid,
    action_list,
    mc_samples=DP_MC_SAMPLES,
    seed=2025
):
    rng = np.random.default_rng(seed)

    Q_dp = np.full(
        (T, len(wealth_grid), len(weight_grid), len(action_list)),
        -np.inf,
        dtype=float
    )

    for t in reversed(range(T)):
        for wealth_idx in range(len(wealth_grid)):
            wealth = wealth_grid[wealth_idx]

            sample_weight_indices = np.linspace(
                0, len(weight_grid) - 1, min(60, len(weight_grid)), dtype=int
            )

            for weight_idx in sample_weight_indices:
                weights = weight_grid[weight_idx]
                feasible = feasible_action_indices(weights, action_list)

                for a_idx in feasible:
                    returns = []

                    for _ in range(mc_samples):
                        next_wealth, next_weights = step_environment(
                            wealth, weights, a_idx, action_list, rng, means, stds
                        )

                        if t == T - 1:
                            returns.append(cara_utility(next_wealth))
                        else:
                            nw_idx, nwt_idx = discretize_state(
                                next_wealth, next_weights, wealth_grid, weight_grid
                            )
                            next_feasible = feasible_action_indices(next_weights, action_list)
                            next_val = np.max(Q_dp[t + 1, nw_idx, nwt_idx, next_feasible])
                            returns.append(next_val)

                    Q_dp[t, wealth_idx, weight_idx, a_idx] = np.mean(returns)

    return Q_dp


def compare_qlearning_to_dp(Q_learned, Q_dp, T, weight_grid, action_list, wealth_grid):
    total_states = 0
    matching_actions = 0
    value_gaps = []

    sample_weight_indices = np.linspace(
        0, len(weight_grid) - 1, min(60, len(weight_grid)), dtype=int
    )

    for t in range(T):
        for wealth_idx in range(len(wealth_grid)):
            for weight_idx in sample_weight_indices:
                weights = weight_grid[weight_idx]
                feasible = feasible_action_indices(weights, action_list)

                ql_vals = Q_learned[t, wealth_idx, weight_idx, feasible]
                dp_vals = Q_dp[t, wealth_idx, weight_idx, feasible]

                if np.all(np.isneginf(dp_vals)):
                    continue

                ql_best = feasible[np.argmax(ql_vals)]
                dp_best = feasible[np.argmax(dp_vals)]

                ql_val = float(np.max(ql_vals))
                dp_val = float(np.max(dp_vals))

                total_states += 1
                if ql_best == dp_best:
                    matching_actions += 1

                value_gaps.append(abs(dp_val - ql_val))

    if total_states == 0:
        return {
            "action_match_rate": np.nan,
            "avg_value_gap": np.nan,
            "max_value_gap": np.nan,
        }

    return {
        "action_match_rate": matching_actions / total_states,
        "avg_value_gap": float(np.mean(value_gaps)),
        "max_value_gap": float(np.max(value_gaps)),
    }


# ============================================================
# SINGLE EXPERIMENT
# ============================================================

def run_single_case(n_risky, T):
    params = PARAMS_BY_N[n_risky]
    means = params["means"]
    variances = params["variances"]
    stds = np.sqrt(variances)

    p_init = equal_weight_portfolio(n_risky)
    validate_inputs(T, n_risky, means, variances, p_init)

    n_assets = n_risky + 1

    wealth_grid = np.arange(WEALTH_MIN, WEALTH_MAX + 1e-12, WEALTH_STEP)
    weight_grid = generate_weight_grid(n_assets, WEIGHT_STEP)
    action_list = generate_action_list(n_assets)

    t0 = time.time()

    Q, best_train_score, best_history = train_best_of_n_runs(
        T=T,
        p_init=p_init,
        means=means,
        stds=stds,
        wealth_grid=wealth_grid,
        weight_grid=weight_grid,
        action_list=action_list,
        n_runs=N_TRAIN_RUNS,
    )

    eval_stats = evaluate_policy(
        Q=Q,
        T=T,
        p_init=p_init,
        means=means,
        stds=stds,
        wealth_grid=wealth_grid,
        weight_grid=weight_grid,
        action_list=action_list,
        n_eval=EVAL_EPISODES,
        seed=999,
    )

    feas_pass, feas_msg = test_policy_feasibility(
        Q=Q,
        T=T,
        means=means,
        stds=stds,
        wealth_grid=wealth_grid,
        weight_grid=weight_grid,
        action_list=action_list,
    )

    benchmark_results, benchmark_summary = compare_against_benchmarks(
        Q=Q,
        T=T,
        p_init=p_init,
        means=means,
        stds=stds,
        wealth_grid=wealth_grid,
        weight_grid=weight_grid,
        action_list=action_list,
    )

    bellman_stats = bellman_residual_test(
        Q=Q,
        T=T,
        means=means,
        stds=stds,
        wealth_grid=wealth_grid,
        weight_grid=weight_grid,
        action_list=action_list,
    )

    dp_stats = {
        "action_match_rate": np.nan,
        "avg_value_gap": np.nan,
        "max_value_gap": np.nan,
        "dp_ran": False,
    }

    if RUN_DP_TEST and (n_risky, T) in DP_ALLOWED_CASES:
        Q_dp = compute_dp_reference(
            T=T,
            means=means,
            stds=stds,
            wealth_grid=wealth_grid,
            weight_grid=weight_grid,
            action_list=action_list,
            mc_samples=DP_MC_SAMPLES,
        )
        comp = compare_qlearning_to_dp(
            Q_learned=Q,
            Q_dp=Q_dp,
            T=T,
            weight_grid=weight_grid,
            action_list=action_list,
            wealth_grid=wealth_grid,
        )
        dp_stats.update(comp)
        dp_stats["dp_ran"] = True

    elapsed = time.time() - t0

    final_avg_q_update = np.nan
    final_avg_td_error = np.nan
    if best_history and len(best_history) > 0:
        final_avg_q_update = best_history[-1]["avg_q_update"]
        final_avg_td_error = best_history[-1]["avg_td_error"]

    result = {
        "n_risky": n_risky,
        "T": T,
        "n_assets": n_assets,
        "initial_portfolio": p_init.tolist(),
        "num_weight_states": len(weight_grid),
        "num_wealth_states": len(wealth_grid),
        "num_actions": len(action_list),
        "best_train_score": best_train_score,
        "avg_terminal_wealth": eval_stats["avg_terminal_wealth"],
        "std_terminal_wealth": eval_stats["std_terminal_wealth"],
        "avg_terminal_utility": eval_stats["avg_terminal_utility"],
        "min_terminal_wealth": eval_stats["min_terminal_wealth"],
        "max_terminal_wealth": eval_stats["max_terminal_wealth"],
        "feasibility_pass": feas_pass,
        "feasibility_message": feas_msg,
        "learned_minus_do_nothing": benchmark_summary["learned_minus_do_nothing"],
        "learned_minus_random": benchmark_summary["learned_minus_random"],
        "learned_minus_greedy_mean": benchmark_summary["learned_minus_greedy_mean"],
        "beats_do_nothing": benchmark_summary["beats_do_nothing"],
        "beats_random": benchmark_summary["beats_random"],
        "bellman_avg_residual": bellman_stats["avg_residual"],
        "bellman_max_residual": bellman_stats["max_residual"],
        "dp_ran": dp_stats["dp_ran"],
        "dp_action_match_rate": dp_stats["action_match_rate"],
        "dp_avg_value_gap": dp_stats["avg_value_gap"],
        "dp_max_value_gap": dp_stats["max_value_gap"],
        "final_avg_q_update": final_avg_q_update,
        "final_avg_td_error": final_avg_td_error,
        "runtime_seconds": elapsed,
    }

    return result


# ============================================================
# MAIN GRID RUNNER
# ============================================================

def main():
    all_results = []

    print("Running all cases...")
    print("=" * 80)
    print(f"Weight step = {WEIGHT_STEP}")
    print(f"T values = {T_VALUES}")
    print(f"n values = {N_VALUES}")
    print(f"RUN_DP_TEST = {RUN_DP_TEST}")
    print("=" * 80)

    for n_risky in N_VALUES:
        for T in T_VALUES:
            print(f"\nRunning case: n={n_risky}, T={T}")
            result = run_single_case(n_risky, T)
            all_results.append(result)

            print(f"  avg_terminal_utility = {result['avg_terminal_utility']:.6f}")
            print(f"  feasibility_pass     = {result['feasibility_pass']}")
            print(f"  beats_do_nothing     = {result['beats_do_nothing']}")
            print(f"  bellman_avg_residual = {result['bellman_avg_residual']:.6f}")
            print(f"  final_avg_q_update   = {result['final_avg_q_update']:.6f}")
            print(f"  final_avg_td_error   = {result['final_avg_td_error']:.6f}")
            if result["dp_ran"]:
                print(f"  dp_action_match_rate = {result['dp_action_match_rate']:.4f}")
            print(f"  runtime_seconds      = {result['runtime_seconds']:.2f}")

    df = pd.DataFrame(all_results)

    print("\nFinal summary table")
    print("=" * 80)
    display_cols = [
        "n_risky",
        "T",
        "avg_terminal_utility",
        "avg_terminal_wealth",
        "feasibility_pass",
        "beats_do_nothing",
        "beats_random",
        "bellman_avg_residual",
        "final_avg_q_update",
        "final_avg_td_error",
        "dp_ran",
        "dp_action_match_rate",
        "runtime_seconds",
    ]
    print(df[display_cols].to_string(index=False))

    output_file = "rl_grid_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSaved full results to {output_file}")


if __name__ == "__main__":
    main()