import time
import numpy as np


# ============================================================
# BASE CASE SETTINGS
# ============================================================

# Base case: n = 3 risky assets, T = 3
N_RISKY = 3
T = 3
N_ASSETS = N_RISKY + 1

RISK_FREE_R = 0.02
ALPHA = 2.0
W0 = 1.0

# Faster grid for quick testing
WEIGHT_STEP = 0.05
WEALTH_MIN = 0.50
WEALTH_MAX = 2.00
WEALTH_STEP = 0.10

ACTION_STEP_SMALL = 0.05
ACTION_STEP_BIG = 0.10
MAX_TURNOVER = 0.10

RETURN_FLOOR = -0.99

# Risky asset parameters
MEANS = np.array([0.08, 0.10, 0.12], dtype=float)
VARIANCES = np.array([0.04, 0.09, 0.16], dtype=float)
STDS = np.sqrt(VARIANCES)

# Equal-weight initial portfolio: cash + 3 risky = 4 assets
P_INIT = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)

# Faster learning settings
N_EPISODES = 5000
LEARNING_RATE = 0.05
EPSILON_START = 0.50
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
TRAIN_SEED = 42

# Convergence logging
CONVERGENCE_LOG_INTERVAL = 500

# Faster test settings
EVAL_EPISODES = 500
FEASIBILITY_TESTS = 100
BELLMAN_MC_SAMPLES = 5

# No DP in base case
RUN_DP_TEST = False


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def cara_utility(wealth, alpha=ALPHA):
    return (1.0 - np.exp(-alpha * wealth)) / alpha


def validate_inputs():
    assert N_RISKY == len(MEANS), "MEANS length must equal N_RISKY"
    assert N_RISKY == len(VARIANCES), "VARIANCES length must equal N_RISKY"
    assert len(P_INIT) == N_ASSETS, "P_INIT length must equal N_ASSETS"
    assert abs(np.sum(P_INIT) - 1.0) < 1e-10, "P_INIT must sum to 1"
    assert np.all(P_INIT >= 0), "P_INIT must be nonnegative"


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
    actions = [np.zeros(num_assets, dtype=float)]  # do nothing

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


WEALTH_GRID = np.arange(WEALTH_MIN, WEALTH_MAX + 1e-12, WEALTH_STEP)
WEIGHT_GRID = generate_weight_grid(N_ASSETS, WEIGHT_STEP)
ACTION_LIST = generate_action_list(N_ASSETS)


# ============================================================
# STATE / ACTION HELPERS
# ============================================================

def nearest_wealth_index(w):
    w_clipped = min(max(w, WEALTH_MIN), WEALTH_MAX)
    idx = int(round((w_clipped - WEALTH_MIN) / WEALTH_STEP))
    return max(0, min(idx, len(WEALTH_GRID) - 1))


def nearest_weight_index(w_vec):
    dists = [np.sum((w_vec - g) ** 2) for g in WEIGHT_GRID]
    return int(np.argmin(dists))


def discretize_state(wealth, weights):
    return nearest_wealth_index(wealth), nearest_weight_index(weights)


def feasible_action_indices(current_weights):
    feasible = []
    for a_idx, u in enumerate(ACTION_LIST):
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


def check_turnover(action_idx, tol=1e-8):
    u = ACTION_LIST[action_idx]
    turnover = 0.5 * np.sum(np.abs(u))
    return turnover <= MAX_TURNOVER + tol


# ============================================================
# ENVIRONMENT
# ============================================================

def sample_risky_returns(rng):
    r = rng.normal(loc=MEANS, scale=STDS)
    r = np.maximum(r, RETURN_FLOOR)
    return r


def step_environment(wealth, weights, action_idx, rng):
    u = ACTION_LIST[action_idx]
    post_trade_weights = weights + u

    risky_returns = sample_risky_returns(rng)

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


def best_action_for_state(Q, t, wealth, weights):
    wealth_idx, weight_idx = discretize_state(wealth, weights)
    feasible = feasible_action_indices(weights)
    qvals = Q[t, wealth_idx, weight_idx, feasible]
    best_local = int(np.argmax(qvals))
    return int(feasible[best_local])


def train_q_learning():
    rng = np.random.default_rng(TRAIN_SEED)

    q_shape = (T, len(WEALTH_GRID), len(WEIGHT_GRID), len(ACTION_LIST))
    Q = np.zeros(q_shape, dtype=float)

    epsilon = EPSILON_START
    convergence_history = []

    running_update_sum = 0.0
    running_td_error_sum = 0.0
    running_steps = 0

    for episode in range(N_EPISODES):
        wealth = W0
        weights = P_INIT.copy()

        for t in range(T):
            wealth_idx, weight_idx = discretize_state(wealth, weights)
            feasible = feasible_action_indices(weights)

            action_idx = epsilon_greedy_action(
                Q[t], wealth_idx, weight_idx, feasible, epsilon, rng
            )

            next_wealth, next_weights = step_environment(wealth, weights, action_idx, rng)

            if t == T - 1:
                td_target = cara_utility(next_wealth)
            else:
                next_wealth_idx, next_weight_idx = discretize_state(next_wealth, next_weights)
                next_feasible = feasible_action_indices(next_weights)
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

        if (episode + 1) % CONVERGENCE_LOG_INTERVAL == 0:
            avg_q_update = running_update_sum / max(running_steps, 1)
            avg_td_error = running_td_error_sum / max(running_steps, 1)

            convergence_history.append({
                "episode": episode + 1,
                "avg_q_update": avg_q_update,
                "avg_td_error": avg_td_error,
                "epsilon": epsilon,
            })

            print(
                f"Episode {episode + 1}/{N_EPISODES} | "
                f"avg_q_update={avg_q_update:.6f} | "
                f"avg_td_error={avg_td_error:.6f} | "
                f"epsilon={epsilon:.4f}"
            )

            running_update_sum = 0.0
            running_td_error_sum = 0.0
            running_steps = 0

    return Q, convergence_history


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy(Q, n_eval=EVAL_EPISODES, seed=999):
    rng = np.random.default_rng(seed)

    terminal_wealths = []
    terminal_utilities = []

    for _ in range(n_eval):
        wealth = W0
        weights = P_INIT.copy()

        for t in range(T):
            action_idx = best_action_for_state(Q, t, wealth, weights)
            wealth, weights = step_environment(wealth, weights, action_idx, rng)

        terminal_wealths.append(wealth)
        terminal_utilities.append(cara_utility(wealth))

    return {
        "avg_terminal_wealth": float(np.mean(terminal_wealths)),
        "std_terminal_wealth": float(np.std(terminal_wealths)),
        "avg_terminal_utility": float(np.mean(terminal_utilities)),
        "min_terminal_wealth": float(np.min(terminal_wealths)),
        "max_terminal_wealth": float(np.max(terminal_wealths)),
    }


# ============================================================
# BENCHMARK POLICIES
# ============================================================

def do_nothing_policy(t, wealth, weights, rng):
    return 0


def random_policy(t, wealth, weights, rng):
    feasible = feasible_action_indices(weights)
    return int(rng.choice(feasible))


def greedy_mean_policy(t, wealth, weights, rng):
    best_risky = 1 + int(np.argmax(MEANS))
    feasible = feasible_action_indices(weights)

    candidate = None
    candidate_amt = -1.0
    for a_idx in feasible:
        u = ACTION_LIST[a_idx]
        if u[best_risky] > candidate_amt:
            candidate = a_idx
            candidate_amt = u[best_risky]

    return int(candidate if candidate is not None else 0)


def evaluate_named_policy(policy_fn, n_eval=EVAL_EPISODES, seed=999):
    rng = np.random.default_rng(seed)

    terminal_wealths = []
    terminal_utilities = []

    for _ in range(n_eval):
        wealth = W0
        weights = P_INIT.copy()

        for t in range(T):
            action_idx = policy_fn(t, wealth, weights, rng)
            wealth, weights = step_environment(wealth, weights, action_idx, rng)

        terminal_wealths.append(wealth)
        terminal_utilities.append(cara_utility(wealth))

    return {
        "avg_terminal_wealth": float(np.mean(terminal_wealths)),
        "std_terminal_wealth": float(np.std(terminal_wealths)),
        "avg_terminal_utility": float(np.mean(terminal_utilities)),
    }


def compare_against_benchmarks(Q, tolerance=1e-3):
    learned_results = evaluate_policy(Q)
    do_nothing_results = evaluate_named_policy(do_nothing_policy)
    random_results = evaluate_named_policy(random_policy)
    greedy_results = evaluate_named_policy(greedy_mean_policy)

    learned_u = learned_results["avg_terminal_utility"]
    do_nothing_u = do_nothing_results["avg_terminal_utility"]
    random_u = random_results["avg_terminal_utility"]
    greedy_u = greedy_results["avg_terminal_utility"]

    return {
        "learned_avg_utility": learned_u,
        "do_nothing_avg_utility": do_nothing_u,
        "random_avg_utility": random_u,
        "greedy_mean_avg_utility": greedy_u,
        "learned_minus_do_nothing": learned_u - do_nothing_u,
        "learned_minus_random": learned_u - random_u,
        "learned_minus_greedy_mean": learned_u - greedy_u,
        "beats_do_nothing": learned_u + tolerance >= do_nothing_u,
        "beats_random": learned_u + tolerance >= random_u,
    }


# ============================================================
# TESTS
# ============================================================

def test_policy_feasibility(Q, n_tests=FEASIBILITY_TESTS, seed=1234):
    rng = np.random.default_rng(seed)

    for _ in range(n_tests):
        wealth = rng.uniform(WEALTH_GRID[0], WEALTH_GRID[-1])
        weight_idx = int(rng.integers(0, len(WEIGHT_GRID)))
        weights = WEIGHT_GRID[weight_idx].copy()

        for t in range(T):
            a = best_action_for_state(Q, t, wealth, weights)

            if not check_turnover(a):
                return False, f"Turnover violated at t={t}"

            post_trade = weights + ACTION_LIST[a]
            if not check_weights_valid(post_trade):
                return False, f"Invalid post-trade weights at t={t}"

            next_wealth, next_weights = step_environment(wealth, weights, a, rng)

            if next_wealth <= 0:
                return False, f"Nonpositive wealth at t={t}"

            if not check_weights_valid(next_weights):
                return False, f"Invalid next weights at t={t}"

            wealth, weights = next_wealth, next_weights

    return True, "passed"


def bellman_residual_test(Q, mc_samples=BELLMAN_MC_SAMPLES, seed=77):
    rng = np.random.default_rng(seed)
    residuals = []

    for t in range(T):
        for wealth_idx in range(len(WEALTH_GRID)):
            wealth = WEALTH_GRID[wealth_idx]

            sample_weight_indices = np.linspace(
                0, len(WEIGHT_GRID) - 1, min(10, len(WEIGHT_GRID)), dtype=int
            )

            for weight_idx in sample_weight_indices:
                weights = WEIGHT_GRID[weight_idx]
                feasible = feasible_action_indices(weights)

                for a_idx in feasible[:min(5, len(feasible))]:
                    targets = []

                    for _ in range(mc_samples):
                        next_wealth, next_weights = step_environment(
                            wealth, weights, a_idx, rng
                        )

                        if t == T - 1:
                            targets.append(cara_utility(next_wealth))
                        else:
                            nw_idx, nwt_idx = discretize_state(next_wealth, next_weights)
                            next_feasible = feasible_action_indices(next_weights)
                            targets.append(np.max(Q[t + 1, nw_idx, nwt_idx, next_feasible]))

                    bellman_target = np.mean(targets)
                    q_val = Q[t, wealth_idx, weight_idx, a_idx]
                    residuals.append(abs(q_val - bellman_target))

    return {
        "avg_residual": float(np.mean(residuals)),
        "max_residual": float(np.max(residuals)),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    validate_inputs()

    print("Base Case Q-Learning Run")
    print("=" * 60)
    print(f"N_RISKY = {N_RISKY}")
    print(f"T = {T}")
    print(f"Initial portfolio = {P_INIT}")
    print(f"WEIGHT_STEP = {WEIGHT_STEP}")
    print(f"N_EPISODES = {N_EPISODES}")
    print("=" * 60)
    print(f"Number of weight states: {len(WEIGHT_GRID)}")
    print(f"Number of wealth states: {len(WEALTH_GRID)}")
    print(f"Number of actions: {len(ACTION_LIST)}")
    print("=" * 60)

    t0 = time.time()

    Q, convergence_history = train_q_learning()

    eval_stats = evaluate_policy(Q)
    feas_pass, feas_msg = test_policy_feasibility(Q)
    benchmark_stats = compare_against_benchmarks(Q)
    bellman_stats = bellman_residual_test(Q)

    elapsed = time.time() - t0

    final_avg_q_update = np.nan
    final_avg_td_error = np.nan
    if convergence_history:
        final_avg_q_update = convergence_history[-1]["avg_q_update"]
        final_avg_td_error = convergence_history[-1]["avg_td_error"]

    print("\nResults")
    print("=" * 60)
    print(f"avg_terminal_wealth   = {eval_stats['avg_terminal_wealth']:.6f}")
    print(f"std_terminal_wealth   = {eval_stats['std_terminal_wealth']:.6f}")
    print(f"avg_terminal_utility  = {eval_stats['avg_terminal_utility']:.6f}")
    print(f"min_terminal_wealth   = {eval_stats['min_terminal_wealth']:.6f}")
    print(f"max_terminal_wealth   = {eval_stats['max_terminal_wealth']:.6f}")

    print("\nTests")
    print("=" * 60)
    print(f"feasibility_pass      = {feas_pass}")
    print(f"feasibility_message   = {feas_msg}")
    print(f"beats_do_nothing      = {benchmark_stats['beats_do_nothing']}")
    print(f"beats_random          = {benchmark_stats['beats_random']}")
    print(f"learned-do_nothing    = {benchmark_stats['learned_minus_do_nothing']:.6f}")
    print(f"learned-random        = {benchmark_stats['learned_minus_random']:.6f}")
    print(f"learned-greedy_mean   = {benchmark_stats['learned_minus_greedy_mean']:.6f}")
    print(f"bellman_avg_residual  = {bellman_stats['avg_residual']:.6f}")
    print(f"bellman_max_residual  = {bellman_stats['max_residual']:.6f}")

    print("\nConvergence")
    print("=" * 60)
    print(f"final_avg_q_update    = {final_avg_q_update:.6f}")
    print(f"final_avg_td_error    = {final_avg_td_error:.6f}")

    print("\nRuntime")
    print("=" * 60)
    print(f"runtime_seconds       = {elapsed:.2f}")


if __name__ == "__main__":
    main()