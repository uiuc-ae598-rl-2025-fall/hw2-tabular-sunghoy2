import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# %% MDP Definition

lake_map = ["SFFF", "FHFH", "FFFH", "HFFG"]

def make_env(is_slippery=True):
    """
    Create an environement using Gymnasium package
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=lake_map,
        is_slippery=is_slippery,
        success_rate=1.0/3.0,
        reward_schedule=(1, 0, 0)
    )
    return env

# %%  Controal & Evaluation Helper Functions


# sample action based on epsilon-greedy over Q
def choose_action_e_greedy(Q, s, eps):
    """
    epsilon-greedy of Q to choose an action for s
    """
    q_row = Q[s]
    best_val = np.max(q_row)    # best Q value for the state s
    greedy_acts = np.flatnonzero(np.isclose(q_row, best_val))   # all actions with the best Q value

    # construct probability distribution
    A = Q.shape[1]
    probs = np.full(A, eps / A, dtype=float)    # equal probability for all actions (ensure exploration)
    probs[greedy_acts] += (1.0 - eps) / len(greedy_acts)    # add mass to greedy actions

    # sample action using the probability distribution
    a_sample = int(np.random.choice(np.arange(A), p=probs))

    return a_sample


# roll out one episode under eps-greedy(Q)
def generate_e_greedy_episode_for_train(env, Q, eps, max_steps=200):
    """
    Roll out one episode under eps-greedy(Q)
    """
    S_traj, A_traj, R_traj = [], [], []
    s, _ = env.reset()                    # Gymnasium returns (obs, info)
    s = int(s)
    
    steps = 0
    while True:
        a = choose_action_e_greedy(Q, s, eps)    # ε-greedy action
        s_next, r, terminated, truncated, _ = env.step(a)

        S_traj.append(s)
        A_traj.append(int(a))
        R_traj.append(float(r))

        s = int(s_next)
        steps += 1
        if terminated or truncated or (steps >= max_steps):
            break
    return S_traj, A_traj, R_traj


def generate_greedy_episode_for_eval(env, Q, max_steps=200):
    """
    Generates a single episode with the greedy policy from Q
    (evaluation episode)
    """
    # Initialize
    s0, _ = env.reset()   # reset env to the start state (observation, info)
    s0 = int(s0)          # convert state to int
    G, steps = 0.0, 0
    
    s = s0
    while True:
        # Greey Improvement
        a = int(np.argmax(Q[s]))    # greedy action for a given s
        s, r, terminated, truncated, _ = env.step(a)    # take an action
        G += float(r)               # update the total return
        steps += 1
        
        # Stop the episode
        if terminated or truncated or steps >= max_steps:
            # Current G is the return of this episode
            break
    return G, steps

def eval_greedy_success(env, Q, n_episodes=200):
    """
    Average greedy return over many evaluation episodes
    (success rate on FrozenLake).
    """
    # initialize
    n_success = 0
    
    # Success rate
    for _ in range(n_episodes):
        G, steps_single = generate_greedy_episode_for_eval(env, Q, max_steps=200)
        n_success += (G > 0.0)  # count successful episodes
        
    return n_success / n_episodes





# %% [1] On-policy, First-visit MC


def train_mc_with_eval_steps(
    seed: int,
    is_slippery: bool = True,
    episodes: int = 8000,
    eps: float = 0.10,
    eval_every: int = 500,
    eval_episodes: int = 200,
    max_steps: int = 200,
    gamma: float = 0.95,
):
    """
    On-policy First-Visit Monte Carlo Control with periodic greedy evaluation.
    Uses epsilon-greedy for training rollouts, greedy for evaluation.
    """
    # FrozenLake Setup
    np.random.seed(seed)
    env = make_env(is_slippery=is_slippery)
    nS, nA = env.observation_space.n, env.action_space.n
    # Multi-seed evaluation in Ch2.7 Albrecht
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # Initialize
    Q = np.zeros((nS, nA), dtype=float)
    N = np.zeros_like(Q, dtype=int)       # visit counts for incremental mean
    # For plotting
    eval_scores, eval_steps = [], []
    cum_steps = 0  # track how many steps we’ve taken in training

    # Training for n # of episodes
    for k in range(1, episodes + 1):
        # Generate one e-greedy training episode
        S, A, R = generate_e_greedy_episode_for_train(env, Q, eps, max_steps)
        cum_steps += len(S)

        # First-Visit MC backward pass
        # initialize
        G_tp1 = 0.0
        is_seen = set()
        # loop over timesteps
        for t in range(len(S) - 1, -1, -1):  # traverse trajectory backwards
            s_t, a_t, r_tp1 = S[t], A[t], R[t]
            G_t = r_tp1 + gamma * G_tp1
            if (s_t, a_t) not in is_seen:       # first-visit condition
                is_seen.add((s_t, a_t))
                N[s_t, a_t] += 1
                Q[s_t, a_t] += (G_t - Q[s_t, a_t]) / N[s_t, a_t]
            G_tp1 = G_t

        # Periodic Greedy Evaluation
        if k % eval_every == 0:
            env_eval = make_env(is_slippery=is_slippery)
            score = eval_greedy_success(env_eval, Q, n_episodes=eval_episodes)
            eval_scores.append(score)
            eval_steps.append(cum_steps)   # x-axis = total env steps so far
            env_eval.close()

    env.close()
    return np.array(eval_steps), np.array(eval_scores), Q


# %% [2] On-policy SARSA

def train_sarsa_with_eval_steps(
    seed: int,
    is_slippery: bool = True,
    episodes: int = 8000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    eps: float = 0.10,
    eval_every: int = 500,
    eval_episodes: int = 200,
    max_steps: int = 200,
):
    """
    On-policy SARSA control with periodic greedy evaluation.
    Uses epsilon-greedy for training rollouts, greedy for evaluation.
    """
    # FrozenLake Setup
    np.random.seed(seed)
    env = make_env(is_slippery=is_slippery)
    nS, nA = env.observation_space.n, env.action_space.n
    # Multi-seed evaluation in Ch2.7 Albrecht
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # Initialize
    Q = np.zeros((nS, nA), dtype=float)
    eval_scores, eval_steps = [], []
    cum_steps = 0  # total steps taken in training

    # Training for n # of episodes
    for k in range(1, episodes + 1):
        s, _ = env.reset()
        s = int(s)
        a = choose_action_e_greedy(Q, s, eps)
        steps_ep = 0

        # loop over timesteps
        while True:
            # Step in env
            s_next, r, terminated, truncated, _ = env.step(a)
            s_next = int(s_next)

            # Choose next action on-policy (e-greedy), except at terminal
            if not (terminated or truncated):
                a_next = choose_action_e_greedy(Q, s_next, eps)
            else:
                a_next = None

            # TD target = on-policy e-greedy target
            td_target = r if (terminated or truncated) else (r + gamma * Q[s_next, a_next])
            # Update Q
            Q[s, a]  += alpha * (td_target - Q[s, a])
            
            # Break loop when reaching terminal or Exceeding max steps
            steps_ep += 1
            if terminated or truncated or steps_ep >= max_steps:
                break

            # Move forward
            s, a = s_next, a_next

        # track total steps
        cum_steps += steps_ep

        # Periodic greedy evaluation
        if k % eval_every == 0:
            env_eval = make_env(is_slippery=is_slippery)
            score = eval_greedy_success(env_eval, Q, n_episodes=eval_episodes)
            eval_scores.append(score)
            eval_steps.append(cum_steps)
            env_eval.close()

    env.close()
    return np.array(eval_steps), np.array(eval_scores), Q


# %% [3] Off-policy Q-learning

def train_q_learning_with_eval_steps(
    seed: int,
    is_slippery: bool = True,
    episodes: int = 8000,
    alpha: float = 0.10,
    gamma: float = 0.95,
    eps: float = 0.10,
    eval_every: int = 500,
    eval_episodes: int = 200,
    max_steps: int = 200,
):
    """
    Off-policy Q-learning control with periodic greedy evaluation.
    Uses epsilon-greedy for training rollouts, greedy for evaluation.
    """
    # FrozenLake Setup
    np.random.seed(seed)
    env = make_env(is_slippery=is_slippery)
    nS, nA = env.observation_space.n, env.action_space.n
    # Multi-seed evaluation setup
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # Initialize
    Q = np.zeros((nS, nA), dtype=float)
    eval_scores, eval_steps = [], []
    cum_steps = 0  # total timesteps so far

    # Training for n # of episodes
    for k in range(1, episodes + 1):
        s, _ = env.reset()
        s = int(s)
        steps_ep = 0

        # loop over timesteps
        while True:               
            # e-greedy action selection
            a = choose_action_e_greedy(Q, s, eps)

            # Step in env
            s_next, r, terminated, truncated, _ = env.step(a)
            s_next = int(s_next)

            # TD target = off-policy greedy target
            td_target = r if (terminated or truncated) else (r + gamma * np.max(Q[s_next]))
            # Update Q
            Q[s, a]  += alpha * (td_target - Q[s, a])

            # Break loop when reaching terminal or Exceeding max steps
            steps_ep += 1
            if terminated or truncated or steps_ep >= max_steps:
                break

            # Move forward
            s = s_next

        # track total steps
        cum_steps += steps_ep

        # Periodic greedy evaluation
        if k % eval_every == 0:
            env_eval = make_env(is_slippery=is_slippery)
            score = eval_greedy_success(env_eval, Q, n_episodes=eval_episodes)
            eval_scores.append(score)
            eval_steps.append(cum_steps)
            env_eval.close()

    env.close()
    return np.array(eval_steps), np.array(eval_scores), Q






# %% Run experiments for both slippery and non-slippery FrozenLake

EPISODES = 8000
EVAL_EVERY = 500
EVAL_EPISODES = 200
SEEDS = [0, 1, 2, 3, 4]

def run_multi_seed(trainer_fn, is_slippery):
    steps_list, scores_list, Q_list = [], [], []
    for sd in SEEDS:
        steps_sd, scores_sd, Q_sd = trainer_fn(
            seed=sd,
            is_slippery=is_slippery,
            episodes=EPISODES,
            eval_every=EVAL_EVERY,
            eval_episodes=EVAL_EPISODES,
            max_steps=200
        )
        steps_list.append(steps_sd)
        scores_list.append(scores_sd)
        Q_list.append(Q_sd)
    steps = steps_list[0]  # checkpoints identical across seeds
    scores = np.vstack(scores_list)  # (n_seeds, n_checkpoints)
    return steps, scores, Q_list

def mean_std(scores):
    return scores.mean(axis=0), scores.std(axis=0)

def render_policy(Q, lake_map):
    """
    Print arrows for greedy policy in Frozen Lake setting
    """
    lake_arr = np.array([list(row) for row in lake_map])
    n_rows, n_cols = lake_arr.shape
    
    def rc_to_s(r,c): 
        return r*n_cols + c
    
    holes = {(r, c) for r, c in zip(*np.where(lake_arr == "H"))}
    goal  = tuple(zip(*np.where(lake_arr == "G")))[0]
    terminals = {rc_to_s(*goal)} | {rc_to_s(r,c) for (r,c) in holes}
    arrow = {0:"←", 1:"↓", 2:"→", 3:"↑"}
    policy = np.argmax(Q, axis=1)

    print("Greedy policy:")
    for r in range(n_rows):
        row_str = []
        for c in range(n_cols):
            s = rc_to_s(r,c)
            if s in terminals:
                row_str.append(lake_arr[r, c])
            else:
                row_str.append(arrow[int(policy[s])])
        print(" ".join(row_str))


def plot_value_function(Q, lake_map, method_name="RL method"):
    """
    For given Q, heatmap of V(s) = max_a Q(s,a) -> Greedy
    """
    lake_arr = np.array([list(row) for row in lake_map])
    n_rows, n_cols = lake_arr.shape
    V = np.max(Q, axis=1).reshape(n_rows, n_cols)
    
    # Plot Value Function
    plt.figure(figsize=(4.2, 3.8))
    im = plt.imshow(V, origin="upper")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"{method_name}: V(s) = max_a Q(s,a)")
    plt.xticks(range(n_cols)); plt.yticks(range(n_rows))
    for (r,c), val in np.ndenumerate(V):
        plt.text(c, r, f"{val:.2f}", ha="center", va="center",
                 fontsize=8, color="white")
    plt.tight_layout()
    plt.show()


def run_and_plot(is_slippery: bool):
    # Run three methods
    mc_steps, mc_scores, mc_Qs       = run_multi_seed(train_mc_with_eval_steps, is_slippery)
    sarsa_steps, sarsa_scores, s_Qs  = run_multi_seed(train_sarsa_with_eval_steps, is_slippery)
    ql_steps, ql_scores, ql_Qs       = run_multi_seed(train_q_learning_with_eval_steps, is_slippery)

    # Compute mean ± std
    mc_mean, mc_std       = mean_std(mc_scores)
    sarsa_mean, sarsa_std = mean_std(sarsa_scores)
    ql_mean, ql_std       = mean_std(ql_scores)

    # Plot learning curves (Albrecht §2.7 style)
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(mc_steps, mc_mean, label="MC (greedy eval)")
    plt.fill_between(mc_steps, mc_mean-mc_std, mc_mean+mc_std, alpha=0.2)

    plt.plot(sarsa_steps, sarsa_mean, label="SARSA (greedy eval)")
    plt.fill_between(sarsa_steps, sarsa_mean-sarsa_std, sarsa_mean+sarsa_std, alpha=0.2)

    plt.plot(ql_steps, ql_mean, label="Q-learning (greedy eval)")
    plt.fill_between(ql_steps, ql_mean-ql_std, ql_mean+ql_std, alpha=0.2)

    suffix = "slippery" if is_slippery else "non-slippery"
    plt.title(f"Greedy Evaluation Return vs. Time Steps ({suffix})")
    plt.xlabel("Environment timesteps (cumulative)")
    plt.ylabel("Evaluation return (mean over seeds)")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Visualize learned greedy policies and values (take seed 0 run)
    lake_map = ["SFFF","FHFH","FFFH","HFFG"]

    print(f"\n=== First-Visit MC — greedy policy ({suffix}) ===")
    render_policy(mc_Qs[0], lake_map)
    plot_value_function(mc_Qs[0], lake_map, method_name=f"MC ({suffix})")
    
    print(f"\n=== SARSA — greedy policy ({suffix}) ===")
    render_policy(s_Qs[0], lake_map)
    plot_value_function(s_Qs[0], lake_map, method_name=f"SARSA ({suffix})")
    
    print(f"\n=== Q-learning — greedy policy ({suffix}) ===")
    render_policy(ql_Qs[0], lake_map)
    plot_value_function(ql_Qs[0], lake_map, method_name=f"Q-learning ({suffix})")


    return mc_Qs, s_Qs, ql_Qs  # return trained Qs in case you need them

# %% Run for slippery and non-slippery

if __name__ == "__main__":
    mc_Qs_det, s_Qs_det, ql_Qs_det = run_and_plot(is_slippery=False)  # non-slippery
    mc_Qs_slip, s_Qs_slip, ql_Qs_slip = run_and_plot(is_slippery=True)  # slippery
