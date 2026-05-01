import numpy as np
import random
import pandas as pd
import time
from pathlib import Path
from itertools import product

Blotto_Troops = 10
fields = 3  # consistent for the game
Lotso_increments = 2
MIN_LOTSO_TROOPS = 4
MAX_LOTSO_TROOPS = 16
LOTSO_TROOP_VALUES = list(range(MIN_LOTSO_TROOPS, MAX_LOTSO_TROOPS + 1, Lotso_increments))
FINAL_Q_TABLE_PATH = Path(__file__).with_name("final_q_table.json")

moves = [p for p in product(range(0, 11), repeat=3) if sum(p) == 10]


def _render_progress(current, total, prefix="", width=28):
    total = max(int(total), 1)
    current = min(max(int(current), 0), total)
    filled = int(width * current / total)
    bar = "█" * filled + "-" * (width - filled)
    percent = 100 * current / total
    return f"{prefix}[{bar}] {percent:6.2f}% ({current}/{total})"


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _print_progress(completed_steps, total_steps, start_time):
    elapsed = time.time() - start_time
    rate = elapsed / completed_steps if completed_steps > 0 else 0.0
    remaining = rate * max(total_steps - completed_steps, 0)
    line = (
        _render_progress(completed_steps, total_steps, prefix="Progress ")
        + f"  elapsed {_format_duration(elapsed)}"
        + f"  eta {_format_duration(remaining)}"
    )
    print(f"\r{line}", end="", flush=True)


def _finish_progress(start_time, total_steps):
    elapsed = time.time() - start_time
    print(f"\r{_render_progress(total_steps, total_steps, prefix='Progress ')}  elapsed {_format_duration(elapsed)}  eta 0:00")


def _sample_random_distribution(total_troops, n_fields):
    """Randomly place total_troops over n_fields bins."""
    move = np.zeros(n_fields, dtype=int)
    for _ in range(total_troops):
        nbin = random.randint(0, n_fields - 1)
        move[nbin] += 1
    return tuple(move.tolist())



def Lotso_Choose():
    # Randomly selects Lotso troops in increments of 5 between 5 and 30.
    troops = random.choice(LOTSO_TROOP_VALUES)
    move = _sample_random_distribution(troops, fields)
    return move, troops


def Blotto_Choose(Q, lotso_troops):
    state = lotso_troops
    state_values = Q.loc[state]
    max_q = state_values.max()
    best_actions = [a for a in state_values.index if state_values[a] == max_q]
    action = random.choice(best_actions)
    move = action
    return move, action, state

def Define_Winner(lotso_move, blotto_move):
    score = 0
    for field_i in range(fields):
        if lotso_move[field_i] > blotto_move[field_i]:
            score -= 1
        elif lotso_move[field_i] < blotto_move[field_i]:
            score += 1
    return score

"""
Update the Q-value for a given state-action pair using the Q-learning rule/equation.
Q: Q-table storing action-value estimates (Q[s,a])
state(int): the current state s where the agent takes action
action(int): the action taken by the agent.
reward(float): the immediate reward received after taking the action.
next_state(int): the state the agent transitions to after doing an action.
alpha(float): the learning rate/step size.
gamma(float): the discount factor which determines how much futur rewards are vlaued comapred to immediate reward.
returns: updated Q table and sampled next state
"""
def update_Q(Q, state, action, reward, next_state, alpha, gamma):
    current_q_value = Q.at[state, action]
    max_future_q = Q.loc[next_state].max()
    td_target = reward + gamma * max_future_q
    td_error = td_target - current_q_value
    Q.at[state, action] = current_q_value + alpha * td_error
    return Q


def run_episode(Q, T, experiment):
    lotso_move, lotso_troops = Lotso_Choose()
    state = lotso_troops
    T += 1

    alpha = experiment.get("alpha", 0.1)
    gamma = experiment.get("gamma", 0.9)
    explore = experiment.get("epsilon", 0.1)

    if np.random.random() < explore:
        action = random.choice(moves)
        blotto_move = action
    else:
        blotto_move, action, _ = Blotto_Choose(Q, lotso_troops)

    game_score = Define_Winner(lotso_move, blotto_move)
    if game_score > 0:
        reward = 1
        winner = "Blotto"
        loser = "Lotso"
    elif game_score < 0:
        reward = -1
        winner = "Lotso"
        loser = "Blotto"
    else:
        reward = 0
        winner = "Tie"
        loser = "Tie"

    _, next_state = Lotso_Choose()
    Q = update_Q(Q, state, action, reward, next_state, alpha, gamma)

    episode_info = {
        "state_lotso_troops": state,
        "winner": winner,
        "loser": loser,
        "reward": reward,
        "blotto_move": str(blotto_move),
        "lotso_move": str(lotso_move),
    }
    return episode_info, T, Q


def run_simulation(experiment, n_episodes, n_sims=100):
    wins_by_episode = [[] for _ in range(n_episodes)]
    final_q_tables = []
    detailed_log_rows = []
    start_time = time.time()
    total_steps = max(1, n_episodes * n_sims)
    completed_steps = 0

    for sim in range(n_sims):
        q_table = pd.DataFrame(
            data=0.0,
            index=LOTSO_TROOP_VALUES,
            columns=moves,
        )
        T = 0

        _print_progress(sim * n_episodes, total_steps, start_time)

        for episode in range(n_episodes):
            episode_info, T, q_table = run_episode(q_table, T, experiment)
            # Convert winner to win (1 for Blotto win, 0 otherwise)
            win = 1 if episode_info["winner"] == "Blotto" else 0
            wins_by_episode[episode].append(win)
            
            # Store detailed log for per-troop analysis
            detailed_log_rows.append({
                "episode": episode + 1,
                "state_lotso_troops": episode_info["state_lotso_troops"],
                "blotto_win": win,
            })

            completed_steps += 1
            if completed_steps % max(1, total_steps // 200) == 0 or completed_steps == total_steps:
                _print_progress(completed_steps, total_steps, start_time)

        final_q_tables.append(q_table)

    # Calculate average wins per episode across all simulations
    avg_wins = [float(np.mean(ep_wins)) for ep_wins in wins_by_episode]
    
    # Save detailed logs for per-troop analysis
    detailed_df = pd.DataFrame(detailed_log_rows)
    detailed_csv_path = Path(__file__).with_name("detailed_logs.csv")
    detailed_df.to_csv(detailed_csv_path, index=False)
    
    # Save the final Q-table (averaged across all simulations)
    avg_q_table = sum(final_q_tables) / len(final_q_tables)
    avg_q_table.to_json(FINAL_Q_TABLE_PATH, orient="split")

    _finish_progress(start_time, total_steps)
    
    return avg_wins, avg_q_table



def run_experiment():
    experiment = {
        "alpha": 0.1,
        "gamma": 0.9,
        "epsilon": 0.1,
    }
    n_episodes = 20000
    n_sims = 150
    avg_rewards, q_table = run_simulation(experiment, n_episodes, n_sims=n_sims)
    return avg_rewards, q_table


def main():
    print("Better Blotto")
    avg_rewards, q_table = run_experiment()

    print(f"Episodes trained: {len(avg_rewards)}")
    print(f"Average reward over all episodes: {np.mean(avg_rewards):.4f}")
    print("\nQ-table preview (rows = Lotso troops, columns = Blotto troop distributions):")
    print(q_table.iloc[:, :10])

main()