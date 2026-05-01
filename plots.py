import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DETAILED_LOGS_CSV = Path(__file__).with_name("detailed_logs.csv")
FINAL_Q_TABLE_PATH = Path(__file__).with_name("final_q_table.json")
PLOTS_DIR = Path(__file__).with_name("plots_output")

def _load_detailed_logs():
    if not DETAILED_LOGS_CSV.exists() or DETAILED_LOGS_CSV.stat().st_size == 0:
        raise FileNotFoundError("detailed_logs.csv is missing or empty. Run main.py first to generate detailed logs.")
    return pd.read_csv(DETAILED_LOGS_CSV)


def _load_final_q_table():
    if not FINAL_Q_TABLE_PATH.exists():
        raise FileNotFoundError(f"{FINAL_Q_TABLE_PATH} is missing. Run main.py first to generate the final Q-table.")
    return pd.read_json(FINAL_Q_TABLE_PATH, orient="split")


def avg_wins_vs_episode():
    # Plot average wins across all simulations vs episode number, broken down by opponent troops.
    detailed_df = _load_detailed_logs()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Group by episode and opponent troops, calculate average wins
    wins_by_episode_troops = (
        detailed_df
        .groupby(["episode", "state_lotso_troops"], as_index=False)["blotto_win"]
        .mean()
    )

    # Bin episodes into a fixed number of bins so we have the same number of x points per troop
    NUM_BINS = 100
    ep_min = int(wins_by_episode_troops["episode"].min())
    ep_max = int(wins_by_episode_troops["episode"].max())
    if ep_max <= ep_min:
        bins = np.array([ep_min, ep_min + 1])
    else:
        bins = np.linspace(ep_min, ep_max, NUM_BINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Assign bins and compute binned mean per troop
    wins_by_episode_troops["bin_idx"] = pd.cut(wins_by_episode_troops["episode"], bins=bins, labels=False, include_lowest=True)
    binned = (
        wins_by_episode_troops
        .groupby(["bin_idx", "state_lotso_troops"], as_index=False)["blotto_win"]
        .mean()
    )

    # Prepare a pivot-like structure ensuring all bins exist for each troop
    troop_values = sorted(wins_by_episode_troops["state_lotso_troops"].unique())
    binned_full = []
    for troops in troop_values:
        troop_bins = binned[binned["state_lotso_troops"] == troops].set_index("bin_idx").reindex(range(len(bin_centers)))
        binned_full.append(pd.DataFrame({
            "bin_idx": range(len(bin_centers)),
            "bin_center": bin_centers,
            "state_lotso_troops": troops,
            "blotto_win": troop_bins["blotto_win"].to_numpy(),
        }))
    binned_full_df = pd.concat(binned_full, ignore_index=True)

    plt.figure(figsize=(14, 7))

    # Plot a line for each opponent troop value (binned)
    for troops in troop_values:
        troop_data = binned_full_df[binned_full_df["state_lotso_troops"] == troops]
        plt.plot(troop_data["bin_center"], troop_data["blotto_win"], linewidth=1.2, label=f"{int(troops)} troops", alpha=0.6)

    # Compute mean across troop values at each bin (unweighted)
    mean_across_bins = (
        binned_full_df
        .groupby("bin_idx", as_index=False)["blotto_win"]
        .mean()
    )

    # Add a thick black line showing the mean across troop values per bin
    plt.plot(bin_centers, mean_across_bins["blotto_win"].to_numpy(), color="black", linewidth=3.5, label="Mean across troop values", zorder=10)

    plt.title("Average Blotto Wins vs Episode Number (binned into 100 points per troop)")
    plt.xlabel("Episode Number")
    plt.ylabel("Average Wins (Fraction)")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.grid(alpha=0.3)
    out_path = PLOTS_DIR / "avg_wins_vs_episode.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def Q_learning_heatmap():
    # Enhanced 2D heatmap of the final Q-table showing all possible moves (troop distributions).
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    q_table = _load_final_q_table()
    
    q_values = q_table.to_numpy(dtype=float)
    
    # Format column labels to show troop distributions
    column_labels = [str(col) for col in q_table.columns]
    row_labels = [str(int(idx)) for idx in q_table.index]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(q_values, aspect="auto", cmap="RdYlGn")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(q_table.columns)))
    ax.set_yticks(np.arange(len(q_table.index)))
    ax.set_xticklabels(column_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(row_labels, fontsize=9)
    
    plt.colorbar(im, ax=ax, label="Q-Value")
    plt.title("Final Q-Learning Board - State-Action Values\n(Rows: Opponent Troops, Columns: Blotto Troop Distributions)")
    plt.xlabel("Blotto Actions (Troop Distribution)")
    plt.ylabel("Lotso Opponent Troops")
    
    out_path = PLOTS_DIR / "final_q_learning_heatmap.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def avg_wins_vs_opponent_troops_last_10_episodes():
    # Average win rate vs opponent troops for only the last 10 episodes across all simulations.
    detailed_df = _load_detailed_logs()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get max episode number and filter for last 10 episodes
    max_episode = int(detailed_df["episode"].max())
    last_10_cutoff = max_episode - 9  # Last 10 episodes
    last_10_df = detailed_df[detailed_df["episode"] >= last_10_cutoff].copy()
    
    # Calculate average wins per opponent troop value
    avg_wins_vs_troops = (
        last_10_df
        .groupby("state_lotso_troops", as_index=True)["blotto_win"]
        .mean()
        .sort_index()
    )
    
    # Reindex to the full troop grid so the x-axis reflects the actual troop values.
    troop_values = np.sort(detailed_df["state_lotso_troops"].unique())
    avg_wins_vs_troops = avg_wins_vs_troops.reindex(troop_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        avg_wins_vs_troops.index.to_numpy(),
        avg_wins_vs_troops.to_numpy(),
        marker="o",
        linewidth=2,
        color="tab:green",
        markersize=8,
    )
    ax.set_xticks(troop_values)
    ax.set_xlim(troop_values.min() - 0.5, troop_values.max() + 0.5)
    ax.set_title(f"Average Blotto Wins vs Opponent Troops (Last 10 Episodes: {last_10_cutoff}-{max_episode})")
    ax.set_xlabel("Lotso Opponent Troops")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    out_path = PLOTS_DIR / "avg_wins_vs_opponent_troops_last_10_eps.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def wins_vs_games_counts():
    # Plot episode number vs average cumulative Blotto wins per simulation
    detailed_df = _load_detailed_logs()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine episodes and number of simulations (assumes logs are written sim-by-sim)
    n_episodes = int(detailed_df["episode"].nunique()) if "episode" in detailed_df.columns else 0
    total_rows = len(detailed_df)
    n_sims = total_rows // max(1, n_episodes)

    # Reconstruct simulation id by row order: sim_id = index // n_episodes
    df = detailed_df.reset_index(drop=True).copy()
    df["sim_id"] = (df.index // max(1, n_episodes)).astype(int)

    # Ensure rows are ordered by sim then episode, then compute cumulative wins per sim
    df = df.sort_values(["sim_id", "episode"]).reset_index(drop=True)
    df["cum_wins"] = df.groupby("sim_id")["blotto_win"].cumsum()

    # Average cumulative wins across simulations for each episode
    avg_cum_wins = df.groupby("episode", as_index=True)["cum_wins"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(avg_cum_wins.index, avg_cum_wins.values, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Average Cumulative Wins per Simulation")
    ax.set_title("Average Cumulative Blotto Wins per Simulation vs Episode")
    ax.grid(alpha=0.3)

    out_path = PLOTS_DIR / "wins_vs_games_counts.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def wins_vs_games_counts_first_n(n=200):
    # Plot average cumulative Blotto wins per simulation for the first `n` episodes
    detailed_df = _load_detailed_logs()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine episodes and number of simulations
    n_episodes = int(detailed_df["episode"].nunique()) if "episode" in detailed_df.columns else 0
    if n_episodes == 0:
        raise ValueError("No episode data found in detailed_logs.csv")

    # Reconstruct simulation id by row order: sim_id = index // n_episodes
    df = detailed_df.reset_index(drop=True).copy()
    df["sim_id"] = (df.index // max(1, n_episodes)).astype(int)

    # Ensure rows are ordered by sim then episode, then compute cumulative wins per sim
    df = df.sort_values(["sim_id", "episode"]).reset_index(drop=True)
    df["cum_wins"] = df.groupby("sim_id")["blotto_win"].cumsum()

    # Limit to first n episodes (or available max if fewer)
    max_episode_available = int(df["episode"].max())
    last_episode = min(n, max_episode_available)
    df_limited = df[df["episode"] <= last_episode]

    # Average cumulative wins across simulations for each episode
    avg_cum_wins = df_limited.groupby("episode", as_index=True)["cum_wins"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(avg_cum_wins.index, avg_cum_wins.values, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Average Cumulative Wins per Simulation")
    ax.set_title(f"Average Cumulative Blotto Wins per Simulation vs Episode (first {last_episode} episodes)")
    ax.grid(alpha=0.3)

    out_path = PLOTS_DIR / f"wins_vs_games_counts_first_{last_episode}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return str(out_path)


def main():
    print("Saving plots to:", PLOTS_DIR)
    print("avg_wins_vs_episode:", avg_wins_vs_episode())
    print("avg_wins_vs_opponent_troops_last_10_episodes:", avg_wins_vs_opponent_troops_last_10_episodes())
    print("wins_vs_games_counts:", wins_vs_games_counts())
    try:
        print("wins_vs_games_counts_first_200:", wins_vs_games_counts_first_n(200))
    except Exception:
        # If there isn't enough data or another issue, skip the first-200 plot gracefully
        print("wins_vs_games_counts_first_200: skipped (insufficient data)")
    print("Q_learning_heatmap:", Q_learning_heatmap())


if __name__ == "__main__":
    main()
