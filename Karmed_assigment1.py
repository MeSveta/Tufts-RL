# K-Armed Bandit Problem Implementation

import numpy as np
import matplotlib.pyplot as plt


class KArmedBandit:
    def __init__(self, k=10):
        """
        Initialize the k-armed bandit problem.

        Parameters:
            k (int): Number of arms (actions).
        """
        self.k = k
        self.q_true = np.random.normal(0, 1, self.k)  # True action values
        self.q_estimates = np.zeros(self.k)  # Estimated action values
        self.action_counts = np.zeros(self.k)  # Number of times each action was taken


    def get_reward(self, action):
        """
        Simulate the reward for the selected action.

        Parameters:
            action (int): Index of the chosen action.

        Returns:
            float: Reward drawn from a normal distribution centered at q_true[action].
        """
        return np.random.normal(self.q_true[action], 1)

    def update_estimates(self, action, reward):
        """
        Update the estimated value for a given action based on the received reward.

        Parameters:
            action (int): Index of the chosen action.
            reward (float): Observed reward.
        """
        self.action_counts[action] += 1
        self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]


class BanditPlotter:
    @staticmethod
    def plot_reward_distribution(bandit):
        """
        Plot the reward distributions for the k-armed bandit.

        Parameters:
            bandit (KArmedBandit): Instance of the KArmedBandit class.
        """
        k = bandit.k
        data = [np.random.normal(q, 1, 1000) for q in bandit.q_true]

        plt.figure(figsize=(8, 6))
        plt.violinplot(data, showmeans=True, showextrema=True)

        # Formatting the plot
        plt.title("Reward Distributions for the 10-Armed Bandit", fontsize=14)
        plt.xlabel("Action", fontsize=12)
        plt.ylabel("Reward Distribution", fontsize=12)
        plt.xticks(range(1, k + 1), labels=[f"a_{i + 1}" for i in range(k)])
        plt.grid(alpha=0.3)
        plt.show()

    @staticmethod
    def plot_learning_curves(steps, rewards, optimal_actions, epsilons):
        """
        Plot the learning curves for average reward and % optimal actions.

        Parameters:
            steps (int): Number of steps in the simulation.
            rewards (dict): Dictionary containing rewards for each epsilon.
            optimal_actions (dict): Dictionary containing % optimal actions for each epsilon.
            epsilons (list): List of epsilon values used in the simulation.
        """
        plt.figure(figsize=(12, 6))

        # Plot Average Reward
        plt.subplot(1, 2, 1)
        for eps in epsilons:
            plt.plot(range(steps), rewards[eps], label=f"ε={eps}")
        plt.title("Average Reward", fontsize=14)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        # Plot % Optimal Action
        plt.subplot(1, 2, 2)
        for eps in epsilons:
            plt.plot(range(steps), optimal_actions[eps], label=f"ε={eps}")
        plt.title("% Optimal Action", fontsize=14)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("% Optimal Action", fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


# Simulation
np.random.seed(42)  # For reproducibility

k = 10
steps = 1000
runs = 2000

# Epsilon values to compare
epsilons = [0, 0.01, 0.1]

rewards = {eps: np.zeros(steps) for eps in epsilons}
optimal_actions = {eps: np.zeros(steps) for eps in epsilons}

for eps in epsilons:
    for run in range(runs):
        bandit = KArmedBandit(k=k)
        optimal_action = np.argmax(bandit.q_true)

        for step in range(steps):
            if np.random.rand() < eps:
                action = np.random.choice(k)  # Explore
            else:
                action = np.argmax(bandit.q_estimates)  # Exploit

            reward = bandit.get_reward(action)
            bandit.update_estimates(action, reward)

            rewards[eps][step] += reward
            if action == optimal_action:
                optimal_actions[eps][step] += 1

# Average over runs
for eps in epsilons:
    rewards[eps] /= runs
    optimal_actions[eps] = (optimal_actions[eps] / runs) * 100

# Plotting learning curves
BanditPlotter.plot_learning_curves(steps, rewards, optimal_actions, epsilons)
y=1