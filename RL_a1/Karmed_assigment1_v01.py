# K-Armed Bandit Problem Implementation

import numpy as np
import matplotlib.pyplot as plt

class BanditEnvironment:
    def __init__(self, k=10,q_init_method="random", std = 0.1):
        """
        Initialize the k-armed bandit environment.

        Parameters:
            k (int): Number of arms (actions).
        """
        self.k = k
        self.std = std
        self.q_init_method = q_init_method
        self.q_true = self.initialize_q_true()

        

    def initialize_q_true(self):
        """
        Initialize the true action values based on the specified method.

        Parameters:
            method (str): Initialization method ("normal" or "uniform").

        Returns:
            np.ndarray: Initialized true action values.
        """
        if self.q_init_method == "random":
            return np.random.normal(0, 1, self.k)  # Default normal initialization
        elif self.q_init_method == "nonstationary":
            # value = np.random.normal(0, 1, 1)[0]  # Single random value
            # np.full(self.k, value) # Duplicate the same value for all actions
            return np.zeros(self.k) # init with zeros

    def update_q_true(self):
        if self.q_init_method=='nonstationary':
            self.q_true += np.random.normal(loc=0.0, scale=0.01, size=self.k)

    def get_reward(self, action):
        """
        Simulate the reward for the selected action.

        Parameters:
            action (int): Index of the chosen action.

        Returns:
            float: Reward drawn from a normal distribution centered at q_true[action].
        """
        return np.random.normal(self.q_true[action], 1)

    def get_optimal_action(self):
        """
        Get the optimal action based on the highest true action value.

        Returns:
            int: Index of the optimal action.
        """
        max_value = np.max(self.q_true)
        max_actions = np.where(self.q_true == max_value)[0]

        return np.random.choice(max_actions)# Randomly pick among the best actions

class BanditAgent:
    def __init__(self, k=10, epsilon=0.1, learning_method="random", step_size = 0.01, alpha = 0.1, beta=0.9):
        """
        Initialize the bandit agent.

        Parameters:
            k (int): Number of arms (actions).
            epsilon (float): Probability of choosing a random action (exploration).
        """
        self.k = k
        self.epsilon = epsilon
        self.q_estimates = np.zeros(self.k)  # Estimated action values
        self.action_counts = np.zeros(self.k)  # Number of times each action was taken
        self.learning_method = learning_method
        self.step_size = step_size
        self.G = np.zeros(self.k)
        self.beta = beta
        self.alpha = alpha


    def choose_action(self):
        """
        Choose an action using the epsilon-greedy strategy.

        Returns:
            int: Index of the chosen action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)  # Explore
        else:

            max_value = np.max(self.q_estimates)
            max_actions = np.where(self.q_estimates == max_value)[0]
            return np.random.choice(max_actions)  # Randomly pick among the best actions # Exploit

    def update_estimates(self, action, reward):
        """
        Update the estimated value for a given action based on the received reward.

        Parameters:
            action (int): Index of the chosen action.
            reward (float): Observed reward.
        """
        if self.learning_method == ("incrimental"):
            self.action_counts[action] += 1
            self.q_estimates[action] += (reward - self.q_estimates[action]) / self.action_counts[action]
        if self.learning_method==("constant_stepsize"):
            self.q_estimates[action] += (reward - self.q_estimates[action]) * self.step_size

    def update_step_size(self,action,reward):
        """
           Update Q-value estimates using RMSProp for adaptive learning rate.

           Parameters:
               action (int): Action taken.
               reward (float): Reward received.
           """
        # Temporal Difference (TD) error
        g_i = reward - self.q_estimates[action]

        # Update the moving average of squared gradients (RMSProp step)
        self.G[action] = self.beta * self.G[action] + (1 - self.beta) * (g_i ** 2)

        # Adaptive learning rate using RMSProp
        self.step_size = self.alpha / (np.sqrt(self.G[action]) + 1e-8)



class BanditPlotter:
    @staticmethod
    def plot_reward_distribution(environment):
        """
        Plot the reward distributions for the k-armed bandit environment.

        Parameters:
            environment (BanditEnvironment): Instance of the BanditEnvironment class.
        """
        k = environment.k
        data = [np.random.normal(q, 1, 1000) for q in environment.q_true]

        plt.figure(figsize=(8, 6))
        plt.violinplot(data, showmeans=True, showextrema=True)

        # Formatting the plot
        plt.title("Reward Distributions for the 10-Armed Bandit", fontsize=14)
        plt.xlabel("Action", fontsize=12)
        plt.ylabel("Reward Distribution", fontsize=12)
        plt.xticks(range(1, k + 1), labels=[f"a_{i+1}" for i in range(k)])
        plt.grid(alpha=0.3)
        plt.show()

    @staticmethod
    def plot_learning_curves(steps, rewards, optimal_actions, step_size_average, epsilons, learning_method_vec):
        """
        Plot the learning curves for average reward and % optimal actions.

        Parameters:
            steps (int): Number of steps in the simulation.
            rewards (dict): Dictionary containing rewards for each epsilon.
            optimal_actions (dict): Dictionary containing % optimal actions for each epsilon.
            epsilons (list): List of epsilon values used in the simulation.
        """
        plt.figure(figsize=(12, 6), dpi=100, facecolor='white')  # Correct placement

        # # Plot Average Reward
        # plt.subplot(2, 1, 1)  # No figsize here
        for l_method in learning_method_vec:
            for eps in epsilons:
                plt.plot(range(steps), rewards[(l_method, eps)], label=f"{l_method}, ε={eps}")

        plt.title("Average Reward", fontsize=14)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Average Reward", fontsize=12)
        plt.legend(loc='upper right')  # Adjust legend if needed
        plt.grid(alpha=0.3)

        plt.tight_layout()  # Adjust layout to avoid overlap
        plt.show()  # Display the plot


        # Plot % Optimal Action
        plt.figure(figsize=(12, 6), dpi=100, facecolor='white')
        for l_method in learning_method_vec:
            for eps in epsilons:
                plt.plot(range(steps), optimal_actions[(l_method, eps)], label=f"{l_method}, ε={eps}")
        plt.title("% Optimal Action", fontsize=14)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("% Optimal Action", fontsize=12)
        plt.ylim(0, 100)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.savefig('learning_curves.png')

        # Plot % Average Step Size
        plt.figure(figsize=(12, 6), dpi=100, facecolor='white')
        for l_method in learning_method_vec:
            for eps in epsilons:
                plt.plot(range(steps), step_size_average[(l_method, eps)], label=f"{l_method}, ε={eps}")
        plt.title("Average Step Size", fontsize=14)
        plt.xlabel("Steps", fontsize=12)
        plt.ylabel("Average Step Size", fontsize=12)
        plt.ylim(0, 2)
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        #plt.savefig('learning_curves.png')
        

        # # Plot Reward Distribution
        # plt.subplot(1, 1, 1)
        # data = [np.random.normal(q, 1, 1000) for q in environment.q_true]
        # plt.violinplot(data, showmeans=True, showextrema=True)
        # plt.title("Reward Distributions", fontsize=14)
        # plt.xlabel("Action", fontsize=12)
        # plt.ylabel("Reward Distribution", fontsize=12)
        # plt.grid(alpha=0.3)
        # 
        # plt.tight_layout()
        # plt.show()
        # plt.savefig('reward_distribution.png')

# Simulation
#np.random.seed(42)  # For reproducibility

#Environment parameters:
q_init_method = 'nonstationary'
std = 0.01
k = 10
# Agent parameters
learning_method_vec = ['constant_stepsize']
step_size = 0.1
#Experiment parameters
steps = 10000
runs = 2000

# Epsilon values to compare
epsilons = [0.1]

rewards = {
    (l_method, eps): np.zeros((runs, steps))
    for l_method in learning_method_vec
    for eps in epsilons
}

step_size_m = {
    (l_method, eps): np.zeros((runs, steps))
    for l_method in learning_method_vec
    for eps in epsilons
}

step_size_average = {
    (l_method, eps): np.zeros((runs, steps))
    for l_method in learning_method_vec
    for eps in epsilons
}
rewards_average = {
    (l_method, eps): np.zeros(steps)
    for l_method in learning_method_vec
    for eps in epsilons
}
optimal_actions = {
    (l_method, eps): np.zeros((runs, steps))
    for l_method in learning_method_vec
    for eps in epsilons
}
optimal_actions_average = {
    (l_method, eps): np.zeros(steps)
    for l_method in learning_method_vec
    for eps in epsilons
}

for learning_m in learning_method_vec:
    learning_method = learning_m
    
    for eps in epsilons:
        for run in range(runs):
            environment = BanditEnvironment(k=k, q_init_method = q_init_method, std=std)
            agent = BanditAgent(k=k, epsilon=eps, learning_method=learning_method, step_size=step_size)
            
            for step in range(steps):
                step_size_m[(learning_m,eps)][run][step] = agent.step_size
                optimal_action = environment.get_optimal_action()
                action = agent.choose_action()
                environment.update_q_true()
                reward = environment.get_reward(action)
                agent.update_estimates(action, reward)
                agent.update_step_size(action, reward)

                # updates q_true by random walk (meaning can change optimal action on every step)
    
                rewards[(learning_m,eps)][run][step] = reward  # Track per run and step
                if action == optimal_action:
                    optimal_actions[(learning_m,eps)][run][step] += 1

# Average over runs
for learning_m in learning_method_vec:
    for eps in epsilons:
        sum_rewards = np.sum(rewards[(learning_m,eps)],axis=0)
        rewards_average[(learning_m,eps)] = np.mean(rewards[(learning_m,eps)], axis=0)
        optimal_actions_average[(learning_m,eps)] = (np.mean(optimal_actions[(learning_m,eps)], axis=0)) * 100
        step_size_average[(learning_m,eps)] = np.mean(step_size_m[(learning_m,eps)], axis=0)


#save data
np.save('step_size_average01_rms_1.npy', step_size_average)
np.save('rewards_average01_rms_1.npy', rewards_average)
np.save('optimal_actions_average01_rms_1.npy', optimal_actions_average)

# Plotting learning curves
BanditPlotter.plot_learning_curves(steps, rewards_average, optimal_actions_average, step_size_average, epsilons, learning_method_vec)

# rewards_average01 = np.load('rewards_average_nonstationary_epsilon01.npy', allow_pickle=True).item()
# optimal_actions_average01 = np.load('optimal_actions_nonstationary_epsilon01.npy', allow_pickle=True).item()
#
y=1
