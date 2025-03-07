import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from race_track_environment import RaceTrackEnvironment
from MonteCarlo_control import OffPolicyMCC

class PlotResults:
    def __init__(self, env, Q, rewards):
        self.env = env
        self.Q = Q
        self.target_policy = self.generate_optimal_policy()
        self.rewards = rewards

    def generate_optimal_policy(self):
        target_policy = {}
        for state in self.Q.keys():
            target_policy[state] = OffPolicyMCC.max_argmax(self,input=self.Q[state])
        return target_policy

    def moving_average(self,y,window_size):
        """Compute moving average for smoothing."""
        return np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    def generate_trajectories(self):
        trajectories = [[],[]]
        for i in range(2):
            self.env.reset()
            state = OffPolicyMCC.convert_state_to_key(self,state=(self.env.state,self.env.speed))
            finished = False
            while finished==False:
                trajectories[i].append(state)
                action = self.target_policy[state]
                next_state, next_speed, reward, finished = self.env.one_step(action = action)
                state = OffPolicyMCC.convert_state_to_key(self,state=(next_state,next_speed))
            trajectories[i].append(state)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for i in range(2):
            map = self.env.map.copy()
            for traj in trajectories[i]:
                map[traj[0], traj[1]] = 0.6
              # Create 1 row, 2 columns
            axes[i].imshow(map)
            sns.heatmap(map, linewidths=1, ax=axes[i])
            axes[i].set_title('trajectories trace B no acc env')

        plt.savefig(f'./plots/trajectories trace B no acc.png')  # Save the figure
        plt.show()

    def plot_rewards(self):

        # Apply moving average smoothing
        window_size = 100
        y1 = np.array(self.rewards[0])
        y2 = self.rewards[1]
        x = np.arange(len(y1))
        # Adjust for better smoomthing
        y1_smooth = self.moving_average(y1, window_size)
        y2_smooth = self.moving_average(y2, window_size)
        x_smooth = x[:len(y1_smooth)]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(x_smooth, y1_smooth, 'b-', label="epsilon=0.2", alpha=0.8)
        plt.plot(x_smooth, y2_smooth, 'r-', label="epsilon=0.1", alpha=0.8)
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title("EMA Smoothed Training Reward Progress Trace A diff epsilon")
        plt.legend()
        plt.grid(True)
        plt.ylim([-100, 0])
        plt.savefig(f'./plots/track_A_regular_diff_epsilon.png')
        plt.show()
        # Adjust x to match new smoothed y

        y=1







