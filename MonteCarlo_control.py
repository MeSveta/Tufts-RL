import pickle

import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from race_track_environment import RaceTrackEnvironment


class OffPolicyMCC:
    def __init__(self, env, num_episodes=10000, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.Q = self.init_Q(env.map, env.action_to_acceleration)  # Action-value function
        self.C = defaultdict(lambda: defaultdict(lambda: 0)) # Cumulative weights
        self.target_policy = defaultdict(lambda: 0, format='int')  # Target policy
        self.num_episodes = num_episodes
        self.reward_hist = np.zeros(shape=(num_episodes), dtype=np.float32)

    def init_Q(self, track_map, actions, velocity_range=(-4, 4)):
        """
        Initialize Q-values with two keys:
          - First key: (state, velocity) -> (x, y, v_x, v_y)
          - Second key: Action

        :param track_map: 2D numpy array representing the racing track (0 = obstacle, 1 = valid track)
        :param actions: List of possible actions [(a_x, a_y)]
        :param velocity_range: Tuple (min_velocity, max_velocity) defining velocity limits.
        :return: Nested dictionary Q[state, velocity][action] initialized to zero.
        """
        Q = {}
        min_v, max_v = velocity_range

        for x in range(track_map.shape[0]):  # Loop over rows
            for y in range(track_map.shape[1]):  # Loop over columns
                if track_map[x, y] != 0:  # Ignore walls

                    # If Start Position: Only initialize for (0,0) velocity
                    if x in self.env.start_states[0] and y in self.env.start_states[1]:
                        for v_x in range(-4, 1):
                            for v_y in range(min_v, max_v + 1):
                                #Q[(x, y, v_x, v_y)] = {action: np.random.normal()-500 for action in actions}
                                Q[(x, y, v_x, v_y)] = {action: 0 for action in actions}


                    # If Finish Position: No velocity needed
                    elif x in self.env.finish_states[0] and y in self.env.finish_states[1]:
                        for v_x in range(-4, 1):
                            for v_y in range(min_v, max_v + 1):
                                Q[(x, y, v_x, v_y)] = {action: np.random.normal()-500 for action in actions}


                    # Normal case: Iterate over velocities
                    else:
                        for v_x in range(-4, 1):
                            for v_y in range(min_v, max_v + 1):
                                Q[(x, y, v_x, v_y)] = {action: np.random.normal()-500 for action in actions}
        return Q

    def convert_state_to_key(self,state):
        """
        Converts (state, speed) tuple into a single key in (x, y, v_x, v_y) format.

        :param state: Tuple (self.env.state, self.env.speed)
                      - state[0]: NumPy array (x, y)
                      - state[1]: NumPy array (v_x, v_y)
        :return: Tuple (x, y, v_x, v_y)
        """
        return tuple(state[0].tolist() + state[1].tolist())

    def generate_episode(self, behavior_policy_epsilon, action_zero):
        """Generate an episode following behavior policy."""
        episode = []
        self.env.reset()
        state = self.convert_state_to_key((self.env.state,self.env.speed))
        finished= False
        while finished!=True:
            action, act_prob = self.create_behavior_policy(behavior_policy_epsilon,state)
            if action_zero and np.random.rand()<=0.1:
                action_no_acc = 4
                next_state, next_speed, reward, finished = self.env.one_step(action_no_acc)
            else:
                next_state, next_speed, reward, finished = self.env.one_step(action)
            episode.append((state, action, reward, act_prob))
            state = self.convert_state_to_key((next_state,next_speed))
        return episode

    def train(self, behavior_policy_epsilon=0.1, action_zero = False):
        num_episodes = self.num_episodes
        """Train using Off-Policy MC Control with Weighted Importance Sampling."""

        for state in self.Q.keys():

            self.target_policy[state] = self.max_argmax(self.Q[state])

        for episode_num in range(num_episodes):

            # Generate episode using behavior policy
            episode = self.generate_episode(behavior_policy_epsilon,action_zero)

            # Initialize G and importance sampling weight W
            G = 0
            W = 1

            self.reward_hist[episode_num] = np.sum([ii[2] for ii in episode])

            # Iterate backward through episode
            for t in reversed(range(len(episode))):
                state, action, reward, act_prob = episode[t]
                G = self.gamma * G + reward  # Compute return

                # Update cumulative sum of weights
                self.C[state][action] += W

                # Update action-value function using weighted importance sampling
                self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])

                # Improve policy (greedy update)
                self.target_policy[state] = self.max_argmax(self.Q[state])

                # Check if action differs from target policy
                if action != self.target_policy[state]:
                    break  # Stop the update early

                # Update importance sampling weight
                W *= 1 / act_prob
                if W == 0:
                    break  # Stop early if weight goes to zero
            if np.mod(episode_num,100) == 0:
                print(f'Episode: {episode_num}, reward: {self.reward_hist[episode_num]}, epsilon:{behavior_policy_epsilon}')

    def max_argmax(self,input):
        best_action = np.random.choice([key for key, value in input.items() if value == max(input.values())])
        return best_action

    def create_behavior_policy(self, epsilon, state):
        """Create an ε-soft behavior policy for exploration."""

        rand_val = np.random.rand()
        greedy_act = self.target_policy[state]

        if rand_val > epsilon:
            return greedy_act,(1 - epsilon + epsilon / self.env.num_actions)
        else:
            action = np.random.choice(self.env.num_actions)
            if action == greedy_act:
                return action,(1 - epsilon + epsilon / self.env.num_actions)
            else:
                return action,epsilon / self.env.num_actions

    def get_optimal_policy(self):
        """Return the final learned optimal policy."""
        return {state: self.max_argmax(actions) for state, actions in self.Q.items()}





