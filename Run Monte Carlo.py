import pickle
import numpy as np
from race_track_environment import RaceTrackEnvironment
from showing_results import PlotResults
from MonteCarlo_control import MCC_policy

train = False
with open('./race_track_env/maps/track_b.npy', 'rb') as f:
    map = np.load(f)
env = RaceTrackEnvironment(map)
if train:
    num_episodes = 100000
    agent = MCC_policy(env,num_episodes=num_episodes)
    agent.train(behavior_policy_epsilon=0.1)

    agent_no_acc = MCC_policy(env, num_episodes=num_episodes)
    agent_no_acc.train(behavior_policy_epsilon=0.1, action_zero = True)

    #Save agents learning

    with open(f'./plots/track_b_regular_Q_0.pkl', 'wb') as f:
        pickle.dump(agent.Q,f)

    with open(f'./plots/track_b_no_acc_Q_0.pkl','wb') as f:
        pickle.dump(agent_no_acc.Q,f)

    with open(f'./plots/track_b_regular_reward_Q_0.pkl', 'wb') as f:
        pickle.dump(agent.reward_hist,f)

    with open(f'./plots/track_b_no_acc_reward_Q_0.pkl','wb') as f:
        pickle.dump(agent_no_acc.reward_hist,f)

else:
    agent_rewards = []
    with open("./plots/track_a_regular.pkl", "rb") as f:
         agent_Q = pickle.load(f)
    with open("./plots/track_a_regular_reward_epsilon_02.pkl", "rb") as f:
        agent_rewards.append(np.squeeze(pickle.load(f)))
    with open("./plots/track_a_regular_reward.pkl", "rb") as f:
        agent_rewards.append(np.squeeze(pickle.load(f)))

    plot_results = PlotResults(env, agent_Q, agent_rewards)

    # Generate Trajectories
    #plot_results.generate_trajectories()
    # Generate rewrds convergence
    plot_results.plot_rewards()

    y=1