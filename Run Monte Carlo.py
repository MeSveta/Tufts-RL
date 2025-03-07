import pickle
import numpy as np
from race_track_environment import RaceTrackEnvironment
from showing_results import PlotResults
from MonteCarlo_control import OffPolicyMCC

train = False
with open('./race_track_env/maps/track_b.npy', 'rb') as f:
    map = np.load(f)
env = RaceTrackEnvironment(map)
if train:
    num_episodes = 100000
    agent = OffPolicyMCC(env,num_episodes=num_episodes)
    agent.train(behavior_policy_epsilon=0.1)

    agent_no_acc = OffPolicyMCC(env, num_episodes=num_episodes)
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
    #plot_results.generate_trajectories()
    plot_results.plot_rewards()


    # # Evaluation
    # policy = np.argmax(agent.Q, axis=-1)  # greedy policy
    # env = RaceTrack(track_sel, None, 20)
    # fig = plt.figure(figsize=(12, 5), dpi=150)
    # fig.suptitle('Sample trajectories', size=12, weight='bold')
    #
    # for i in range(10):
    #     track_map = np.copy(env.track_map)
    #     state, obs = env.reset()
    #     terminated = False
    #     while not terminated:
    #         track_map[state[0], state[1]] = 0.6
    #         action = policy[state]
    #         next_state, reward, terminated = env.step(action)
    #         state = next_state
    #
    #     ax = plt.subplot(2, 5, i + 1)
    #     ax.axis('off')
    #     ax.imshow(track_map, cmap='GnBu')
    # plt.tight_layout()
    # plt.savefig(f'./plots/exercise_5_12/track_{track_sel}_paths.png')
    # plt.show()
    #
    # reward = 1
    # rewards_vec = []
    # #self check
    # action = np.random.choice(len(env._action_to_acceleration.keys()))
    #
    # while reward!=0:
    #     state,speed, reward = env.one_step(action)
    #     rewards_vec.append(reward)

    y=1