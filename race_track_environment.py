import numpy as np


STARTING = 0.8
FINISHING = 0.4

class RaceTrackEnvironment:
    def __init__(self, map):
        self.map = map
        # self.action = action
        self.start_states = np.where(map == STARTING)
        self.finish_states = np.where(map == FINISHING)
        self.reset()
        # Mapping the integer action to acceleration tuple
        self.action_to_acceleration = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1)
        }
        self.num_actions = len(self.action_to_acceleration)

    # reset the car to one of the starting positions
    def reset(self):
        # Select start position randomly from the starting line
        start_idx = np.random.choice(self.start_states[0].shape[0])
        self.state = np.array((self.start_states[0][start_idx],self.start_states[1][start_idx]))
        self.speed = np.zeros(2, dtype=int)

    # state update
    def state_update(self, action):
        action_vec = self.action_to_acceleration[action]
        # Can't move back , vertically down
        self.speed[0] = np.clip(self.speed[0] + action_vec[0], -4, 0)
        # Can move left and right
        self.speed[1] = np.clip(self.speed[1] + action_vec[1], -4, 4)
        self.state = self.state + self.speed
        if self.check_out_of_track():
            self.reset()

    # checks if the car out of track
    def check_out_of_track(self):
        status = False
        map_size = self.map.shape
        if (self.state[0]>map_size[0]-1 or self.state[0]<0) or (self.state[1]>map_size[1]-1 or self.state[1]<0):
            status = True
        elif self.map[self.state[0],self.state[1]]==0:
            status = True
        return status

    def reached_finish_line(self):
        if self.map[self.state[0],self.state[1]] == FINISHING:
            return True
        else: return False

    def one_step(self, action):
        finished = False
        self.state_update(action)
        finished = self.reached_finish_line()
        return self.state, self.speed, -1, finished



if __name__ == '__main__':
    with open('./race_track_env/maps/track_b.npy', 'rb') as f:
        map = np.load(f)
    env = RaceTrackEnvironment(map)
    reward = 1
    rewards_vec = []
    #self check
    action = np.random.choice(len(env.action_to_acceleration.keys()))

    while reward!=0:
        state,speed,reward = env.one_step(action)
        rewards_vec.append(reward)

    y=1




