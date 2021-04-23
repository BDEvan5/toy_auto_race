import torch
from toy_auto_race.ImitationLearning import ImitationNet, BufferIL, Actor

import numpy as np 


class ImitationBase: 
    def __init__(self, agent_name, sim_conf) -> None:
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer

        self.distance_scale = 20 # max meters for scaling

    def transform_obs(self, obs):
        max_angle = 3.14

        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_steer]
        target_angle = [obs[5]/max_angle]
        target_distance = [obs[6]/self.distance_scale]

        scan = obs[7:-1]

        nn_obs = np.concatenate([cur_v, cur_d, target_angle, target_distance, scan])
        # nn_obs = np.concatenate([cur_d, target_angle, scan])

        return nn_obs

    def plan_act(self, obs):
        v = 4

        nn_obs = self.transform_obs(obs)
        nn_obs = np.reshape(nn_obs, (1, -1))
        nn_act = self.actor(nn_obs).data.numpy().flatten()
        steering = nn_act[0] * self.max_steer

        action = np.array([steering, v])

        return action

    def reset_lap(self):
        pass


class ImitationTrain(ImitationNet, ImitationBase): 
    def __init__(self, agent_name, sim_conf):
        ImitationNet.__init__(self, agent_name)
        ImitationBase.__init__(self, agent_name, sim_conf)

        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.distance_scale = 10

    def load_buffer(self, buffer_name):
        self.buffer.load_data(buffer_name)

    def aggregate_buffer(self, new_buffer):
        for sample in new_buffer.storage:
            self.buffer.add(sample)

        new_buffer.storage.clear()
        new_buffer.ptr = 0

    def save_step(self, state, action):
        nn_obs = self.transform_obs(state)
        action = action[0] / self.max_steer
        self.buffer.add((nn_obs, action))

    # def pre_train(self):
    #     self.train(20000)


class ImitationTest(ImitationBase):
    def __init__(self, agent_name, sim_conf) -> None:
        ImitationBase.__init__(self, agent_name, sim_conf)

        filename = '%s/%s_actor.pth' % ("Vehicles", self.name)
        self.actor = torch.load(filename)


