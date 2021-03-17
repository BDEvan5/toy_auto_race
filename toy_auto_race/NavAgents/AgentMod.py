import numpy as np 
import csv
from matplotlib import pyplot as plt

from toy_auto_race.TD3 import TD3
from toy_auto_race.Utils import LibFunctions as lib
from toy_auto_race.Utils.HistoryStructs import TrainHistory


class BaseMod:
    def __init__(self, agent_name, map_name, sim_conf, mod_conf) -> None:
        self.name = agent_name
        self.map_name = map_name
        self.path_name = None

        mu = sim_conf.mu
        g = sim_conf.g
        self.m = sim_conf.m
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.wheelbase = sim_conf.l_r + sim_conf.l_f
        self.n_beams = sim_conf.n_beams

        self.lookahead = mod_conf.lookahead
        self.vgain = mod_conf.v_gain 
        self.plan_f = mod_conf.plan_frequency #TODO: refactor name to be clear if it is the ratio or frequency

        self.f_max = mu * self.m * g #* safety_f

        self.wpts = None
        self.vs = None
        self.steps = 0

        # TODO: move to agent history class
        self.mod_history = []
        self.d_ref_history = []
        self.reward_history = []
        self.critic_history = []

        self.loop_counter = 0
        self.plan_f = mod_conf.plan_frequency
        self.action = None

        self.aim_pts = []

        try:
            # raise FileNotFoundError
            self._load_csv_track()
        except FileNotFoundError:
            print(f"Problem Loading map - generate new one")

    def _load_csv_track(self):
        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.N = len(track)
        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.wpts
        o_ss = self.ss
        o_vs = self.vs
        new_line = []
        new_ss = []
        new_vs = []
        for i in range(self.N-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                ds = o_ss[i+1] - o_ss[i]
                new_ss.append(o_ss[i] + dz*j*ds)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.wpts = np.array(new_line)
        self.ss = np.array(new_ss)
        self.vs = np.array(new_vs)
        self.N = len(new_line)

    def _get_current_waypoint(self, position, theta):
        # nearest_pt, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, self.wpts)
        nearest_pt, nearest_dist, t, i = self.nearest_pt(position)

        if nearest_dist < self.lookahead:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, self.lookahead, self.wpts, i+t, wrap=True)
            if i2 == None:
                return None
            i = i2
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = self.wpts[i2]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < 20:
            return np.append(self.wpts[i], self.vs[i])


    def act_pp(self, obs):
        """
        Takes the action of a pure pursuit controller

        Args:
            obs: Observation array from toy_f110

        Returns
            action (ndarray(2)): action in form of [steering, speed]
        """
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos, pose_th)

        if lookahead_point is None:
            return 4.0, 0.0

        self.aim_pts.append(lookahead_point)
        # print(f"Lhd Pt: {lookahead_point}")

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.vgain * speed

        # steering_angle = self.limit_inputs(max(speed, obs[3]), steering_angle)

        # return speed, steering_angle
        return np.array([steering_angle, speed])

    def limit_inputs(self, speed, steering_angle):
        max_steer = np.arctan(self.f_max * self.wheelbase / (speed**2 * self.m))
        new_steer = np.clip(steering_angle, -max_steer, max_steer)

        # if max_steer < abs(steering_angle):
            # print(f"Problem, Steering clipped from: {steering_angle} --> {max_steer}")

        return new_steer

    def reset_lap(self):
        # TODO: reset all the history objects 
        self.steps = 0
        self.mod_history = []

    def get_actuation(self, pose_theta, lookahead_point, position):
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead_point[0:2]-position)
        
        speed = lookahead_point[2]
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint_y/self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase/radius)

        return speed, steering_angle

    def nearest_pt(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

    def show_vehicle_history(self):
        # plt.figure(1)
        # plt.clf()
        # plt.title("Mod History")
        # plt.ylim([-1.1, 1.1])
        # plt.plot(self.mod_history)
        # np.save('Vehicles/mod_hist', self.mod_history)
        # # plt.plot(self.d_ref_history)
        # plt.legend(['NN'])


        # plt.figure(3)
        # plt.clf()
        # plt.title('Rewards')
        # plt.ylim([-1.5, 4])
        # plt.plot(self.reward_history, 'x', markersize=12)
        # plt.plot(self.critic_history)
        # plt.pause(0.001)

        plt.figure(5)
        pts = np.array(self.aim_pts)
        plt.plot(pts[:, 0], pts[:, 1])
        plt.pause(0.001)

    def show_wpts(self):
        plt.figure(6)
        pts = np.array(self.wpts)
        plt.plot(pts[:, 0], pts[:, 1])
        plt.pause(0.001)


    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
            steer_ref: pure pursuit steering reference
            speed_ref: pure pursuit speed reference
        """
        steer_ref, speed_ref = self.act_pp(obs)

        cur_v = [obs[3]/self.max_v]
        cur_d = [obs[4]/self.max_steer]
        vr_scale = [(speed_ref)/self.max_v]
        dr_scale = [steer_ref/self.max_steer]

        scan = obs[5:-1]

        nn_obs = np.concatenate([cur_v, cur_d, vr_scale, dr_scale, scan])

        return nn_obs, steer_ref, speed_ref

    def modify_references(self, nn_action, d_ref):
        """
        Modifies the reference quantities for the steering.
        Mutliplies the nn_action with the max steering and then sums with the reference

        Args:
            nn_action: action from neural network in range [-1, 1]
            d_ref: steering reference from PP

        Returns:
            d_new: modified steering reference
        """
        d_max = self.max_steer
        d_phi = d_max * nn_action[0] # rad
        d_new = d_ref + d_phi

        return d_new

    def act(self, obs) -> np.ndarray(2): 
        if self.action is None or self.loop_counter == self.plan_f:
            self.action = self.update_action(obs)
            pp = self.act_pp(obs)
            self.loop_counter = 0
        self.loop_counter += 1
        return self.action

# @njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    # print min_dist_segment, dists[min_dist_segment], projections[min_dist_segment]



class ModVehicleTrain(BaseMod):
    def __init__(self, agent_name, map_name, sim_conf, mod_conf=None, load=False):
        """
        Training vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
            load: if the network should be loaded or recreated.
        """
        if mod_conf is None:
            mod_conf = lib.load_conf("mod_conf")

        BaseMod.__init__(self, agent_name, map_name, sim_conf, mod_conf)

        self.path = 'Vehicles/' + agent_name
        state_space = 4 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        h_size = mod_conf.h
        self.agent.try_load(load, h_size)

        self.reward_fcn = None
        self.state = None
        self.nn_state = None
        self.nn_act = None

        self.t_his = TrainHistory(agent_name)

    def set_reward_fcn(self, r_fcn):
        self.reward_fcn = r_fcn

    def update_action(self, obs):
        nn_obs, steer_ref, speed_ref = self.transform_obs(obs)
        self.add_memory_entry(obs, nn_obs)

        self.state = obs
        nn_action = self.agent.act(nn_obs)
        # nn_action = [0]
        self.nn_act = nn_action

        self.d_ref_history.append(steer_ref)
        self.mod_history.append(self.nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.nn_state = nn_obs

        steering_angle = self.modify_references(self.nn_act, steer_ref)

        return np.array([steering_angle, speed_ref])

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.state is not None:
            reward = self.reward_fcn(self.state, self.action, s_prime, self.nn_act)

            self.t_his.add_step_data(reward)
            mem_entry = (self.nn_state, self.nn_act, nn_s_prime, reward, False)

            self.agent.replay_buffer.add(mem_entry)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime, d, v = self.transform_obs(s_prime)
        reward = self.reward_fcn(self.state, self.action, s_prime, self.nn_act)

        self.t_his.add_step_data(reward)
        self.t_his.lap_done(False)
        # self.t_his.lap_done(True)
        if len(self.t_his.ep_rewards) % 10 == 0:
            self.t_his.print_update()
            self.agent.save(self.path)
        self.state = None
        mem_entry = (self.nn_state, self.nn_act, nn_s_prime, reward, True)

        self.agent.replay_buffer.add(mem_entry)



class ModVehicleTest(BaseMod):
    def __init__(self, agent_name, map_name, sim_conf, mod_conf=None, load=False):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """
        if mod_conf is None:
            mod_conf = lib.load_conf("mod_conf")

        BaseMod.__init__(self, agent_name, map_name, sim_conf, mod_conf)

        self.path = 'Vehicles/' + agent_name
        state_space = 4 
        self.agent = TD3(state_space, 1, 1, agent_name)
        self.agent.load(directory=self.path)

        print(f"NN: {self.agent.actor.type}")

        nn_size = self.agent.actor.l1.in_features
        n_beams = nn_size - 4
        print(f"Agent loaded: {agent_name}")

        self.current_v_ref = None
        self.current_phi_ref = None

    def update_action(self, obs):
        nn_obs, steer_ref, speed_ref = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs, noise=0)
        # nn_action = [0]
        self.nn_act = nn_action

        self.d_ref_history.append(steer_ref)
        self.mod_history.append(self.nn_act[0])
        self.critic_history.append(self.agent.get_critic_value(nn_obs, nn_action))
        self.state_action = [nn_obs, self.nn_act]

        steer_ref = self.modify_references(self.nn_act, steer_ref)

        self.steps += 1

        return np.array([steer_ref, speed_ref])


