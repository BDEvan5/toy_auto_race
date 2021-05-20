
import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import csv

from numpy.core.numerictypes import maximum_sctype

import toy_auto_race.Utils.LibFunctions as lib

from toy_auto_race.speed_utils import calculate_speed, calculate_safe_speed
from toy_auto_race.Utils import pure_pursuit_utils

from toy_auto_race.lidar_viz import *




class SafetyPP:
    def __init__(self, sim_conf) -> None:
        self.name = "Safety Car"
        self.path_name = None

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.max_steer = sim_conf.max_steer
        self.max_v = sim_conf.max_v

        self.v_gain = 0.5
        self.lookahead = 1.6
        self.max_reacquire = 20

        self.waypoints = None
        self.vs = None

        self.aim_pts = []

    def _get_current_waypoint(self, position):
        lookahead_distance = self.lookahead
    
        wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.waypoints[i, 2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, 2])
        else:
            return None

    def act_pp(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos)

        self.aim_pts.append(lookahead_point[0:2])

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(pose_th, lookahead_point, pos, self.lookahead, self.wheelbase)
        steering_angle = np.clip(steering_angle, -self.max_steer, self.max_steer)


        # speed = 4
        speed = calculate_speed(steering_angle)

        return np.array([steering_angle, speed])

    def reset_lap(self):
        self.aim_pts.clear()

    def plan(self, env_map):
        if self.waypoints is None:
            track = []
            filename = 'maps/' + env_map.map_name + "_opti.csv"
            with open(filename, 'r') as csvfile:
                csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
                for lines in csvFile:  
                    track.append(lines)

            track = np.array(track)
            print(f"Track Loaded: {filename}")

            wpts = track[:, 1:3]
            vs = track[:, 5]

            self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)
            self.expand_wpts()

            return self.waypoints[:, 0:2]

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints[:, 0:2]
        # o_ss = self.ss
        o_vs = self.waypoints[:, 2]
        new_line = []
        # new_ss = []
        new_vs = []
        for i in range(len(self.waypoints)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                # ds = o_ss[i+1] - o_ss[i]
                # new_ss.append(o_ss[i] + dz*j*ds)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        wpts = np.array(new_line)
        # self.ss = np.array(new_ss)
        vs = np.array(new_vs)
        self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)


class SafetyCar(SafetyPP):
    def __init__(self, sim_conf):
        SafetyPP.__init__(self, sim_conf)
        self.sim_conf = sim_conf # kept for optimisation
        self.n_beams = 1000

        safety_f = 0.9
        self.max_a = sim_conf.max_a * safety_f
        self.max_steer = sim_conf.max_steer

        self.vis = LidarViz(1000)
        self.old_steers = []
        self.new_steers = []

        self.last_scan = None
        self.new_action = None
        self.col_vals = None
        self.o_col_vals = None
        self.o_action = None

        self.fov = np.pi
        self.dth = self.fov / (self.n_beams-1)
        self.center_idx = int(self.n_beams/2)

        self.angles = np.empty(self.n_beams)
        for i in range(self.n_beams):
            self.angles[i] =  self.fov/(self.n_beams-1) * i

    def plan_act(self, obs):
        state = obs['state']
        pp_action = self.act_pp(state)

        pp_action[1] = max(pp_action[1], state[3])
        action = self.run_safety_check(obs, pp_action)

        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        return action 


    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()

    def show_history(self, wait=False):
        # plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

        plt.figure(5)
        plt.clf()
        plt.plot(self.old_steers)
        plt.plot(self.new_steers)
        plt.legend(['Old', 'New'])
        plt.title('Old and New Steering')
        plt.ylim([-0.5, 0.5])

        plt.pause(0.0001)
        if wait:
            plt.show()

    def run_safety_check(self, obs, pp_action):

        # check if current corridor is safe
        scan = obs['scan']
        scan *= 0.95
        state = obs['state']

        deltas, vs = self.calculate_envelope(scan)

        action = self.modify_action(deltas, vs, pp_action, state)
        while action is None: # if it can't find an angle, reduce speed and try again.
            # print(f"No action found: retry")
            pp_action[1] = pp_action[1] * 0.95
            action = self.modify_action(deltas, vs, pp_action, state)

        # self.plot_lidar_line(scan, deltas, vs, pp_action, action, state)

        # if action[0] != pp_action[0]:
        #     plt.show()

        return action

    def modify_action(self, deltas, vs, pp_action, state):
        new_action = None
        if not check_action_safe(pp_action, deltas, vs):
            # action must be Changed
            n_search = 20 
            d_delta = self.max_steer / n_search 

            # add extra condition for if it is equal to search both spaces

            if state[4] == pp_action[0]:
                # print(f"Searching both spaces")
                for i in range(1, n_search):
                    p_act = np.array([pp_action[0] + d_delta * i, pp_action[1]])
                    if check_action_safe(p_act, deltas, vs):
                        new_action = p_act
                        break
                    n_act = np.array([pp_action[0] - d_delta * i, pp_action[1]]) 
                    if check_action_safe(n_act, deltas, vs):
                        new_action = n_act
                        break
            elif state[4] > pp_action[0]:
                # print(f"Searching positive delta space")
                for i in range(1, n_search):
                    p_act = np.array([pp_action[0] + d_delta * i, pp_action[1]])
                    if check_action_safe(p_act, deltas, vs):
                        new_action = p_act
                        break
            else:
                # print(f"Searching negative delta space")
                for i in range(1, n_search):
                    n_act = np.array([pp_action[0] - d_delta * i, pp_action[1]]) 
                    if check_action_safe(n_act, deltas, vs):
                        new_action = n_act
                        break
        
            # print(f"Action unsafe: ({pp_action}) --> modify to ({new_action})")
            return new_action
        return pp_action
                

    def calculate_envelope(self, scan):
        # angles = get_angles()
        # l_d = 4
        # deltas = np.arctan(2 * self.wheelbase * np.sin(angles)/l_d)
        # deltas = np.clip(deltas, -self.max_steer, self.max_steer)

        # vs = np.sqrt(scan*2*self.max_a)
        # vs = np.clip(vs, 0, self.max_v)

        # return deltas, vs

        return create_envelope(scan)

    def plot_lidar_line(self, scan, deltas, vs, pp_action, action, state, fig_n=1):
        xs, ys = convert_scan_xy(scan)

        plt.figure(fig_n)
        plt.clf()
        plt.ylim([0, 10])
        plt.title("Lidar line")
        plt.plot(xs, ys, '-+')

        delta = action[0]
        for i in range(50):
            l_d = i * 5
            alpha = np.arcsin(l_d * np.tan(delta) / (2*self.wheelbase))
            x, y = polar_to_xy(l_d, alpha)
            if check_scan_location(xs, ys, x, y) or i == 50 -1:
                break 
        
        xss = [0, x]
        yss = [0, y]
        plt.plot(xss, yss)


        xs, ys = get_feasible_projection()
        plt.plot(xs, ys, '--')

        plt.pause(0.0001)

        plt.figure(fig_n+1)
        plt.clf()
        plt.xlim([-0.4, 0.4])
        plt.ylim([0, 7.2])
        plt.title("Control Envelope")
        plt.plot(deltas, vs)

        # pp action 
        plt.plot(pp_action[0], pp_action[1], '+', markersize=18, color='b')
        
        # normal action 
        plt.plot(action[0], action[1], '+', markersize=18, color='r')

        # state
        speed = state[3]
        plt.plot(state[4], speed, '*', markersize=18)

        w = 0.32
        h = 0.8
        x = state[4] - w
        y = speed - h
        rectangle = plt.Rectangle((x, y), 2*w, 2*h, fc=(1, 1, 1), ec='blue')
        plt.gca().add_patch(rectangle)

        plt.legend(['Safety env', 'pp', 'action', 'state', 'feasible'])

        # plt.plot()


        plt.pause(0.1)
        # plt.show()

    def show_lidar(self, wait=False):
        # plot_lidar_col_vals(self.last_scan, self.col_vals, self.o_action[0], False, fig_n=2)
        # plt.title("Old Lidar")


        # plot_lidar_col_vals(self.last_scan, self.col_vals, self.new_action[0], False, fig_n=5)
        # plt.title("New Lidar action")

        if wait:
            plt.show()


@njit(cache=True)
def convert_angle_idx(angle):
    return int((angle + np.pi/2) / np.pi)

@njit(cache=True)
def convert_idx_angle(idx):
    return (idx - 500) * np.pi / 999

@njit(cache=True)
def get_angles(n_beams=1000, fov=np.pi):
    angles = np.empty(n_beams)
    for i in range(n_beams):
        angles[i] = -fov/2 + fov/(n_beams-1) * i
    return angles

@njit(cache=True)
def get_trigs(n_beams, fov=np.pi):
    angles = np.empty(n_beams)
    for i in range(n_beams):
        angles[i] = -fov/2 + fov/(n_beams-1) * i
    sines = np.sin(angles)
    cosines = np.cos(angles)

    return sines, cosines

@njit(cache=True)
def convert_scan_xy(scan, fov=np.pi):
    sines, cosines = get_trigs(len(scan))
    xs = scan * sines
    ys = scan * cosines    
    return xs, ys

# @njit(cache=True)
def get_feasible_projection():
    n_pts = 50
    delta_max = 0.4 
    wheelbase = 0.33
    x_max = 1

    xs = np.empty(n_pts)
    thetas = np.empty(n_pts)

    alpha = np.pi/4
    l_d = 1.1 # max value
    y_max = l_d * np.cos(alpha)

    ys = np.linspace(0, y_max, n_pts)

    thetas[0] = 0
    for i in range(1, n_pts):
        thetas[i] = thetas[i-1] +  np.tan(delta_max) * (ys[i] - ys[i-1]) / (wheelbase * np.cos(thetas[i-1]))

    xs[0] = 0 # eliminate t from equations
    for i in range(1, n_pts):
        xs[i] = xs[i-1] + (ys[i] - ys[i-1]) * np.tan(thetas[i-1]) 
    
    xs[-1] = x_max 

    xs = np.hstack([-xs[::-1], xs])
    ys = np.hstack([ys[::-1], ys])

    return xs, ys

@njit(cache=True)
def find_v_idx(deltas, action, vs):
    action[0] = min(action[0], deltas[-5])
    action[0] = max(action[0], deltas[5])
    for i in range(len(deltas)):
        if action[0] < deltas[i]:
            idx = i 
            break 
    v_idx = np.min(vs[idx-2:idx+4]) # take min value to check arround current value as well

    return v_idx

@njit(cache=True)
def check_action_safe(pp_action, deltas, vs):
    # pp_action[0] *= 0.95 # makes sure it is within range
    v_idx = find_v_idx(deltas, pp_action, vs)

    eps = 0.5
    if v_idx - pp_action[1] < eps:
        return False 

    # add check for crossing verticle lines
    # if 

    return True

# @njit(cache=True)
# def check_scan_location(xs, ys, x, y):
#     for i in range(len(xs)):
#         if x < xs[i]:
#             idx = i 
#             break 
#     y_idx = np.mean(ys[idx:idx+2])
#     if y > y_idx:
#         return True # occupied
#     return False  # empty

# @njit(cache=True)
def check_scan_location(xs, ys, x, y):
    distances = np.abs(xs - np.ones_like(xs) * x)   
    idxs = np.argpartition(distances, 5)
    idxs = idxs[:5] 

    y_min = np.min(ys[idxs])

    if y > y_min:
        return True # occupied
    return False  # empty

@njit(cache=True)
def polar_to_xy(r, theta):
    x = r * np.sin(theta)
    y = r * np.cos(theta) 
    return x, y

# @njit
def create_envelope(scan):
    n_ds = 500 
    max_steer = 0.4 
    max_a = 4
    max_v = 5
    L = 0.33
    deltas = np.linspace(-max_steer, max_steer, n_ds)
    xs, ys = convert_scan_xy(scan)

    n_search = 100
    s = 10 / n_search # resolution 5cm, max len, 5m
    #TODO: in future use variable s with dt. 
    l_ds = np.zeros_like(deltas)
    alphas = np.zeros_like(deltas)

    for j, delta in enumerate(deltas):
        for i in range(n_search):
            l_d = i * s 
            if l_d * np.tan(delta)*0.98 >= 2*L:
                l_d = 2*L / np.tan(delta)
            alpha = np.arcsin(l_d * np.tan(delta) / (2*L))
            x, y = polar_to_xy(l_d, alpha)
            if check_scan_location(xs, ys, x, y) or i == n_search -1:
                l_ds[j] = l_d 
                alphas[j] = alpha
                break 

    # plt.figure(3)
    # plt.clf()
    # plt.title(' lengths for pp')
    # n_beams = len(alphas)

    # for i in range(n_beams):
    #     xs = [0, np.sin(alphas[i]) * l_ds[i]]
    #     ys = [0, np.cos(alphas[i]) * l_ds[i]]
    #     plt.plot(xs, ys, 'b')

    # plt.pause(0.0001)

    vs = np.sqrt(l_ds*2*max_a)
    vs = np.clip(vs, 0, max_v)

    return deltas, vs

        
