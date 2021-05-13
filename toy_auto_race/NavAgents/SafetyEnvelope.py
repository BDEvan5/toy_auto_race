
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

        # mu = sim_conf.mu
        # g = sim_conf.g
        # self.m = sim_conf.m
        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        # self.f_max = mu * self.m * g #* safety_f
        self.max_steer = sim_conf.max_steer

        self.v_gain = 0.5
        self.lookahead = 0.8
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

        return [steering_angle, speed]

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

        action = self.run_safety_check(obs, pp_action)

        return action 


    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()

    def show_history(self, wait=False):
        # plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

        plt.figure(1)
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
        state = obs['state']

        o_proj_state = get_projected_state(state, pp_action)
        # proj_state = [theta, velocity]

        if self.check_coridor_free(scan, o_proj_state):
            return pp_action
        
        self.o_col_vals = self.col_vals
        self.o_action = pp_action
        self.last_scan = scan
        # unsafe
        
        action = self.find_safe_action(scan, state, pp_action)
        self.new_action = action

        return action 

        # if unsafe:
            # find next best option
            # select a lhd distance, and check if there is enough width.

    def check_coridor_free(self, scan, proj_state):
        w = 0.15 #width each side
        max_d_stop = 4 #TODO: move to config file
        speed = proj_state[1]
        angle = proj_state[0]

        d_stop = speed**2 / (2*self.max_a) * 2
        d_stop = min(max_d_stop, d_stop)

        collision_values = w / abs(np.cos(self.angles - np.ones_like(self.angles)*angle))
        collision_values = np.clip(collision_values, 0, d_stop)
        self.col_vals = collision_values

        output = np.greater(collision_values, scan)
        if output.any():
            # plot_lidar_col_vals(scan, collision_values, angle, False, fig_n=2)
            return False

        return True

    def find_safe_action(self, scan, state, o_action):
        new_steer = self.find_safe_steer(scan, state, o_action)

        while new_steer is None:  # if no steer found
            o_action[1] = o_action[1] / 2
            new_steer = self.find_safe_steer(scan, state, o_action)

            if o_action[1] < 1:
                print(f"Admit defeat")
                return np.array([0, 0])

        action = np.array([new_steer, o_action[1]])
        
        return action

    def find_safe_steer(self, scan, state, o_action):
        n_searches = 10
        step = 0.08
        for i in range(1, n_searches):
            p_angle = min(o_action[0] + i * step, self.max_steer)
            proj_state = get_projected_state(state, np.array([o_action[1], p_angle]))
            if self.check_coridor_free(scan, proj_state):
               return p_angle

            n_angle = max(o_action[0] - i * step, -self.max_steer)
            proj_state = get_projected_state(state, np.array([o_action[1], n_angle]))
            if self.check_coridor_free(scan, proj_state):
               return n_angle
        return None

    def show_lidar(self, wait=False):
        plot_lidar_col_vals(self.last_scan, self.col_vals, self.o_action[0], False, fig_n=2)
        plt.title("Old Lidar")


        plot_lidar_col_vals(self.last_scan, self.col_vals, self.new_action[0], False, fig_n=5)
        plt.title("New Lidar action")

        if wait:
            plt.show()

@njit(cache=True)
def convert_angle_idx(angle):
    return int((angle + np.pi/2) / np.pi)

@njit(cache=True)
def convert_idx_angle(idx):
    return (idx - 500) * np.pi / 999

@njit(cache=True)
def get_projected_state(state, action):
    for i in range(10):
        u = control_system(state, action)
        state = update_kinematic_state(state, u, 0.01)
    return state[2:4]    

@njit(cache=True)
def update_kinematic_state(x, u, dt, whlb=0.33, max_steer=0.4, max_v=7):
    """
    Updates the kinematic state according to bicycle model

    Args:
        X: State, x, y, theta, velocity, steering
        u: control action, d_dot, a
    Returns
        new_state: updated state of vehicle
    """
    dx = np.array([x[3]*np.sin(x[2]), # x
                x[3]*np.cos(x[2]), # y
                x[3]/whlb * np.tan(x[4]), # theta
                u[1], # velocity
                u[0]]) # steering

    new_state = x + dx * dt 

    # check limits
    new_state[4] = min(new_state[4], max_steer)
    new_state[4] = max(new_state[4], -max_steer)
    new_state[3] = min(new_state[3], max_v)

    return new_state

@njit(cache=True)
def control_system(state, action, max_v=7, max_steer=0.4, max_a=8.5, max_d_dot=3.2):
    """
    Generates acceleration and steering velocity commands to follow a reference
    Note: the controller gains are hand tuned in the fcn

    Args:
        v_ref: the reference velocity to be followed
        d_ref: reference steering to be followed

    Returns:
        a: acceleration
        d_dot: the change in delta = steering velocity
    """
    # clip action
    v_ref = min(action[1], max_v)
    d_ref = max(action[0], -max_steer)
    d_ref = min(action[0], max_steer)

    kp_a = 10
    a = (v_ref-state[3])*kp_a
    
    kp_delta = 40
    d_dot = (d_ref-state[4])*kp_delta

    # clip actions
    a = min(a, max_a)
    a = max(a, -max_a)
    d_dot = min(d_dot, max_d_dot)
    d_dot = max(d_dot, -max_d_dot)
    
    u = np.array([d_dot, a])

    return u