
import numpy as np
from numba import njit, jit
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
        self.max_d_dot = sim_conf.max_d_dot

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

        # pp_action[1] = max(pp_action[1], state[3])
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
        
    def show_lidar(self):
        pass

    def run_safety_check(self, obs, pp_action):

        # check if current corridor is safe
        scan = obs['scan']
        scan *= 0.95
        state = obs['state']

        v = state[3]
        d = state[4]
        dw_vs, dw_ds = build_dynamic_window(v, d, self.max_v, self.max_steer, self.max_a, self.max_d_dot, 0.1)

        valid_window, end_pts = check_dw(dw_vs, dw_ds, self.max_a, scan)

        self.plot_valid_window(dw_vs, dw_ds, valid_window)
        self.plot_lidar_scan(scan, end_pts)

        return pp_action

    def plot_valid_window(self, dw_vs, dw_ds, valid_window):
        plt.figure(1)
        plt.clf()
        plt.title("Valid window")

        sf = 1.1
        plt.xlim([dw_ds[0]*sf, dw_ds[-1]*sf])
        plt.ylim([dw_vs[0]/sf, dw_vs[-1]*sf])

        for i, v in enumerate(dw_vs):
            for j, d in enumerate(dw_ds):
                if valid_window[i, j]:
                    plt.plot(d, v, '+', color='green', markersize=18)
                else:
                    plt.plot(d, v, '+', color='red', markersize=18)

        # plt.show()
        plt.pause(0.0001)


    def plot_lidar_scan(self, scan, end_pts):
        plt.figure(2)
        plt.clf()
        plt.title('Lidar Scan')
        xs, ys = convert_scan_xy(scan)

        plt.ylim([0, 10])
        plt.plot(xs, ys, '-+')

        xs = end_pts[:, :, 0].flatten()
        ys = end_pts[:, :, 1].flatten()
        for x, y in zip(xs, ys):
            x_p = [0, x]
            y_p = [0, y]
            plt.plot(x_p, y_p, '--')

        plt.pause(0.0001)

@njit(cache=True) 
def build_dynamic_window(v, delta, max_v, max_steer, max_a, max_d_dot, dt):
    uvb = min(max_v, v+dt*max_a)
    lvb = max(0, v-max_a*dt)
    udb = min(max_steer, delta+dt*max_d_dot)
    ldb = max(-max_steer, delta-dt*max_d_dot)

    n_v_pts = 20 
    n_delta_pts = 50 
    
    vs = np.linspace(lvb, uvb, n_v_pts)
    ds = np.linspace(ldb, udb, n_delta_pts)

    return vs, ds

# @jit(cache=True)
def check_dw(dw_vs, dw_ds, max_a, scan):

    dt = 0.1
    n_steps = 10 
    valids = np.empty((len(dw_vs), len(dw_ds)))
    end_pts = np.empty((len(dw_vs), len(dw_ds), 2))
    for i, v in enumerate(dw_vs):
        for j, d in enumerate(dw_ds):
            t_xs, t_ys = predict_trajectory(v, d, n_steps, dt)
            safe = check_trajcetory_safe(t_xs, t_ys, scan)

            valids[i, j] = safe 
            end_pts[i, j, 0] = t_xs[-1]
            end_pts[i, j, 1] = t_ys[-1]

    return valids, end_pts

# @njit(cache=True)
def predict_trajectory(v, d, n_steps, dt):
    L = 0.33
    xs = np.empty(n_steps)
    ys = np.empty(n_steps)
    theta = 0 
    xs[0] = 0
    ys[0] = 0
    for i in range(1, n_steps):
        theta += dt * v / L * np.tan(d)
        xs[i] = xs[i-1] + v * np.sin(theta) * dt 
        ys[i] = ys[i-1] + v * np.cos(theta) * dt 
    
    return xs, ys

@jit(cache=True)
def check_trajcetory_safe(t_xs, t_ys, scan):
    for x, y in zip(t_xs, t_ys):
        if not check_pt_safe(x, y, scan):
            return False 
    return True

@njit(cache=True)
def check_pt_safe(x, y, scan):
    angle = np.arctan2(x, y) # range -90, 90
    idx = int(angle / (np.pi / 999) + 500)
    ld = (x**2 + y**2)**0.5

    idxs = scan[idx-1:idx+3]
    min_idx = np.min(idxs)
    if ld > min_idx: # average four scans
        return False 
    return True  # ld shorter than scan.



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








