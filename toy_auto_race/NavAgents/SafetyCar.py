import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import csv

from toy_auto_race.TrajectoryPlanner import Max_velocity, Max_velocity_conf, MinCurvatureTrajectoryForest, MinCurvatureTrajectory, ObsAvoidTraj
import toy_auto_race.Utils.LibFunctions as lib

from toy_auto_race.speed_utils import calculate_speed
from toy_auto_race.Utils import pure_pursuit_utils

from toy_auto_race.lidar_viz import LidarViz


class SafetyPP:
    def __init__(self, sim_conf) -> None:
        self.name = "Oracle Path Follower"
        self.path_name = None

        # mu = sim_conf.mu
        # g = sim_conf.g
        # self.m = sim_conf.m
        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        # self.f_max = mu * self.m * g #* safety_f

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

        # speed = 4
        speed = calculate_speed(steering_angle)

        return [steering_angle, speed]

    def reset_lap(self):
        self.aim_pts.clear()

    def plan(self, env_map):
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

    def plan_act(self, obs):
        state = obs['state']
        pp_action = self.act_pp(state)

        action = self.run_safety_check(obs, pp_action)

        return action 

    def check_collision(self, scan, current_speed, proposed_steer):
        d_stop = current_speed**2 / (2*self.max_a) * 2

        d_th = np.pi/len(scan)
        bubble = 50 # beams around important one
        zero_pt = len(scan)/2
        steer_idx = int(proposed_steer / d_th + zero_pt)

        idxs = np.arange(steer_idx-bubble, steer_idx+bubble, 1)
        # min_range = np.argmin(scan[idxs])
        min_range = np.min(scan[idxs])

        print(f"MinR: {min_range} --> d_stop: {d_stop}")

        if min_range < d_stop: # collision
            return True
        return False

    def run_safety_check(self, obs, pp_action):
        proposed_steer = pp_action[0]
        scan = obs['scan']
        current_speed = obs['state'][3]

        n_steps = 8
        d_step = self.max_steer/n_steps
        n_steer = proposed_steer
        p_steer = proposed_steer
        new_steer = -1
        if self.check_collision(scan, current_speed, proposed_steer):
            for i in range(n_steps):
                n_steer = max(proposed_steer - i * d_step, -self.max_steer)
                p_steer = min(proposed_steer + i * d_step, self.max_steer)

                if not self.check_collision(scan, current_speed, n_steer):
                    new_steer = n_steer
                    break 
                if not self.check_collision(scan, current_speed, p_steer):
                    new_steer = p_steer
                    break 
            if new_steer == -1:
                action = np.array([proposed_steer, 0])
            else:
                action = np.array([new_steer, current_speed])
            print(f"Old steer: {proposed_steer} --> New Steer: {new_steer} (V: {current_speed})")

            self.vis.add_step(scan, new_steer)
            return action
        else:
            self.vis.add_step(scan, proposed_steer)
            return pp_action

        


