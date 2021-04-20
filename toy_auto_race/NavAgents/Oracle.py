import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import csv

from toy_auto_race.TrajectoryPlanner import Max_velocity, Max_velocity_conf, MinCurvatureTrajectoryForest, MinCurvatureTrajectory, ObsAvoidTraj
import toy_auto_race.Utils.LibFunctions as lib

from toy_auto_race.Utils import pure_pursuit_utils


class OraclePP:
    def __init__(self, sim_conf) -> None:
        self.name = "Oracle Path Follower"
        self.path_name = None

        # mu = sim_conf.mu
        # g = sim_conf.g
        # self.m = sim_conf.m
        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        # self.f_max = mu * self.m * g #* safety_f

        self.v_gain = 1
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

        speed = self.v_gain * speed

        return [steering_angle, speed]

    def reset_lap(self):
        self.aim_pts.clear()


class Oracle(OraclePP):
    def __init__(self, sim_conf):
        OraclePP.__init__(self, sim_conf)
        self.sim_conf = sim_conf # kept for optimisation

    def plan_track(self, env_map):
        pts = env_map.t_pts
        nvecs = env_map.nvecs
        ws = env_map.ws
        check_location = env_map.check_plan_location

        try:
            n_set = ObsAvoidTraj(pts, nvecs, ws, check_location)
        except:
            print(f"Problem with optimisation: defaulting to env_map opti pts")
            self.waypoints = np.concatenate([env_map.wpts, env_map.vs[:, None]], axis=-1)
            return self.waypoints[:, 0:2]
        
        deviation = np.array([nvecs[:, 0] * n_set[:, 0], nvecs[:, 1] * n_set[:, 0]]).T
        wpts = pts + deviation

        vs = Max_velocity_conf(wpts, self.sim_conf, False)
        vs = np.append(vs, vs[-1])

        self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)

        return self.waypoints[:, 0:2]

    def plan_no_obs(self, env_map):
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

        return self.waypoints[:, 0:2]

    def plan_forest(self, env_map):
        # load center pts
        start_x = env_map.start_pose[0]
        start_y = env_map.start_pose[1]
        max_width = env_map.start_pose[0] * 0.8
        length = env_map.end_y - start_y
        num_pts = 50

        y_pts = np.linspace(start_y, length+start_y, num_pts)[:, None]
        t_pts = np.concatenate([np.ones_like(y_pts)*start_x, y_pts], axis=-1)
        
        # set true widths
        # t_pts = set_viable_t_pts(t_pts, max_width, env_map.check_plan_location)
        ws = find_true_widths2(t_pts, max_width, env_map.check_plan_location)

        # Optimise n_set
        N = len(t_pts)
        nvecs = np.concatenate([np.ones((N, 1)), np.zeros((N, 1))], axis=-1)
        n_set = MinCurvatureTrajectory(t_pts, nvecs, ws)

        waypoints = np.concatenate([np.ones((N, 1))*start_x + n_set, y_pts], axis=-1)
        velocity = 1
        vs = np.ones((N, 1)) * velocity

        self.waypoints = np.concatenate([waypoints, vs], axis=-1)

        # self.plot_plan(env_map, t_pts, ws, waypoints)

        self.reset_lap()

        return waypoints

    def plot_plan(self, env_map, t_pts, ws, waypoints=None):
        env_map.render_map(4)

        plt.figure(4)
        env_map.render_wpts(t_pts)
        env_map.render_wpts(waypoints)
        # env_map.render_aim_pts(t_pts)

        rs = t_pts[:, 0] - ws[:, 0]
        r_pts = np.stack([rs, t_pts[:, 1]], axis=1)
        xs, ys = env_map.convert_positions(r_pts)
        plt.plot(xs, ys, 'b')

        ls = t_pts[:, 0] + ws[:, 1]
        l_pts = np.stack([ls, t_pts[:, 1]], axis=1)
        xs, ys = env_map.convert_positions(l_pts)
        plt.plot(xs, ys, 'b')

        plt.show()

    def plan_act(self, obs):
        action = self.act_pp(obs)
        
        return action


def set_viable_t_pts(t_pts, max_width, check_scan_location):
    tx = t_pts[:, 0]
    ty = t_pts[:, 1]

    N = len(t_pts)
    new_t_pts = []
    for i in range(N):
        pt = np.array([tx[i], ty[i]])

        if check_scan_location(pt):
            print(f"Obs in way of pt: {i}")

            for j in np.linspace(0, max_width, 10):
                p_pt = pt + [j, 0]
                n_pt = pt - [j, 0]
                if not check_scan_location(p_pt):
                    new_t_pts.append(p_pt)
                    break
                elif not check_scan_location(n_pt):
                    new_t_pts.append(n_pt)
                    break 
        else:
            new_t_pts.append(pt)

    return np.array(new_t_pts)

def find_true_widths(t_pts, max_width, check_scan_location):
    tx = t_pts[:, 0]
    ty = t_pts[:, 1]

    stp_sze = 0.1
    sf = 1 # safety factor
    N = len(t_pts)
    nws, pws = [], []
    for i in range(N):
        pt = np.array([tx[i], ty[i]])

        j = 0
        s_pt = pt + [j, 0]
        while not check_scan_location(s_pt) and j < max_width:
            j += stp_sze
            s_pt = pt + [j, 0]
        pws.append((j-stp_sze)*sf)

        j = 0
        s_pt = pt - np.array([j, 0])
        while not check_scan_location(s_pt) and j < max_width:
            j += stp_sze
            s_pt = pt - np.array([j, 0])
        nws.append((j-stp_sze)*sf)



    nws, pws = np.array(nws)[:, None], np.array(pws)[:, None]
    ws = np.concatenate([nws, pws], axis=-1)
    print(f"Ws: {ws}")

    return ws


def find_true_widths2(t_pts, max_width, check_scan_location):
    tx = t_pts[:, 0]
    ty = t_pts[:, 1]

    stp_sze = 0.1
    sf = 1 # safety factor
    N = len(t_pts)
    nws, pws = [], []
    for i in range(N):
        pt = np.array([tx[i], ty[i]])

        if not check_scan_location(pt):
            j = stp_sze
            s_pt = pt + [j, 0]
            while not check_scan_location(s_pt) and j < max_width:
                j += stp_sze
                s_pt = pt + [j, 0]
            pws.append(j*sf)

            j = stp_sze
            s_pt = pt - np.array([j, 0])
            while not check_scan_location(s_pt) and j < max_width:
                j += stp_sze
                s_pt = pt - np.array([j, 0])
            nws.append(j*sf)
        else:
            print(f"Obs in way of pt: {i}")

            for j in np.linspace(0, max_width, 10):
                p_pt = pt + [j, 0]
                n_pt = pt - [j, 0]
                if not check_scan_location(p_pt):
                    nws.append(-j*(1))
                    pws.append(max_width)
                    break
                elif not check_scan_location(n_pt):
                    pws.append(-j*(1))
                    nws.append(max_width)
                    break 

    nws, pws = np.array(nws)[:, None], np.array(pws)[:, None]
    ws = np.concatenate([nws, pws], axis=-1)

    return ws
