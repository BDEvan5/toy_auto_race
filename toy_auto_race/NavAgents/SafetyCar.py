import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import csv

from toy_auto_race.TrajectoryPlanner import Max_velocity, Max_velocity_conf, MinCurvatureTrajectoryForest, MinCurvatureTrajectory, ObsAvoidTraj
import toy_auto_race.Utils.LibFunctions as lib

from toy_auto_race.speed_utils import calculate_speed
from toy_auto_race.Utils import pure_pursuit_utils

from toy_auto_race.lidar_viz import LidarViz



#TODO: most of this can be njitted
class ForestFGM:    
    BUBBLE_RADIUS = 250
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 100
    MAX_LIDAR_DIST = 10
    REDUCTION = 200
    
    def __init__(self):
        # used when calculating the angles of the LiDAR data
        # self.vis = LidarViz(1000)
        self.degrees_per_elem = None
        self.name = "Follow the Forest Gap"
        self.n_beams = 1000
    
    def preprocess_lidar(self, ranges):
        self.degrees_per_elem = (180) / len(ranges)
        proc_ranges = np.array(ranges[self.REDUCTION:-self.REDUCTION])
        proc_ranges = ranges
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

        return proc_ranges

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), 'same') 
        averaged_max_gap = averaged_max_gap / self.BEST_POINT_CONV_SIZE
        best = averaged_max_gap.argmax()
        idx = best + start_i

        return idx

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data
        """
        return (range_index - (range_len/2)) * self.degrees_per_elem 

    def process_lidar(self, ranges):
        proc_ranges = self.preprocess_lidar(ranges)
        closest = proc_ranges.argmin()

        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges)-1
        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = find_max_gap(proc_ranges)

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = self.get_angle(best, len(proc_ranges))
        # self.vis.add_step(proc_ranges, steering_angle)

        return steering_angle

    def plan_act(self, obs):
        scan = obs['scan']
        ranges = np.array(scan, dtype=np.float)

        steering_angle = self.process_lidar(ranges)
        steering_angle = steering_angle * np.pi / 180

        # speed = 4
        speed = calculate_speed(steering_angle)

        action = np.array([steering_angle, speed])

        return action

    def reset_lap(self):
        pass

# @njit
def find_max_gap(free_space_ranges):
    """ Return the start index & end index of the max gap in free_space_ranges
        free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
    """
    # mask the bubble
    masked = np.ma.masked_where(free_space_ranges==0, free_space_ranges)
    # get a slice for each contigous sequence of non-bubble data
    slices = np.ma.notmasked_contiguous(masked)
    if len(slices) == 0:
        return 0, len(free_space_ranges)
    max_len = slices[0].stop - slices[0].start
    chosen_slice = slices[0]
    # I think we will only ever have a maximum of 2 slices but will handle an
    # indefinitely sized list for portablility
    for sl in slices[1:]:
        sl_len = sl.stop - sl.start
        if sl_len > max_len:
            max_len = sl_len
            chosen_slice = sl
    return chosen_slice.start, chosen_slice.stop


#TODO: consider turning FGM into a simple function that can be njitted and then called.
# possibly have param data struct
#TODO: remove random np functions and use std calls for njit


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
        self.old_steers = []
        self.new_steers = []

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

        # print(f"MinR: {min_range} --> d_stop: {d_stop}")

        if min_range < d_stop: # collision
            return True
        return False

    def run_safety_check(self, obs, pp_action):
        scan = obs['scan']
        state = obs['state']

        if self.check_collision(scan, state[3], pp_action[0]):
            action = self.prevent_collision(scan, state, pp_action)

        else:
            action = pp_action

        self.vis.add_step(scan, action[0])
        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        return action

    def prevent_collision(self, scan, state, proposed_action):
        """
        This function is called if a collision is detected. It takes the current scan, state and pp action and returns an action that will ensure no crash. 
        """
        proposed_steer = proposed_action[0]
        current_speed = state[3]
        n_steps = 8
        d_step = self.max_steer/n_steps
        n_steer = proposed_steer
        p_steer = proposed_steer
        new_steer = -1
        for i in range(n_steps):
            n_steer = max(proposed_steer - i * d_step, -self.max_steer)
            p_steer = min(proposed_steer + i * d_step, self.max_steer)

            if not self.check_collision(scan, current_speed, n_steer):
                new_steer = n_steer
                break 
            if not self.check_collision(scan, current_speed, p_steer):
                new_steer = p_steer
                break 

        # no safe option
        # the best we can do is follow the gap
        if new_steer == -1:
            print(f"No safe option")

            d_th = np.pi/len(scan)

            bubble = 125 # beams around center that are driveable
            zero_pt = int(len(scan)/2)

            idxs = np.arange(zero_pt-bubble, zero_pt+bubble, 1)
            d_scan = scan[idxs]
            fgm_bubble = 20
            min_range_idx = np.argmin(d_scan)
            min_range_idx = np.clip(min_range_idx, 0, bubble*2-1)
            zero_idxs = np.arange(max(min_range_idx-fgm_bubble, 0), min(min_range_idx+fgm_bubble, bubble*2), 1)
            d_scan[zero_idxs] = np.zeros_like(d_scan[zero_idxs])
            d_scan = np.convolve(d_scan, np.ones(20), 'same') / 20
            max_range_idx = np.argmax(d_scan)

            new_steer = (max_range_idx - bubble) * d_th
            # new_steer = (max_range_idx + zero_pt - bubble) * d_th

        new_speed = calculate_speed(new_steer)
        action = np.array([new_steer, new_speed])
        print(f"Old steer: {proposed_steer:.4f} --> New Steer: {new_steer:.4f} (V: {current_speed:.4f})")

        return action


    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()

    def show_history(self, wait=False):
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


