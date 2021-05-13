import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import csv

from numpy.core.numerictypes import maximum_sctype

import toy_auto_race.Utils.LibFunctions as lib

from toy_auto_race.speed_utils import calculate_speed, calculate_safe_speed
from toy_auto_race.Utils import pure_pursuit_utils

from toy_auto_race.lidar_viz import *



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
        self.action = None
        self.col_vals = None

        self.fov = np.pi
        self.dth = self.fov / (self.n_beams-1)
        self.center_idx = int(self.n_beams/2)

        self.fgm = ForestFGM()

        self.angles = np.empty(self.n_beams)
        for i in range(self.n_beams):
            self.angles[i] =  self.fov/(self.n_beams-1) * i

    def plan_act(self, obs):
        state = obs['state']
        pp_action = self.act_pp(state)

        action = self.run_safety_check(obs, pp_action)

        return action 

    def check_corridor_clear(self, scan, speed, angle):
        w = 0.15 #width each side
        max_d_stop = 5

        d_stop = speed**2 / (2*self.max_a) * 3
        d_stop = min(max_d_stop, d_stop)

        collision_values = np.empty(len(scan))
        for i in range(len(scan)):
            collision_values[i] = min(w / abs(np.cos(self.angles[i] - angle)), d_stop)
        self.col_vals = collision_values

        for i in range(self.n_beams):
            if scan[i] < collision_values[i]:
                return False 

        return True

    def run_safety_check(self, obs, pp_action):
        scan = obs['scan']
        state = obs['state']

        self.last_scan = scan

        if not self.check_corridor_clear(scan, state[3], pp_action[0]):

            action = self.prevent_collision(scan, state, pp_action)

        else:
            action = pp_action

        self.vis.add_step(scan, action[0])
        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        self.action = action
        # plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

        return action

    def prevent_collision(self, scan, state, proposed_action):
        steering_angle = self.fgm.process_lidar(scan)
        steering_angle = steering_angle * np.pi / 180

        speed = calculate_speed(steering_angle)

        action = np.array([steering_angle, speed])

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



class SafetyCar5(SafetyPP):
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
        self.action = None
        self.col_vals = None

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

    def check_corridor_clear(self, scan, speed, angle):
        w = 0.15 #width each side
        max_d_stop = 4

        d_stop = speed**2 / (2*self.max_a) * 2
        d_stop = min(max_d_stop, d_stop)

        collision_values = np.empty(len(scan))
        for i in range(len(scan)):
            collision_values[i] = min(w / abs(np.cos(self.angles[i] - angle)), d_stop)
        self.col_vals = collision_values

        for i in range(self.n_beams):
            if scan[i] < collision_values[i]:
                return False 

        return True

    def run_safety_check(self, obs, pp_action):
        scan = obs['scan']
        state = obs['state']

        self.last_scan = scan

        if not self.check_corridor_clear(scan, state[3], pp_action[0]):

            action = self.prevent_collision(scan, state, pp_action)

        else:
            action = pp_action

        self.vis.add_step(scan, action[0])
        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        self.action = action
        # plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

        return action

    # def find_widths(self, scan, speed, angle):
        

    def find_new_angle(self, scan, speed, old_angle):
        n_searches = 10
        step = 0.08
        for i in range(1, n_searches):
            p_angle = min(old_angle + i * step, self.max_steer)
            if self.check_corridor_clear(scan, speed, p_angle):
               return p_angle
            # plot_lidar_col_vals(scan, self.col_vals, wait=False)
            n_angle = max(old_angle - i * step, -self.max_steer)
            if self.check_corridor_clear(scan, speed, n_angle):
               return n_angle
            # plot_lidar_col_vals(scan, self.col_vals, wait=False)
        return None

    def prevent_collision(self, scan, state, proposed_action):
        old_angle = proposed_action[0]
        speed = state[3]

        new_angle = None
        while speed > 1:
            new_angle = self.find_new_angle(scan, speed, old_angle)

            if new_angle is not None:
                speed = calculate_speed(new_angle) 
                break 
        
            print(f"Problem: no angle found")
            speed = speed / 2
        
        if new_angle is None: # search didn't break
            print(f"Accept defeat and die")
            return np.array([0, 0]) # accept defeat

        if abs(new_angle) > self.max_steer: # this is impossible since I only search in possible steering places
            # I think this is meant to be for if the steering angle is too big for the current speed, then the speed should be reduced.
            speed = speed / 2
            new_angle = np.clip(new_angle, -self.max_steer, self.max_steer)
            action = np.array([new_angle, speed])
            print(f"Angle was too fast. Old: {old_angle} --> New: {new_angle}")

            return action

        print(f"Angle Changed: Old: {old_angle} --> New: {new_angle}")

        action = np.array([new_angle, speed])

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




class SafetyCar4(SafetyPP):
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
        self.action = None
        self.col_vals = None

        self.fov = np.pi
        self.dth = self.fov / (self.n_beams-1)
        self.center_idx = int(self.n_beams/2)

    def plan_act(self, obs):
        state = obs['state']
        pp_action = self.act_pp(state)

        action = self.run_safety_check(obs, pp_action)

        return action 

    def check_collision(self, scan, current_speed, proposed_steer):
        d_stop = current_speed**2 / (2*self.max_a) * 2

        n_beams = len(scan)
        fov = np.pi
        angles = np.empty(n_beams)
        for i in range(n_beams):
            angles[i] =  fov/(n_beams-1) * i
        car_width = 0.15
        collision_values = np.empty(len(scan))
        # sines, cosines = get_trigs(len(scan))
        for i in range(len(scan)):
            collision_values[i] = min(car_width / abs(np.cos(angles[i] + proposed_steer)), d_stop)
            # note that it is capped at d_stop because we would never need more than that.
        self.col_vals = collision_values
        for i in range(n_beams):
            if scan[i] < collision_values[i]:
                print(f"Col detect i: {i} > scan: {scan[i]:.4f} --> col val: {collision_values[i]:.4f}")
                return True

        # plot_lidar(collision_values, wait=True)

        # if min_range < d_stop: # collision
        #     return True
        return False

    def run_safety_check(self, obs, pp_action):
        scan = obs['scan']
        state = obs['state']

        self.last_scan = scan

        # action = self.prevent_collision(scan, state, pp_action)
        if self.check_collision(scan, state[3], pp_action[0]):
            action = self.prevent_collision(scan, state, pp_action)

        else:
            action = pp_action

        self.vis.add_step(scan, action[0])
        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        self.action = action

        return action

    def prevent_collision(self, scan, state, proposed_action):
        """
        This function is called if a collision is detected. It takes the current scan, state and pp action and returns an action that will ensure no crash. 
        """
        new_steer = self.calculate_best_steer(scan, state)

        new_steer = np.clip(new_steer, -self.max_steer, self.max_steer)
        speed = calculate_speed(new_steer)
        action = np.array([new_steer, speed])

        # if not new_steer:
        #     print(f"Return zero speed")
        #     action = np.array([0, 0])

        return action

    def calculate_best_steer(self, scan, state):
        n_steer = 9
        step = 2 * self.max_steer / (n_steer-1)
        steer_list = [i*step-self.max_steer for i in range(n_steer)]
        steer_list = np.array(steer_list)

        d_stop = state[3]**2 / (2*self.max_a) * 2

        gap_widths = []

        for steer in steer_list:
            # find the max clear width
            n_search = 10
            step_size = 0.05
            for i in range(1, n_search):
                search_w = i * step_size
                if not self.check_w_clear(scan, steer, search_w, d_stop):
                    # the gap isn't clear
                    # take the previous width
                    max_width = (i-1) * step_size
                    break
            gap_widths.append(max_width)

        print(f"Widths: {gap_widths}")

        best_idx = np.argmax(gap_widths)
        best_steer = steer_list[best_idx]

        return best_steer

    def check_w_clear(self, scan, angle, w, d_stop):
        # d_max = max(scan)

        angles = np.empty(self.n_beams)
        for i in range(self.n_beams):
            angles[i] =  self.fov/(self.n_beams-1) * i

        collision_values = np.empty(len(scan))
        for i in range(len(scan)):
            collision_values[i] = min(w / abs(np.cos(angles[i] + abs(angle))), d_stop)
        self.col_vals = collision_values

        for i in range(self.n_beams):
            if scan[i] < collision_values[i]:
                return False 

        return True


    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()

    def show_history(self, wait=False):
        plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

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


class SafetyCar3(SafetyPP):
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
        self.action = None
        self.col_vals = None

        self.fov = np.pi
        self.dth = self.fov / (self.n_beams-1)
        self.center_idx = int(self.n_beams/2)

    def plan_act(self, obs):
        state = obs['state']
        pp_action = self.act_pp(state)

        action = self.run_safety_check(obs, pp_action)

        return action 

    def check_collision(self, scan, current_speed, proposed_steer):
        d_stop = current_speed**2 / (2*self.max_a) * 2

        # d_th = np.pi/len(scan)
        # bubble = 25 # beams around important one
        # zero_pt = len(scan)/2
        # steer_idx = int(proposed_steer / d_th + zero_pt)

        # idxs = np.arange(steer_idx-bubble, steer_idx+bubble, 1)
        # min_range = np.min(scan[idxs])

        # print(f"MinR: {min_range} --> d_stop: {d_stop}")

        n_beams = len(scan)
        fov = np.pi
        angles = np.empty(n_beams)
        for i in range(n_beams):
            angles[i] =  fov/(n_beams-1) * i
        car_width = 0.15
        collision_values = np.empty(len(scan))
        # sines, cosines = get_trigs(len(scan))
        for i in range(len(scan)):
            collision_values[i] = min(car_width / abs(np.cos(angles[i] + proposed_steer)), d_stop)
            # note that it is capped at d_stop because we would never need more than that.
        self.col_vals = collision_values
        for i in range(n_beams):
            if scan[i] < collision_values[i]:
                print(f"Col detect i: {i} > scan: {scan[i]:.4f} --> col val: {collision_values[i]:.4f}")
                return True

        # plot_lidar(collision_values, wait=True)

        # if min_range < d_stop: # collision
        #     return True
        return False

    def run_safety_check(self, obs, pp_action):
        scan = obs['scan']
        state = obs['state']

        self.last_scan = scan

        action = self.prevent_collision(scan, state, pp_action)
        # if self.check_collision(scan, state[3], pp_action[0]):
        #     action = self.prevent_collision(scan, state, pp_action)

        # else:
        #     action = pp_action

        self.vis.add_step(scan, action[0])
        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        self.action = action

        return action

    def prevent_collision(self, scan, state, proposed_action):
        """
        This function is called if a collision is detected. It takes the current scan, state and pp action and returns an action that will ensure no crash. 
        """
        new_steer = self.find_best_gap(scan, state)
        new_steer = np.clip(new_steer, -self.max_steer, self.max_steer)
        speed = calculate_speed(new_steer)
        action = np.array([new_steer, speed])

        if not new_steer:
            print(f"Return zero speed")
            action = np.array([0, 0])

        return action

    def find_best_gap(self, scan, state):
        d_stop = max(state[3]**2 / (2*self.max_a) * 2, 1)

        plot_lidar(scan)

        idxs = scan > d_stop
        new_idxs = []
        for i in range(self.n_beams):
            if idxs[i]:
                new_idxs.append(i)
        idxs = np.array(new_idxs)
        if len(idxs) < 2:
            return False # no possibility

        starts = []
        ends = []
        starts.append(idxs[0])
        for i in range(1, len(idxs)):
            if idxs[i] == idxs[i-1] + 1:
                continue
            ends.append(idxs[i-1])
            starts.append(idxs[i])
        ends.append(idxs[-1])

        n_gaps = len(starts)
        widths = []

        for i in range(n_gaps):
            if ends[i] - starts[i] < 10:
                widths.append(0)
                continue
            w = self.calculate_safe_width(scan, starts[i], ends[i], d_stop)
            widths.append(w)

        widths = np.array(widths)
        # print(f"Widths: {widths}")
        best_gap = np.argmax(widths)
        center_idx = (starts[best_gap] + ends[best_gap])/2

        new_steer = (center_idx - self.center_idx) * self.dth 

        return new_steer

    def calculate_safe_width(self, scan, start_i, end_i, d_stop):
        center = int((start_i + end_i) /2)
        n_gap = end_i - start_i
        print(f"Looking between [{start_i}, {end_i}], n_gap: {n_gap}, center: {center}, o_d_stop: {d_stop}")
        abs_max_gap = 200 
        if n_gap > abs_max_gap:
            n_gap = abs_max_gap
            half_i = int(abs_max_gap/2)
            max_lahed = 4
            d_stop = min(scan[center-half_i], scan[center+half_i], max_lahed)
            print(f"Changed [{center-half_i}, {center+half_i}], n_gap: {n_gap}, center: {center}, d_stop: {d_stop}")

        half_i = int(n_gap/2)

        max_w = d_stop * np.sin(self.dth) * n_gap / 2 # over 2 because w is defined as half
        max_w = min(max_w, min(scan[center-half_i:center+half_i]))

        n_searches = int(n_gap / 10)
        for i in range(n_searches):
            w = max_w * (1-i/n_searches)
            if self.check_w_clear(scan, center, w, d_stop):
                print(f"Width {i} = {max_w} -w found as: {w}")
                plot_lidar_col_vals(scan, self.col_vals, wait=False)
                return w 

        return 0 # no gap

    def check_w_clear(self, scan, gap_center_idx, w, d_stop):
        center_angle = (gap_center_idx - self.center_idx) * self.dth
        # d_max = max(scan)

        angles = np.empty(self.n_beams)
        for i in range(self.n_beams):
            angles[i] =  self.fov/(self.n_beams-1) * i

        collision_values = np.empty(len(scan))
        for i in range(len(scan)):
            collision_values[i] = min(w / abs(np.cos(angles[i] + center_angle)), d_stop)
        self.col_vals = collision_values

        for i in range(self.n_beams):
            if scan[i] < collision_values[i]:
                return False 

        return True


    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()

    def show_history(self, wait=False):
        plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

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



class SafetyCar2(SafetyPP):
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
        self.action = None
        self.col_vals = None

    def plan_act(self, obs):
        state = obs['state']
        pp_action = self.act_pp(state)

        action = self.run_safety_check(obs, pp_action)

        return action 

    def check_collision(self, scan, current_speed, proposed_steer):
        d_stop = current_speed**2 / (2*self.max_a) * 2

        # d_th = np.pi/len(scan)
        # bubble = 25 # beams around important one
        # zero_pt = len(scan)/2
        # steer_idx = int(proposed_steer / d_th + zero_pt)

        # idxs = np.arange(steer_idx-bubble, steer_idx+bubble, 1)
        # min_range = np.min(scan[idxs])

        # print(f"MinR: {min_range} --> d_stop: {d_stop}")

        n_beams = len(scan)
        fov = np.pi
        angles = np.empty(n_beams)
        for i in range(n_beams):
            angles[i] =  fov/(n_beams-1) * i
        car_width = 0.15
        collision_values = np.empty(len(scan))
        # sines, cosines = get_trigs(len(scan))
        for i in range(len(scan)):
            collision_values[i] = min(car_width / abs(np.cos(angles[i] + proposed_steer)), d_stop)
            # note that it is capped at d_stop because we would never need more than that.
        self.col_vals = collision_values
        for i in range(n_beams):
            if scan[i] < collision_values[i]:
                print(f"Col detect i: {i} > scan: {scan[i]:.4f} --> col val: {collision_values[i]:.4f}")
                return True

        # plot_lidar(collision_values, wait=True)

        # if min_range < d_stop: # collision
        #     return True
        return False

    def run_safety_check(self, obs, pp_action):
        scan = obs['scan']
        state = obs['state']

        self.last_scan = scan

        if self.check_collision(scan, state[3], pp_action[0]):
            action = self.prevent_collision(scan, state, pp_action)

        else:
            action = pp_action

        self.vis.add_step(scan, action[0])
        self.old_steers.append(pp_action[0])
        self.new_steers.append(action[0])

        self.action = action

        return action

    def prevent_collision(self, scan, state, proposed_action):
        """
        This function is called if a collision is detected. It takes the current scan, state and pp action and returns an action that will ensure no crash. 
        """
        new_steer = self.run_lateral_search(scan, state, proposed_action)
        if not new_steer: # it returns false if no safe val is found
            print(f"No safe option")
            new_steer = self.find_a_gap(scan)
            self.check_collision(scan, state[3], proposed_action[0])
            plot_lidar_col_vals(scan, self.col_vals, action=new_steer, wait=True)

        range_idx = int((new_steer + np.pi/2) / np.pi * 1000)
        range_val = scan[range_idx]
        new_speed = calculate_safe_speed(new_steer, range_val)
        action = np.array([new_steer, new_speed])
        print(f"Old steer: {proposed_action[0]:.4f} --> New Steer: {new_steer:.4f} (V: {state[3]:.4f})")

        return action


    def run_lateral_search(self, scan, state, proposed_action):
        proposed_steer = proposed_action[0]
        current_speed = state[3]
        n_steps = 12
        d_step = self.max_steer/n_steps
        n_steers = []
        p_steers = []
        new_steer = -1
        for i in range(1, n_steps):
            n_steer = max(proposed_steer - i * d_step, -self.max_steer)
            p_steer = min(proposed_steer + i * d_step, self.max_steer)
            p_steers.append(p_steer)
            n_steers.append(n_steer)
            if not self.check_collision(scan, current_speed, n_steer):
                new_steer = n_steer
                break 
            if not self.check_collision(scan, current_speed, p_steer):
                new_steer = p_steer
                break 
        print(f"P steer: {p_steers}")
        print(f"N steer: {n_steers}")
        
        if new_steer == -1:
            return False 
        return new_steer

    def find_a_gap(self, scan):
        # no safe option
        # the best we can do is follow the gap

        d_th = np.pi/len(scan)

        bubble = 125 # beams around center that are driveable
        zero_pt = int(len(scan)/2)

        idxs = np.arange(zero_pt-bubble, zero_pt+bubble, 1)
        d_scan = scan[idxs]
        # fgm_bubble = 20
        # min_range_idx = np.argmin(d_scan)
        # min_range_idx = np.clip(min_range_idx, 0, bubble*2-1)
        # zero_idxs = np.arange(max(min_range_idx-fgm_bubble, 0), min(min_range_idx+fgm_bubble, bubble*2), 1)
        # d_scan[zero_idxs] = np.zeros_like(d_scan[zero_idxs])

        d_scan = np.convolve(d_scan, np.ones(20), 'same') / 20
        max_range_idx = np.argmax(d_scan)

        new_steer = (max_range_idx - bubble) * d_th
        # new_steer = (max_range_idx + zero_pt - bubble) * d_th

        return new_steer

    def plan(self, env_map):
        super().plan(env_map)
        self.old_steers.clear()
        self.new_steers.clear()

    def show_history(self, wait=False):
        plot_lidar_col_vals(self.last_scan, self.col_vals, self.action[0], False)

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


