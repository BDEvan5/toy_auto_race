import numpy as np 
import casadi as ca 
import csv
from matplotlib import pyplot as plt
from numba import njit
 
import toy_auto_race.Utils.LibFunctions as lib


#TODO: add imports and update to use scan from state.

class FollowTheGap:
    def __init__(self, config):
        self.name = "Follow The Gap"
        self.config = config
        self.env_map = None
        self.map = None
        self.cur_scan = None
        self.cur_odom = None
    
        self.max_speed = config['lims']['max_v']
        self.max_steer = config['lims']['max_steer']
        self.wheelbase = config['car']['l_r'] + config['car']['l_f']
        mu = config['car']['mu']
        self.m = config['car']['m']
        g = config['car']['g']
        safety_f = config['pp']['force_f']
        self.f_max = mu * self.m * g * safety_f

        # n_beams = config['sim']['beams']
        n_beams = 20
        self.scan_sim = ScanSimulator(n_beams, np.pi)
        self.n_beams = n_beams
        
    def init_agent(self, env_map):
        self.scan_sim.set_check_fcn(env_map.check_scan_location)
        self.env_map = env_map


    def act(self, obs):
        scan = self.scan_sim.get_scan(obs[0], obs[1], obs[2])
        ranges = np.array(scan, dtype=np.float)
        o_ranges = ranges
        angle_increment = np.pi / len(ranges)

        max_range = 1
        # ranges = preprocess_lidar(ranges, max_range)

        bubble_r = 0.1
        ranges = create_zero_bubble(ranges, bubble_r)
        
        start_i, end_i = find_max_gap(ranges)

        aim = find_best_point(start_i, end_i, ranges[start_i:end_i])

        half_pt = len(ranges) /2
        steering_angle =  angle_increment * (aim - half_pt)

        val = ranges[aim] * 4
        th = lib.add_angles_complex(obs[2], steering_angle)
        pt = lib.theta_to_xy(th) * val
        self.env_map.targets.append(pt)

        speed = self.max_speed * ranges[aim] / max_range * 0.5
        # steering_angle = self.limit_inputs(speed, steering_angle)

        return np.array([speed, steering_angle])

    def limit_inputs(self, speed, steering_angle):
        max_steer = np.arctan(self.f_max * self.wheelbase / (speed**2 * self.m))
        new_steer = np.clip(steering_angle, -max_steer, max_steer)

        if max_steer < abs(steering_angle):
            print(f"Problem, Steering clipped from: {steering_angle} --> {max_steer}")

        return new_steer


@njit
def preprocess_lidar(ranges, max_range):
    ranges = np.array([min(ran, max_range) for ran in ranges])
    
    # moving_avg
    # n = 3
    # cumsum = np.cumsum(np.insert(ranges, 0, 0))
    # proc_ranges = (cumsum[n:] - cumsum[:-n])/float(n)

    proc_ranges = ranges

    return proc_ranges

# @njit
def create_zero_bubble(input_vector, bubble_r):
    centre = np.argmin(input_vector)
    min_dist = input_vector[centre]
    input_vector[centre] = 0
    size = len(input_vector)

    current_idx = centre
    while(current_idx < size -1 and input_vector[current_idx] < (min_dist + bubble_r)):
        input_vector[current_idx] = 0
        current_idx += 1
    
    current_idx = centre
    while(current_idx > 0  and input_vector[current_idx] < (min_dist + bubble_r)):
        input_vector[current_idx] = 0
        current_idx -= 1

    return input_vector
    
# @njit
def find_max_gap(input_vector):
    max_start = 0
    max_size = 0

    current_idx = 0
    size = len(input_vector)

    # exclude gaps that are smaller than this. Currently 1m
    min_distance = 0.5

    while current_idx < size:
        current_start = current_idx
        current_size = 0
        while current_idx< size and input_vector[current_idx] > min_distance:
            current_size += 1
            current_idx += 1
        if current_size > max_size:
            max_start = current_start
            max_size = current_size
            current_size = 0
        current_idx += 1
    if current_size > max_size:
        max_start = current_start
        max_size = current_size

    if max_size == 1:
        # max_start -= 1
        max_size = 3

    return max_start, max_start + max_size - 1


# @njit  
def find_best_point(start_i, end_i, ranges):
    # return best index to goto
    mid_i = (start_i + end_i) /2
    best_i = np.argmax(ranges)  
    best_i = (mid_i + (best_i + start_i)) /2

    return int(best_i)

