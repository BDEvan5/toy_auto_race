import numpy as np 
import casadi as ca 
import csv
from matplotlib import pyplot as plt
from numba import njit
 
import toy_auto_race.Utils.LibFunctions as lib


#TODO: add imports and update to use scan from state.

class FollowTheGap:
    def __init__(self, sim_conf):
        self.name = "Follow The Gap"
        self.cur_scan = None
        self.cur_odom = None
    
        self.max_speed = sim_conf.max_v
        self.max_steer = sim_conf.max_steer
        self.wheelbase = sim_conf.l_r + sim_conf.l_f
        mu = sim_conf.mu
        self.m = sim_conf.m
        g = sim_conf.g
        self.f_max = mu * self.m * g

        self.n_beams = sim_conf.n_beams
        self.plan_f = 10
        self.loop_counter = 0
        self.action = None

    def reset_lap(self):
        pass # called for likeness with other vehicles

    def plan_act(self, obs):
        scan = obs[5:-1]
        ranges = np.array(scan, dtype=np.float)
        o_ranges = np.copy(ranges)
        angle_increment = np.pi / len(ranges)

        max_range = 1
        # ranges = preprocess_lidar(ranges, max_range)

        bubble_r = 0.1
        ranges = create_zero_bubble(ranges, bubble_r)
        
        start_i, end_i = find_max_gap(ranges)

        aim = find_best_point(start_i, end_i, ranges[start_i:end_i])

        half_pt = (len(ranges) -1) /2
        steering_angle =  angle_increment * (aim - half_pt) * 0.5  # steering smoothing


        # speed = self.max_speed * ranges[aim] / max_range * 0.9 # 
        speed = 4
        # v_safety factor
        # steering_angle = self.limit_inputs(speed, steering_angle)

        return np.array([steering_angle, speed])

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
    min_distance = 0.25

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

