import numpy as np 
import casadi as ca 
import csv
from matplotlib import pyplot as plt
from numba import njit
 
import toy_auto_race.Utils.LibFunctions as lib
from toy_auto_race.lidar_viz import LidarViz



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

        self.vis = LidarViz(sim_conf.n_beams)

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

        action = np.array([steering_angle, speed])
        self.vis.add_step(scan, steering_angle)

        return action

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




class GapFollower:    

    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_SPEED = 5.0
    CORNERS_SPEED = 5.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 9  # 20 degrees
    
    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.vis = LidarViz(996)
        self.degrees_per_elem = None
    
    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # self.degrees_per_elem = (2*np.pi) / len(ranges)
        self.degrees_per_elem = (180) / len(ranges)
	# we won't use the LiDAR data from directly behind us
        # proc_ranges = np.array(ranges[135:-135])
        reduction = 2
        proc_ranges = np.array(ranges[reduction:-reduction])
        # proc_ranges = ranges
        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
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
    
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
	Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), 'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data
        """
        return (range_index - (range_len/2)) * self.degrees_per_elem

    def process_lidar(self, ranges):
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges)
        #Find closest point to LiDAR
        closest = proc_ranges.argmin()

        #Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges)-1
        proc_ranges[min_index:max_index] = 0

        #Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        #Find the best point in the gap 
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        #Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else: speed = self.STRAIGHTS_SPEED
        # print('Steering angle in degrees: {}'.format((steering_angle/np.pi)*180))
        #return 5.0, np.pi/9
        return speed, steering_angle, proc_ranges

    def plan_act(self, obs):
        scan = obs[7:-1]
        ranges = np.array(scan, dtype=np.float)

        speed, steering_angle, proc_ranges = self.process_lidar(ranges)
        steering_angle = steering_angle * np.pi / 180

        action = np.array([steering_angle, speed])
        self.vis.add_step(proc_ranges, steering_angle)

        return action

    def reset_lap(self):
        pass
