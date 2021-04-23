import numpy as np 
import casadi as ca 
import csv
from matplotlib import pyplot as plt
from numba import njit
 
import toy_auto_race.Utils.LibFunctions as lib
from toy_auto_race.lidar_viz import LidarViz
from toy_auto_race.speed_utils import calculate_speed



class TrackFGM:    

    # BUBBLE_RADIUS = 160
    BUBBLE_RADIUS = 250
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 100
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_SPEED = 5.0
    CORNERS_SPEED = 5.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 9  # 20 degrees
    
    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.vis = LidarViz(1000)
        self.degrees_per_elem = None

        self.n_beams = 1000
        fov = np.pi 
        # fov = np.pi * 6/10
        angles = [-fov/2 + fov/(self.n_beams-1) * i  for i in range(self.n_beams)]
        self.sines = np.sin(angles)
        self.cosines = np.cos(angles)
    
    def preprocess_lidar(self, ranges):
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
        # self.degrees_per_elem = (np.pi) / len(ranges)
        self.degrees_per_elem = (180) / len(ranges)
	    # we won't use the LiDAR data from directly behind us
        # proc_ranges = np.array(ranges[135:-135])
        reduction = 200
        # reduction = 1
        proc_ranges = np.array(ranges[reduction:-reduction])
        proc_ranges = ranges
        max_range_val = 10
        proc_ranges = np.clip(proc_ranges, 0, max_range_val)
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
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), 'same') 
        averaged_max_gap = averaged_max_gap / self.BEST_POINT_CONV_SIZE
        best = averaged_max_gap.argmax()
        idx = best + start_i

        # mid_idx = int((start_i + end_i) / 2)

        # r = 0.1
        # steer_idx = int(mid_idx * r + idx * (1-r))  

        # max_range = max(ranges)
        # ranges = ranges / max_range

        # plt.figure(2)
        # plt.clf()
        # plt.title("Ranges")

        # # plt.xlim([-1.2, 1.2])
        # # plt.ylim([-0.5, 1.2])

        # for i in range(self.n_beams):
        #     xs = [0, self.sines[i] * ranges[i]]
        #     ys = [0, self.cosines[i] * ranges[i]]
        #     plt.plot(xs, ys, 'b')

        # xs = [0, self.sines[idx] * averaged_max_gap[best] * 1.2]
        # ys = [0, self.cosines[idx] * averaged_max_gap[best] * 1.2]
        # plt.plot(xs, ys, 'r')

        # plt.pause(0.0001)
        
        # plt.figure(3)
        # plt.clf()
        # plt.title("Averaged max gap")
        
        # for i in range(len(averaged_max_gap)):
        #     xs = [0, self.sines[i+start_i] * averaged_max_gap[i]]
        #     ys = [0, self.cosines[i+start_i] * averaged_max_gap[i]]
        #     plt.plot(xs, ys, 'b')
        
        # xs = [0, self.sines[idx] * averaged_max_gap[best] * 1.2]
        # ys = [0, self.cosines[idx] * averaged_max_gap[best] * 1.2]
        # plt.plot(xs, ys, 'r')    

        # xs = [0, self.sines[mid_idx] * ranges[mid_idx] * 1.2]
        # ys = [0, self.cosines[mid_idx] * ranges[mid_idx] * 1.2]
        # plt.plot(xs, ys, 'g')

        # xs = [0, self.sines[steer_idx] * ranges[steer_idx] * 1.2]
        # ys = [0, self.cosines[steer_idx] * ranges[steer_idx] * 1.2]
        # plt.plot(xs, ys, 'p')

        # plt.pause(0.0001)

        return idx
        # return  steer_idx

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

        # speed = 4
        speed = calculate_speed(steering_angle)


        action = np.array([steering_angle, speed])
        self.vis.add_step(proc_ranges, steering_angle)

        return action

    def reset_lap(self):
        pass


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
        """ Preprocess the LiDAR scan array. Expert implementation includes:
            1.Setting each value to the mean over some window
            2.Rejecting high values (eg. > 3m)
        """
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
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
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
        scan = obs[7:-1]
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


