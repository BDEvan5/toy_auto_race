import numpy as np
from numba import njit

from toy_auto_race.TrajectoryPlanner import MinCurvatureTrajectory
import toy_auto_race.Utils.LibFunctions as lib


class OraclePP:
    def __init__(self, sim_conf) -> None:
        self.name = "Oracle Path Follower"
        self.path_name = None

        mu = sim_conf.mu
        g = sim_conf.g
        self.m = sim_conf.m
        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.f_max = mu * self.m * g #* safety_f

        pp_conf = lib.load_conf("mod_conf")
        self.v_gain = 0.9
        self.lookahead = 1

        self.wpts = None
        self.vs = None

        self.aim_pts = []

    def _get_current_waypoint(self, position, theta):
        # nearest_pt, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, self.wpts)
        nearest_pt, nearest_dist, t, i = self.nearest_pt(position)

        if nearest_dist < self.lookahead:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, self.lookahead, self.wpts, i+t, wrap=True)
            if i2 == None:
                print(f"No wpts to return: _get_current_waypoint returns None")
                return None
            i = i2
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = self.wpts[i2]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < 20:
            return np.append(self.wpts[i], self.vs[i])

    def act_pp(self, obs):
        pose_th = obs[2]
        pos = np.array(obs[0:2], dtype=np.float)

        lookahead_point = self._get_current_waypoint(pos, pose_th)
        self.aim_pts.append(lookahead_point)

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = self.get_actuation(pose_th, lookahead_point, pos)
        speed = self.v_gain * speed

        return [steering_angle, speed]

    def reset_lap(self):
        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

        self.aim_pts.clear()

    def get_actuation(self, pose_theta, lookahead_point, position):
        waypoint_y = np.dot(np.array([np.cos(pose_theta), np.sin(-pose_theta)]), lookahead_point[0:2]-position)
        
        speed = lookahead_point[2]
        if np.abs(waypoint_y) < 1e-6:
            return speed, 0.
        radius = 1/(2.0*waypoint_y/self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase/radius)

        return speed, steering_angle

    def nearest_pt(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

    # print min_dist_segment, dists[min_dist_segment], projections[min_dist_segment]




class Oracle(OraclePP):
    def __init__(self, sim_conf):
        OraclePP.__init__(self, sim_conf)

        self.wpts = None
        self.vs = None 

    def plan(self, env_map):
        # load center pts
        start_y = env_map.start_pose[1]
        start_x = env_map.start_pose[0]
        max_width = env_map.start_pose[0]
        length = env_map.end_y - start_y

        y_pts = np.linspace(start_y, length+start_y, 50)[:, None]
        t_pts = np.concatenate([np.ones_like(y_pts)*start_x, y_pts], axis=-1)
        
        # set true widths
        ws = find_true_widths(t_pts, max_width, env_map.check_plan_location)

        # Optimise n_set
        N = len(t_pts)
        nvecs = np.concatenate([np.ones((N, 1)), np.zeros((N, 1))], axis=-1)
        n_set = MinCurvatureTrajectory(t_pts, nvecs, ws)

        self.wpts = np.concatenate([np.ones((N, 1))*start_x + n_set, y_pts], axis=-1)
        self.vs = np.ones(N) * 3

        self.reset_lap()

        return self.wpts

    def plan_act(self, obs):
        action = self.act_pp(obs)
        
        return action

def find_true_widths(t_pts, max_width, check_scan_location):
    tx = t_pts[:, 0]
    ty = t_pts[:, 1]

    stp_sze = 0.1
    sf = 0.5 # safety factor
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
                    nws.append(-j*(1+sf))
                    pws.append(max_width)
                    break
                elif not check_scan_location(n_pt):
                    pws.append(-j*(1+sf))
                    nws.append(max_width)
                    break 

    nws, pws = np.array(nws)[:, None], np.array(pws)[:, None]
    ws = np.concatenate([nws, pws], axis=-1)

    return ws
