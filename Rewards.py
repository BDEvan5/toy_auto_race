import numpy as np 

import LibFunctions as lib


def find_closest_pt(pt, wpts):
    """
    Returns the two closes points in order along wpts
    """
    dists = [lib.get_distance(pt, wpt) for wpt in wpts]
    min_i = np.argmin(dists)
    d_i = dists[min_i] 
    if min_i == len(dists) - 1:
        min_i -= 1
    if dists[max(min_i -1, 0) ] > dists[min_i+1]:
        p_i = wpts[min_i]
        p_ii = wpts[min_i+1]
        d_i = dists[min_i] 
        d_ii = dists[min_i+1] 
    else:
        p_i = wpts[min_i-1]
        p_ii = wpts[min_i]
        d_i = dists[min_i-1] 
        d_ii = dists[min_i] 

    return p_i, p_ii, d_i, d_ii

def get_tiangle_h(a, b, c):
    s = (a + b+ c) / 2
    A = np.sqrt(s*(s-a)*(s-b)*(s-c))
    h = 2 * A / c

    return h

def distance_potential(s, s_p, end, beta=0.2, scale=0.5):
    prev_dist = lib.get_distance(s[0:2], end)
    cur_dist = lib.get_distance(s_p[0:2], end)
    d_dis = (prev_dist - cur_dist) / scale

    return d_dis * beta



# Gen
class StdNavReward:
    def __init__(self, config, b1, b2, b3) -> None:
        self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.max_v = config['lims']['max_v']
        self.dis_scale = config['lims']["dis_scale"]

        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def init_reward(self, pts, vs):
        pass

    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r 
        else:
            prev_dist = lib.get_distance(s[0:2], self.end)
            cur_dist = lib.get_distance(s_p[0:2], self.end)

            v_sc = s_p[2] / self.max_v
            shaped_r = distance_potential(s, s_p, self.end, self.b2)

            new_r = self.b1 + self.b3 * v_sc 
            return new_r  + r + shaped_r


class CrossTrackHeadingReward:
    def __init__(self, config, t1, t2, t3) -> None:
        self.pts = None 
        self.vs = None

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        self.max_v = config['lims']['max_v']
        self.dis_scale = config['lims']["dis_scale"]
        self.end = [config['map']['end']['x'], config['map']['end']['y']]

    def init_reward(self, pts, vs):
        self.pts = pts
        self.vs = vs
        
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            pt_i, pt_ii, d_i, d_ii = find_closest_pt(s_p[0:2], self.pts)
            d = lib.get_distance(pt_i, pt_ii)
            h = get_tiangle_h(d_i, d_ii, d) 
            d_c = h / self.dis_scale

            th_ref = lib.get_bearing(pt_i, pt_ii)
            th = s_p[2]
            d_th = abs(lib.sub_angles_complex(th_ref, th))

            v = s_p[3] / self.max_v
            shaped_r = distance_potential(s, s_p, self.end)

            new_r = self.t1 * v*(np.cos(d_th) * self.t2 - self.t3 * d_c + 0.4) 

            return new_r  + r + shaped_r

class OnlineSteering:
    def __init__(self, config, s1, s2, s3) -> None:
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.max_steer = config['lims']['max_steer']
        self.max_velocity = config['lims']['max_v']

    def init_reward(self, pts, vs):
        pass
        
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            steer = s_p[4] / self.max_steer
            vel = s_p[3] / self.max_velocity
            new_r = self.s1 - self.s2 * steer ** 2 + self.s3 * vel

            return new_r + r

# Mod
class ModSteerReward:
    def __init__(self, config, mv, ms) -> None:
        self.max_steer = config['lims']['max_steer']
        self.max_v = config['lims']['max_v']
        self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.mv = mv 
        self.ms = ms 

    def init_reward(self, pts, vs):
        pass
        
    def __call__(self, s, a, s_p, r, time=0) -> float:
        if r == -1:
            return r
        else:
            shaped_r = distance_potential(s, s_p, self.end)

            vel = a[0] / self.max_v 
            steer = abs(a[1]) / self.max_steer

            new_r = self.mv * vel - self.ms * steer 

            return new_r + r + shaped_r 

class ModHeadingReward:
    def __init__(self, config, mh, md, m) -> None:
        self.mh = mh 
        self.md = md
        self.m = m
        self.dis_scale = config['lims']["dis_scale"]
        self.max_v = config['lims']["max_v"]
        self.end = [config['map']['end']['x'], config['map']['end']['y']]

        self.pts = None
        self.vs = None

    def init_reward(self, pts, vs):
        self.pts = pts
        self.vs = vs
            
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            pt_i, pt_ii, d_i, d_ii = find_closest_pt(s_p[0:2], self.pts)
            d = lib.get_distance(pt_i, pt_ii)
            d_c = get_tiangle_h(d_i, d_ii, d) / self.dis_scale

            th_ref = lib.get_bearing(pt_i, pt_ii)
            th = s_p[2]
            d_th = abs(lib.sub_angles_complex(th_ref, th))
            v_scale = s_p[3] / self.max_v

            shaped_r = distance_potential(s, s_p, self.end)

            new_r =  self.mh * np.cos(d_th) * v_scale - self.md * d_c
            new_r *= self.m

            return new_r + r + shaped_r

class ModTimeReward:
    def __init__(self, config, mt) -> None:
        self.mt = mt 
        self.dis_scale = config['lims']["dis_scale"]
        self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.max_steer = config['lims']['max_steer']

    def init_reward(self, pts, vs):
        pass

    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            shaped_r = distance_potential(s, s_p, self.end)

            new_r = - self.mt

            return new_r + r + shaped_r



