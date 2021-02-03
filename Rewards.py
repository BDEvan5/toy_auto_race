import numpy as np 

import LibFunctions as lib


def find_closest_pt(pt, wpts):
    """
    Returns the two closes points in order along wpts
    """
    dists = [lib.get_distance(pt, wpt) for wpt in wpts]
    min_i = np.argmin(dists)
    d_i = dists[min_i] 
    if dists[min_i -1 ] > dists[min_i + 1]:
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
            d_dis = (prev_dist - cur_dist) / self.dis_scale

            new_r = self.b1 + d_dis * self.b2 + self.b3 * v_sc + r
            return new_r


class CrossTrackHeadingReward:
    def __init__(self, config, t1, t2, t3) -> None:
        self.pts = None 
        self.vs = None

        self.t1 = t1
        self.t2 = t2
        self.t3 = t3

        self.max_v = config['lims']['max_v']
        self.dis_scale = config['lims']["dis_scale"]

    def init_reward(self, pts, vs):
        self.pts = pts
        self.vs = vs
        
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            pt_i, pt_ii, d_i, d_ii = find_closest_pt(s_p[0:2], self.pts)
            d = lib.get_distance(pt_i, pt_ii)
            h = get_tiangle_h(pt_i, pt_ii, d) 
            d_c = h / self.dis_scale

            th_ref = lib.get_bearing(pt_i, pt_ii)
            th = s_p[2]
            d_th = abs(lib.sub_angles_complex(th_ref, th))

            v = s_p[3] / self.max_v

            new_r = self.t1 * v*(np.cos(d_th) * self.t2 + self.t3 * d_c)

            return new_r

class OnlineSteering:
    def __init__(self, config, s1, s2) -> None:
        self.s1 = s1
        self.s2 = s2

    def init_reward(self, pts, vs):
        pass
        
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            new_r = self.s1 - self.s2 * s_p[4] ** 2

            return new_r

class ModStdTimeReward:
    def __init__(self, config, m1, m2, mt) -> None:
        self.max_steer = config['lims']['max_steer']
        self.m1 = m1
        self.m2 = m2

        self.mt = mt

    def init_reward(self, pts, vs):
        pass
        
    def __call__(self, s, a, s_p, r, time=0) -> float:
        if r == -1:
            return r
        else:
            # time = 0
            steer = abs(a[1]) / self.max_steer
            new_r = self.m1 - self.m2 * steer + self.mt * time
            return new_r

class ModHeadingReward:
    def __init__(self, config, m1, m3, m4) -> None:
        self.m1 = m1
        self.m3 = m3
        self.m4 = m4

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
            d_c = get_tiangle_h(pt_i, pt_ii, d)

            th_ref = lib.get_bearing(pt_i, pt_ii)
            th = s_p[2]
            d_th = abs(lib.sub_angles_complex(th_ref, th))

            new_r = self.m1 - self.m3 * d_c - self.m4 * d_th
            return new_r


# class ModTimeReward:
#     def __init__(self, config, pts, vs) -> None:
#         self.pts = pts 
#         self.vs = vs
        
#     def __call__(self, s, a, s_p, r) -> float:
#         if r == -1:
#             return r
#         else:
#             new_r = 0
#             return new_r

