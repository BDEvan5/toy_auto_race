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
    def __init__(self, config) -> None:
        self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.beta = config['reward']['b1']

    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r 
        else:
            prev_dist = lib.get_distance(s[0:2], self.end)
            cur_dist = lib.get_distance(s_p[0:2], self.end)

            new_r = (prev_dist - cur_dist) * self.beta
            return new_r


class CrossTrackHeadingReward:
    def __init__(self, config, pts, vs) -> None:
        self.pts = pts 
        self.vs = vs

        self.t1 = config['reward']['t1']
        self.t2 = config['reward']['t2']
        self.t3 = config['reward']['t3']
        
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

            v = s_p[3]

            new_r = v*(np.cos(d_th) + abs(np.sin(d_th) + d_c))

            return new_r

class OnlineSteering:
    def __init__(self, config) -> None:
        self.s1 = config['reward']['s1']
        self.s2 = config['reward']['s2']
        
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            new_r = self.s1 - self.s2 * s_p[4] ** 2

            return new_r

class ModStdTimeReward:
    def __init__(self, config) -> None:
        self.m1 = config['reward']['m1']
        self.m2 = config['reward']['m2']

        self.mt = config['reward']['mt']
        
    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r
        else:
            time = 0
            new_r = self.m1 - self.m2 * a[0] + self.mt * time
            return new_r

class ModHeadingReward:
    def __init__(self, config, pts, vs) -> None:
        self.m1 = config['reward']['m1']
        self.m2 = config['reward']['m2']
        self.m3 = config['reward']['m3']
        self.m4 = config['reward']['m4']
        self.mt = config['reward']['mt']
        
        
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

