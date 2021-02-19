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



# Forest Rewards
class SteerReward:
    def __init__(self, config, mv, ms) -> None:
        self.max_steer = config['lims']['max_steer']
        self.max_v = config['lims']['max_v']
        self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.mv = mv 
        self.ms = ms 

    def init_reward(self, pts, vs):
        pass
        
    def __call__(self, s, a, s_p, r, dev) -> float:
        if r == -1:
            return r
        else:
            shaped_r = distance_potential(s, s_p, self.end)

            vel = a[0] / self.max_v 
            steer = abs(a[1]) / self.max_steer

            new_r = self.mv * vel - self.ms * steer 

            return new_r + r + shaped_r 

class CthReward:
    def __init__(self, config, mh, md) -> None:
        self.mh = mh 
        self.md = md
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

            return new_r + r + shaped_r

class TimeReward:
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


# Track rewards
class TrackRewardBase:
    def __init__(self) -> None:
        self.wpts = None
        self.ss = None
        self.vs = None

    def init_reward(self, pts, vs):
        self.wpts = pts
        self.vs = vs

        N = len(pts)
        ss = np.array([lib.get_distance(pts[i], pts[i+1]) for i in range(N-1)])
        ss = np.cumsum(ss)
        self.ss = np.insert(ss, 0, 0)

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 


    def find_s(self, point):
        dots = np.empty((self.wpts.shape[0]-1, ))
        for i in range(dots.shape[0]):
            dots[i] = np.dot((point - self.wpts[i, :]), self.diffs[i, :])
        t = dots / self.l2s

        t = np.clip(dots / self.l2s, 0.0, 1.0)
        projections = self.wpts[:-1,:] + (t*self.diffs.T).T
        dists = np.linalg.norm(point - projections, axis=1)

        min_dist_segment = np.argmin(dists)
        dist_from_cur_pt = dists[min_dist_segment]
        if dist_from_cur_pt > 1: #more than 2m from centerline
            return self.ss[min_dist_segment] - dist_from_cur_pt # big makes it go back

        s = self.ss[min_dist_segment] + dist_from_cur_pt

        return s 

    def get_shpaed_r(self, pt1, pt2):
        s = self.find_s(pt1)
        ss = self.find_s(pt2)
        ds = ss - s
        r = ds * 0.2
        shaped_r = np.clip(r, -0.5, 0.5)

        return shaped_r

#golden oldies
class TrackDevReward(TrackRewardBase):
    def __init__(self, config) -> None:
        TrackRewardBase.__init__(self)
        self.dis_scale = config['lims']["dis_scale"]
        self.max_steer = config['lims']['max_steer']

    def __call__(self, s, a, s_p, r, dev):
        if r == -1:
            return -1
        else:
            beta = 0.2
            shaped_r = self.get_shpaed_r(s[0:2], s_p[0:2])
            ret_r = shaped_r - beta * abs(dev)
            # ret_r = 0.2 - 0.2 * abs(dev)

            return ret_r

class TrackOldReward:
    def __init__(self, config) -> None:
        pass

    def init_reward(self, wpts, vs):
        pass

    def __call__(self, s, a, s_p, r, dev):
        if r == -1:
            return -1
        else:
            beta = 0.2
            ret_r = 0.2 - beta * abs(dev)

            return ret_r

class TrackStdReward(TrackRewardBase):
    def __init__(self, config) -> None:
        TrackRewardBase.__init__(self)
        self.dis_scale = config['lims']["dis_scale"]
        self.max_steer = config['lims']['max_steer']

    def __call__(self, s, a, s_p, r, dev):
        if r == -1:
            return -1
        else:
            shaped_r = self.get_shpaed_r(s[0:2], s_p[0:2])

            return shaped_r

# newbies for testing
class TrackTimeReward(TrackRewardBase):
    def __init__(self, config, mt) -> None:
        TrackRewardBase.__init__(self)
        self.mt = mt 
        self.dis_scale = config['lims']["dis_scale"]
        self.max_steer = config['lims']['max_steer']
        
    def __call__(self, s, a, s_p, r, dev) -> float:
        if r == -1:
            return -1
        else:
            shaped_r = self.get_shpaed_r(s[0:2], s_p[0:2])
            ret_r = shaped_r - self.mt

            return ret_r

class TrackSteerReward(TrackRewardBase):
    def __init__(self, config, mv, ms) -> None:
        TrackRewardBase.__init__(self)
        self.max_steer = config['lims']['max_steer']
        self.max_v = config['lims']['max_v']
        self.mv = mv 
        self.ms = ms 

            
    def __call__(self, s, a, s_p, r, dev) -> float:
        if r == -1:
            return r
        else:
            shaped_r = self.get_shpaed_r(s[0:2], s_p[0:2])

            vel = a[0] / self.max_v 
            steer = abs(a[1]) / self.max_steer

            new_r = self.mv * vel - self.ms * steer 

            return new_r + shaped_r 

class TrackCthReward(TrackRewardBase):
    def __init__(self, config, mh, md) -> None:
        TrackRewardBase.__init__(self)
        self.mh = mh 
        self.md = md
        self.dis_scale = config['lims']["dis_scale"]
        self.max_v = config['lims']["max_v"]

    def __call__(self, s, a, s_p, r, dev):
        if r == -1:
            return r
        else:
            shaped_r = self.get_shpaed_r(s[0:2], s_p[0:2])

            pt_i, pt_ii, d_i, d_ii = find_closest_pt(s_p[0:2], self.wpts)
            d = lib.get_distance(pt_i, pt_ii)
            d_c = get_tiangle_h(d_i, d_ii, d) / self.dis_scale

            th_ref = lib.get_bearing(pt_i, pt_ii)
            th = s_p[2]
            d_th = abs(lib.sub_angles_complex(th_ref, th))
            v_scale = s_p[3] / self.max_v

            new_r =  self.mh * np.cos(d_th) * v_scale - self.md * d_c

            return new_r + shaped_r 





