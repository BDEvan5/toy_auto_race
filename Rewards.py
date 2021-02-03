import numpy as np 

import LibFunctions as lib


class StdNavReward:
    def __init__(self, config) -> None:
        self.end = [config['map']['end']['x'], config['map']['end']['y']]
        self.beta = config['reward']['beta']

    def __call__(self, s, a, s_p, r) -> float:
        if r == -1:
            return r 
        else:
            prev_dist = lib.get_distance(s[0:2], self.end)
            cur_dist = lib.get_distance(s_p[0:2], self.end)

            new_r = (prev_dist - cur_dist) * self.beta
            return new_r




