import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import zeros_like


class LidarViz:
    def __init__(self, n_beams=10):
        self.state_mem = []
        self.action_mem = []

        self.n_beams = n_beams
        fov = np.pi 
        # fov = np.pi * 6/10
        angles = [-fov/2 + fov/(n_beams-1) * i  for i in range(n_beams)]
        self.sines = np.sin(angles)
        self.cosines = np.cos(angles)

    def add_step(self, state, action):
        self.state_mem.append(state)
        self.action_mem.append(action)

    def play_visulisation(self):
        N = len(self.state_mem)
        for i in range(N):
            state = self.state_mem[i]
            ranges = state # for fgm
            # ranges = state[4:] # for mod state
            action = self.action_mem[i]
            self.visualize(ranges, action, i)

        plt.pause(2)

        self.state_mem.clear()
        self.action_mem.clear()

    def visualize(self, ranges, action, number=0):
        max_range = max(ranges)
        ranges = ranges / max_range

        plt.figure(2)
        plt.clf()

        plt.xlim([-1.2, 1.2])
        plt.ylim([-0.5, 1.2])

        for i in range(self.n_beams):
            xs = [0, self.sines[i] * ranges[i]]
            ys = [0, self.cosines[i] * ranges[i]]
            plt.plot(xs, ys, 'b')

        xs = [0, action]
        ys = [-0.2, -0.2]
        plt.plot(xs, ys, 'r', linewidth=5)

        angle = action * 0.4 
        xs = [0, np.sin(angle) * 1.2]
        ys = [0, np.cos(angle) * 1.2]
        plt.plot(xs, ys, 'g', linewidth=2)

        plt.text(0, -0.3, f"{number}")

        plt.pause(0.0001)
        




class LidarVizMod:
    def __init__(self, n_beams=10):
        self.state_mem = []
        self.pp_mem = []
        self.nn_mem = []

        self.n_beams = n_beams
        fov = np.pi 
        angles = [-fov/2 + fov/(n_beams-1) * i  for i in range(n_beams)]
        self.sines = np.sin(angles)
        self.cosines = np.cos(angles)

    def add_step(self, state, pp, nn):
        self.state_mem.append(state)
        self.pp_mem.append(pp)
        self.nn_mem.append(nn)

    def play_visulisation(self):
        N = len(self.state_mem)
        for i in range(N):
            state = self.state_mem[i]
            ranges = state # for fgm
            # ranges = state[4:] # for mod state
            pp = self.pp_mem[i]
            nn = self.nn_mem[i]
            self.visualize(ranges, pp, nn, i)

        plt.pause(2)

        self.state_mem.clear()
        self.pp_mem.clear()
        self.nn_mem.clear()

    def visualize(self, ranges, pp, nn, number=0):
        max_range = max(ranges)
        ranges = ranges / max_range

        plt.figure(2)
        plt.clf()

        plt.xlim([-1.2, 1.2])
        plt.ylim([-0.5, 1.2])

        for i in range(self.n_beams):
            xs = [0, self.sines[i] * ranges[i]]
            ys = [0, self.cosines[i] * ranges[i]]
            plt.plot(xs, ys, 'b')

        xs = [0, pp]
        ys = [-0.2, -0.2]
        plt.plot(xs, ys, 'r', linewidth=5)

        xs = [0, nn]
        ys = [-0.3, -0.3]
        plt.plot(xs, ys, 'r', linewidth=5)

        angle = pp * 0.4 
        xs = [0, np.sin(angle) * 1.1]
        ys = [0, np.cos(angle) * 1.1]
        plt.plot(xs, ys, 'g', linewidth=2)

        angle = nn * 0.4 
        xs = [0, np.sin(angle) * 1.1]
        ys = [0, np.cos(angle) * 1.1]
        plt.plot(xs, ys, 'g', linewidth=2)        

        angle = (nn+pp) * 0.4 
        xs = [0, np.sin(angle) * 1.5]
        ys = [0, np.cos(angle) * 1.5]
        plt.plot(xs, ys, 'r', linewidth=2)

        plt.text(-1, -0.3, f"{number}")

        plt.pause(0.01)
        



