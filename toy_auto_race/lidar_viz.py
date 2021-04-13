import numpy as np
from matplotlib import pyplot as plt
from numpy.core.numeric import zeros_like


class LidarViz:
    def __init__(self, n_beams=10):
        self.state_mem = []
        self.action_mem = []

        self.n_beams = n_beams
        fov = np.pi
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
            ranges = state[4:]
            action = self.action_mem[i]
            self.visualize(ranges, action, i)

        plt.pause(5)

        self.state_mem.clear()
        self.action_mem.clear()

    def visualize(self, ranges, action, number=0):
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

        plt.pause(0.2)
        



