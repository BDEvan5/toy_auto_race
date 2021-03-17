import numpy as np 
import csv
from matplotlib import pyplot as plt
from PIL import Image
import yaml

import toy_auto_race.Utils.LibFunctions as lib

class TestMap:
    #TODO: convert mapping class to a map class which holds and loads map data and a separate premap class that is able to convert the data as needed. 
    def __init__(self, map_name) -> None:
        self.map_name = map_name 
        self.wpts = None
        self.ss = None
        self.vs = None
        self.N = None

        self.diffs = None
        self.l2s = None 

        self.map_img_name = None
        self.map_img = None
        self.resolution = None
        self.origin = None

        self.read_yaml_file()
        self._load_csv_track()
        self.load_map_img()

    def _load_csv_track(self):
        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.N = len(track)
        self.ss = track[:, 0]
        self.wpts = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2 

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.wpts
        o_ss = self.ss
        o_vs = self.vs
        new_line = []
        new_ss = []
        new_vs = []
        for i in range(self.N-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                ds = o_ss[i+1] - o_ss[i]
                new_ss.append(o_ss[i] + dz*j*ds)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.wpts = np.array(new_line)
        self.ss = np.array(new_ss)
        self.vs = np.array(new_vs)
        self.N = len(new_line)

    def load_map_img(self):
        map_img_name = 'maps/' + self.map_name + ".pgm"

        try:
            self.map_img = np.array(Image.open(map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        except Exception as e:
            print(f"MapPath: {map_img_name}")
            print(f"Exception in reading: {e}")
            raise ImportError(f"Cannot read map")

    def read_yaml_file(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)

        yaml_file = dict(documents.items())

        self.resolution = yaml_file['resolution']
        self.origin = yaml_file['origin']
        self.map_img_name = yaml_file['image']
    
    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        return c, r

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)


    def show_map(self, wait=False):
        plt.figure(2)
        plt.clf()
        plt.imshow(self.map_img, origin='lower')

        xs, ys = [], []
        for pt in self.wpts:
            s_x, s_y = self.xy_to_row_column(pt)
            xs.append(s_x)
            ys.append(s_y)
        plt.plot(xs, ys, linewidth=2)


        if self.wpts is not None:
            wpt_x, wpt_y = self.convert_positions(self.wpts)
            plt.plot(wpt_x, wpt_y, linewidth=2)

        # plt.show()
        plt.pause(0.0001)
        if wait:
            plt.show()

if __name__ == "__main__":
    myMap = TestMap("porto")
    myMap.show_map(True)


