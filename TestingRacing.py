from Rewards import CrossTrackHeadingReward, ModHeadingReward, ModStdTimeReward, OnlineSteering, StdNavReward
import numpy as np
import csv, yaml

from HistoryStructs import TrainHistory
from Simulator import ForestSim
from SimMaps import  ForestMap
from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import LibFunctions as lib

# from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenTest, GenVehicle

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'


"""Train"""
def TrainVehicle(config, agent_name, vehicle, reward, steps=20000):
    path = 'Vehicles/' + agent_name
    buffer = ReplayBufferTD3()

    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(config)
    env = ForestSim(env_map)

    t_his = TrainHistory(agent_name)
    print_n = 500

    done, state = False, env.reset()
    wpts = vehicle.init_agent(env_map)

    for n in range(steps):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        new_r = reward(state, a, s_prime, r)
        vehicle.add_memory_entry(new_r, done, s_prime, buffer)
        t_his.add_step_data(new_r)

        state = s_prime
        vehicle.agent.train(buffer, 2)
        
        # env.render(False)

        if n % print_n == 0 and n > 0:
            t_his.print_update()
            vehicle.agent.save(directory=path)
        
        if done:
            t_his.lap_done()
            # vehicle.show_vehicle_history()
            env.render(wait=False, save=False)

            vehicle.reset_lap()
            state = env.reset()


    vehicle.agent.save(directory=path)
    t_his.save_csv_data()

    print(f"Finished Training: {agent_name}")

    return t_his.rewards



"""Testing Function"""
class TestData:
    def __init__(self) -> None:
        self.endings = None
        self.crashes = None
        self.completes = None
        self.lap_times = None

        self.names = []

        self.N = None

    def init_arrays(self, N, laps):
        self.completes = np.zeros((N))
        self.crashes = np.zeros((N))
        self.lap_times = np.zeros((laps, N))
        self.endings = np.zeros((laps, N)) #store env reward
        self.lap_times = [[] for i in range(N)]
        self.N = N
 
    def save_txt_results(self):
        test_name = 'Evals/' + self.eval_name + '.txt'
        with open(test_name, 'w') as file_obj:
            file_obj.write(f"\nTesting Complete \n")
            file_obj.write(f"Map name: {name} \n")
            file_obj.write(f"-----------------------------------------------------\n")
            file_obj.write(f"-----------------------------------------------------\n")
            for i in range(self.N):
                file_obj.write(f"Vehicle: {self.vehicle_list[i].name}\n")
                file_obj.write(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}\n")
                percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
                file_obj.write(f"% Finished = {percent:.2f}\n")
                file_obj.write(f"Avg lap times: {np.mean(self.lap_times[i])}\n")
                file_obj.write(f"-----------------------------------------------------\n")

    def print_results(self):
        print(f"\nTesting Complete ")
        print(f"-----------------------------------------------------")
        print(f"-----------------------------------------------------")
        for i in range(self.N):
            print(f"Vehicle: {self.vehicle_list[i].name}")
            print(f"Crashes: {self.crashes[i]} --> Completes {self.completes[i]}")
            percent = (self.completes[i] / (self.completes[i] + self.crashes[i]) * 100)
            print(f"% Finished = {percent:.2f}")
            print(f"Avg lap times: {np.mean(self.lap_times[i])}")
            print(f"-----------------------------------------------------")
        
    def save_csv_results(self):
        test_name = 'Evals/'  + self.eval_name + '.csv'

        data = [["#", "Name", "%Complete", "AvgTime"]]
        for i in range(self.N):
            v_data = [i]
            v_data.append(self.vehicle_list[i].name)
            v_data.append((self.completes[i] / (self.completes[i] + self.crashes[i]) * 100))
            v_data.append(np.mean(self.lap_times[i]))
            data.append(v_data)

        with open(test_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

    # def load_csv_data(self, eval_name):
    #     file_name = 'Vehicles/Evals' + eval_name + '.csv'

    #     with open(file_name, 'r') as csvfile:
    #         csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
            
    #         for lines in csvFile:  
    #             self.
    #             rewards.append(lines)


    # def plot_eval(self):
    #     pass


class TestVehicles(TestData):
    def __init__(self, config, eval_name) -> None:
        self.config = config
        self.eval_name = eval_name
        self.vehicle_list = []
        self.N = None

        TestData.__init__(self)

    def add_vehicle(self, vehicle):
        self.vehicle_list.append(vehicle)

    def run_eval(self, laps=100, show=False):
        N = self.N = len(self.vehicle_list)
        self.init_arrays(N, laps)

        env_map = ForestMap(self.config)
        env = ForestSim(env_map)    

        for i in range(laps):
        # if add_obs:
            # env_map.reset_map()
            for j in range(N):
                vehicle = self.vehicle_list[j]

                r, steps = self.run_lap(vehicle, env, show)
                print(f"#{i}: Lap time for ({vehicle.name}): {env.steps} --> Reward: {r}")
                self.endings[i, j] = r
                if r == -1 or r == 0:
                    self.crashes[j] += 1
                else:
                    self.completes[j] += 1
                    self.lap_times[j].append(steps)

        self.print_results()
        self.save_txt_results()
        self.save_csv_results()

    def run_lap(self, vehicle, env, show=False):
        vehicle.reset_lap()
        wpts = vehicle.init_agent(env.env_map)
        done, state, score = False, env.reset(), 0.0
        # env.render(wait=True)
        while not done:
            a = vehicle.act(state)
            s_p, r, done, _ = env.step(a)
            state = s_p
            # env.render(False, wpts)

        if show:
            # vehicle.show_vehicle_history()
            # env.show_history()
            env.history.show_history()
            # env.render(wait=False)
            env.render(wait=True)

        return r, env.steps



def load_config(fname):
    with open('config/' + fname + '.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    return conf_dict


""" Training sets"""
def train_gen_std():
    load = False

    agent_name = "GenStd_0_02_0"
    config = load_config("std_config")
    vehicle = GenVehicle(config, agent_name, load)
    reward = StdNavReward(config, 0, 0.2, 0)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_gen_cth():
    load = False

    agent_name = "GenCth_1_1_1"
    config = load_config("std_config")
    vehicle = GenVehicle(config, agent_name, load)
    reward = CrossTrackHeadingReward(config, 1, 1, 1)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_gen_steer():
    load = False

    agent_name = "GenSteer_02_02"
    config = load_config("std_config")
    vehicle = GenVehicle(config, agent_name, load)
    reward = OnlineSteering(config, 0.2, 0.2)

    TrainVehicle(config, agent_name, vehicle, reward)

"""Mod training"""
def train_mod_std():
    load = False

    agent_name = "ModStd_04_02_0"
    config = load_config("std_config")
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = ModStdTimeReward(config, 0.4, 0.2, 0)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_mod_time():
    load = False

    agent_name = "ModTime_01_01_1"
    config = load_config("std_config")
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = ModStdTimeReward(config, 0.1, 0.1, 1)

    TrainVehicle(config, agent_name, vehicle, reward)

def train_mod_cth():
    load = False

    agent_name = "ModCth_1_1_1"
    config = load_config("std_config")
    vehicle = ModVehicleTrain(config, agent_name, load)
    reward = ModHeadingReward(config, 1, 1, 1)

    TrainVehicle(config, agent_name, vehicle, reward)


def test_compare():
    config = load_config("std_config")
    test = TestVehicles(config, 'RaceComparison_t')

    agent_name = "GenStd_0_02_0"
    vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "GenCth_1_1_1"
    # vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "GenSteer_02_02"
    vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "ModStd_04_02_0"
    vehicle = ModVehicleTest(config, agent_name)
    test.add_vehicle(vehicle)

    agent_name = "ModTime_01_01_1"
    # vehicle = ModVehicleTest(config, agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "ModCth_1_1_1"
    # vehicle = GenTest(config, agent_name)
    # test.add_vehicle(vehicle)



    test.run_eval(10, True)


def train():
    pass
    train_gen_std()
    train_gen_steer()
    # train_gen_cth()

    train_mod_std()
    # train_mod_cth()
    # train_mod_time()


if __name__ == "__main__":
    # train()

    test_compare()


