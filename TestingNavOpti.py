import numpy as np
import csv

from HistoryStructs import TrainHistory
from Simulator import ForestSim
from SimMaps import  ForestMap
from ModelsRL import ReplayBufferDQN, ReplayBufferTD3
import LibFunctions as lib

# from AgentOptimal import OptimalAgent
from AgentMod import ModVehicleTest, ModVehicleTrain
from RefGen import GenTrainStd, GenTrainStr, GenTest

names = ['columbia', 'levine_blocked', 'mtl', 'porto', 'torino', 'race_track']
name = names[5]
myMap = 'TrackMap1000'
forest_name = 'forest'



"""General test function"""
def testVehicle(vehicle, show=False, obs=True):
    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    crashes = 0
    completes = 0
    lap_times = []

    wpts = vehicle.init_agent(env_map)
    done, state, score = False, env.reset(), 0.0
    for i in range(100): # 10 laps
        print(f"Running lap: {i}")
        # if obs:
        #     env_map.reset_map()
        while not done:
            a = vehicle.act(state)
            s_p, r, done, _ = env.step(a)
            state = s_p
            # env.render(False, vehicle.scan_sim)
        print(f"Lap time updates: {env.steps}")
        if show:
            # vehicle.show_vehicle_history()
            env.render(wait=False)
            # env.render(wait=True)

        if r == -1:
            crashes += 1
        else:
            completes += 1
            lap_times.append(env.steps)
        state = env.reset()
        
        # env.reset_lap()
        env.reset()
        vehicle.reset_lap()
        done = False

    print(f"Crashes: {crashes}")
    print(f"Completes: {completes} --> {(completes / (completes + crashes) * 100):.2f} %")
    print(f"Lap times: {lap_times} --> Avg: {np.mean(lap_times)}")

"""Train"""
def TrainVehicle(agent_name, vehicle):
    path = 'Vehicles/' + agent_name
    buffer = ReplayBufferTD3()

    # env_map = SimMap(name)
    # env = TrackSim(env_map)

    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    t_his = TrainHistory(agent_name)
    print_n = 500

    done, state = False, env.reset()
    wpts = vehicle.init_agent(env_map)

    for n in range(20000):
        a = vehicle.act(state)
        s_prime, r, done, _ = env.step(a)

        new_r = vehicle.add_memory_entry(r, done, s_prime, buffer)
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

    def init_arrays(self, N):
        self.completes = np.zeros((N))
        self.crashes = np.zeros((N))
        self.lap_times = np.zeros((laps, N))
        self.endings = np.zeros((laps, N)) #store env reward
        self.lap_times = [[] for i in range(N)]
        self.N = N
 
    def save_txt_results(self):
        test_name = 'Vehicles/Evals' + self.eval_name + '.txt'
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
        test_name = 'Vehicles/Evals/'  + self.eval_name + '.csv'

        data = ["#", "Name", "%Complete", "AvgTime"]
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
    def __init__(self, eval_name) -> None:
        self.eval_name = eval_name
        self.vehicle_list = []
        self.N = None

        TestData.__init__()


    def add_vehicle(self, vehicle):
        self.vehicle_list.append(vehicle)

    def run_eval(self, laps=100, show=False):
        N = self.N = len(self.vehicle_list)

        env_map = ForestMap(forest_name)
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





def RunVehicleLap(vehicle, env, show=False):
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

def test_vehicles(vehicle_list, laps, eval_name, add_obs):
    N = len(vehicle_list)

    # env_map = SimMap(name)
    env_map = ForestMap(forest_name)
    env = ForestSim(env_map)

    completes = np.zeros((N))
    crashes = np.zeros((N))
    lap_times = np.zeros((laps, N))
    endings = np.zeros((laps, N)) #store env reward
    lap_times = [[] for i in range(N)]

    for i in range(laps):
        # if add_obs:
            # env_map.reset_map()
        for j in range(N):
            vehicle = vehicle_list[j]

            r, steps = RunVehicleLap(vehicle, env, False)
            # r, steps = RunVehicleLap(vehicle, env, True)
            print(f"#{i}: Lap time for ({vehicle.name}): {env.steps} --> Reward: {r}")
            endings[i, j] = r
            if r == -1 or r == 0:
                crashes[j] += 1
            else:
                completes[j] += 1
                lap_times[j].append(steps)

    test_name = 'Vehicles/' + eval_name + '.txt'
    with open(test_name, 'w') as file_obj:
        file_obj.write(f"\nTesting Complete \n")
        file_obj.write(f"Map name: {name} \n")
        file_obj.write(f"-----------------------------------------------------\n")
        file_obj.write(f"-----------------------------------------------------\n")
        for i in range(N):
            file_obj.write(f"Vehicle: {vehicle_list[i].name}\n")
            file_obj.write(f"Crashes: {crashes[i]} --> Completes {completes[i]}\n")
            percent = (completes[i] / (completes[i] + crashes[i]) * 100)
            file_obj.write(f"% Finished = {percent:.2f}\n")
            file_obj.write(f"Avg lap times: {np.mean(lap_times[i])}\n")
            file_obj.write(f"-----------------------------------------------------\n")


    print(f"\nTesting Complete ")
    print(f"-----------------------------------------------------")
    print(f"-----------------------------------------------------")
    for i in range(N):
        print(f"Vehicle: {vehicle_list[i].name}")
        print(f"Crashes: {crashes[i]} --> Completes {completes[i]}")
        percent = (completes[i] / (completes[i] + crashes[i]) * 100)
        print(f"% Finished = {percent:.2f}")
        print(f"Avg lap times: {np.mean(lap_times[i])}")
        print(f"-----------------------------------------------------")




"""Std functions"""
def train_gen_disV_b2():
    load = False

    agent_name = "GenDisV_0_02_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 0.2, 0)

    TrainGenVehicle(agent_name, vehicle)

    agent_name = "GenDisV_0_05_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 0.5, 0)

    TrainGenVehicle(agent_name, vehicle)

    agent_name = "GenDisV_0_06_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 0.6, 0)

    TrainGenVehicle(agent_name, vehicle)
    agent_name = "GenDisV_0_07_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 0.7, 0)

    TrainGenVehicle(agent_name, vehicle)
    agent_name = "GenDisV_0_08_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 0.8, 0)

    TrainGenVehicle(agent_name, vehicle)
    agent_name = "GenDisV_0_09_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 0.9, 0)

    TrainGenVehicle(agent_name, vehicle)

    agent_name = "GenDisV_0_1_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 1, 0)

    TrainGenVehicle(agent_name, vehicle)
    
    agent_name = "GenDisV_0_2_0"
    vehicle = GenTrainDisV(agent_name, load, 200, 10)
    vehicle.init_reward(0, 2, 0)

    TrainGenVehicle(agent_name, vehicle)

    # test_vehicles(vehicle_list, 10, vehicle_name + "/Eval_NoObs", False)

def test_b2():
    vehicle_list = []

    agent_name = "GenDisV_0_02_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV_0_05_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV_0_06_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV_0_07_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV_0_08_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV_0_09_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV_0_1_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV_0_2_0"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)


    test_vehicles(vehicle_list, 100, "OptiV_b1" , True)


"""Mod functions"""
def train_m1():
    load = False

    agent_name = "Mod_0_02"
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    vehicle.init_reward(0, 0.2)
    TrainModVehicle(agent_name, vehicle)

    agent_name = "Mod_01_02"
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    vehicle.init_reward(0.1, 0.2)
    TrainModVehicle(agent_name, vehicle)

    agent_name = "Mod_02_02"
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    vehicle.init_reward(0.2, 0.2)
    TrainModVehicle(agent_name, vehicle)

    agent_name = "Mod_04_02"
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    vehicle.init_reward(0.4, 0.2)
    TrainModVehicle(agent_name, vehicle)

def train_m2():
    load = False
    agent_name = "Mod_02_01"
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    vehicle.init_reward(0.2, 0.1)
    TrainModVehicle(agent_name, vehicle)

    # agent_name = "Mod_02_02"
    # vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    # vehicle.init_reward(0.2, 0.2)
    # TrainModVehicle(agent_name, vehicle)

    agent_name = "Mod_02_03"
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    vehicle.init_reward(0.2, 0.3)
    TrainModVehicle(agent_name, vehicle)

    agent_name = "Mod_02_04"
    vehicle = ModVehicleTrain(agent_name, load, 200, 10)
    vehicle.init_reward(0.2, 0.4)
    TrainModVehicle(agent_name, vehicle)

def test_m():
    vehicle_list = []

    # m2
    agent_name = "Mod_02_01"
    vehicle = ModVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "Mod_02_02"
    vehicle = ModVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "Mod_02_03"
    vehicle = ModVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "Mod_02_04"
    vehicle = ModVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    # m1
    agent_name = "Mod_0_02"
    vehicle = ModVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "Mod_01_02"
    vehicle = ModVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "Mod_04_02"
    vehicle = ModVehicleTest(agent_name)
    vehicle_list.append(vehicle)



    test_vehicles(vehicle_list, 1000, "Mod_m1m2" , True)


"""Steer functions"""
def train_s1():
    load = False

    # agent_name = "Str_01_01_02"
    # vehicle = GenTrainStr(agent_name, load, 200, 10)
    # vehicle.init_reward(0.1, 0.1, 0.2)
    # TrainVehicle(agent_name, vehicle)

    agent_name = "Str_01_01_08"
    vehicle = GenTrainStr(agent_name, load, 200, 10)
    vehicle.init_reward(0.1, 0.1, 0.8)
    TrainVehicle(agent_name, vehicle)

    # agent_name = "Str_01_01_02"
    # vehicle = GenTrainStr(agent_name, load, 200, 10)
    # vehicle.init_reward(0, 1)
    # TrainVehicle(agent_name, vehicle)

    # agent_name = "Str_01_01_02"
    # vehicle = GenTrainStr(agent_name, load, 200, 10)
    # vehicle.init_reward(0, 1)
    # TrainVehicle(agent_name, vehicle)

def test_s():
    test = TestVehicles("test_s_1")

    # agent_name = "Str_01_01_02"
    # vehicle = GenTest(agent_name)
    # test.add_vehicle(vehicle)

    agent_name = "Str_01_01_08"
    vehicle = GenTest(agent_name)
    test.add_vehicle(vehicle)


    test.run_eval(10, False)


"""Under development still"""
def train_gen_dis():
    load = False
    agent_name = "GenDis"
    vehicle = GenTrainDis(agent_name, load, 200, 10)

    TrainGenVehicle(agent_name, vehicle)

def train_gen_steer():
    load = False
    agent_name = "GenSteer"
    vehicle = GenTrainSteer(agent_name, load, 200, 10)

    TrainGenVehicle(agent_name, vehicle)

def test_gen():
    vehicle_list = []

    agent_name = "GenDis"
    vehicle = GenVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenSteer"
    vehicle = GenVehicleTest(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenDisV"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)

    agent_name = "GenSteerV"
    vehicle = GenVehicleTestV(agent_name)
    vehicle_list.append(vehicle)


    # test_vehicles(vehicle_list, 100, vehicle_name + "/Eval_Obs" , True)
    test_vehicles(vehicle_list, 10, "Eval_gen" , True)

    # test_vehicles(vehicle_list, 10, vehicle_name + "/Eval_NoObs", False)


def train_gen_steerV():
    load = False
    agent_name = "GenSteerV"
    vehicle = GenTrainSteerV(agent_name, load, 200, 10)

    TrainGenVehicle(agent_name, vehicle)





if __name__ == "__main__":


    # train_gen_disV_b2()
    # test_b2()

    # train_m1()
    # train_m2()

    # test_m()

    train_s1()
    # train_s2()
    # train_s3()

    test_s()

















