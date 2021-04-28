import timeit

from toy_auto_race.NavAgents.Oracle import Oracle
from toy_auto_race.NavAgents.AgentNav import NavTrainVehicle, NavTestVehicle

from toy_auto_race.NavAgents.AgentMod import ModVehicleTest, ModVehicleTrain
from toy_auto_race.NavAgents.FollowTheGap import ForestFGM

from toy_auto_race.TD3 import ReplayBufferTD3, TD3

from toy_f110 import ForestSim


def simulator():
    action = [0, 3]
    env = ForestSim("forest2")
    env.reset()
    for i in range(100):
        s = env.step_plan(action)

def buffer():
    buffer = ReplayBufferTD3()
    state = [i for i in range(14)]
    s_p = [i for i in range(14)]
    data = (state, [0, 1], s_p, 1, False)

    for i in range(100):
        buffer.add(data)
        s = buffer.sample(100)

def td3():
    state = [i for i in range(14)]
    s_p = [i for i in range(14)]
    data = (state, [0, 1], s_p, 1, False)

    agent = TD3(14, 1, 1, "test")


    for i in range(10):
        agent.replay_buffer.add(data)

    for i in range(100):
        agent.train(2)


def time_simulator():
    t = timeit.timeit(stmt=simulator, number=100)
    print(f"Simulator: {t}")

def time_buffer():
    t = timeit.timeit(stmt=buffer, number=100)
    print(f"Buffer: {t}")

def time_td3():
    t = timeit.timeit(stmt=td3, number=100)
    print(f"TD3: {t}")


if __name__ == '__main__':
    # time_simulator()
    # time_buffer()
    time_td3()

