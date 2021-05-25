from toy_auto_race.NavAgents.FollowTheGap import ForestFGM
from toy_auto_race.Utils import LibFunctions as lib
from toy_f110 import ForestSim
# from toy_auto_race.NavAgents.SafetyCar import SafetyCar
# from toy_auto_race.NavAgents.SafetyEnvelope import SafetyCar
# from toy_auto_race.NavAgents.SuperSafetySystem import SafetyCar
# from toy_auto_race.NavAgents.DynamicWindow import SafetyCar
from toy_auto_race.NavAgents.TwoDWA import SafetyCar

from TrainTest import *


def test_safety_system():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim("forest2", sim_conf)
    vehicle = SafetyCar(sim_conf)

    test_single_vehicle(env, vehicle, True, 100, wait=False)


def test_forest_system():
    sim_conf = lib.load_conf("fgm_config")
    env = ForestSim("forest2", sim_conf)
    vehicle = ForestFGM()

    test_single_vehicle(env, vehicle, True, 100)



if __name__ == "__main__":
    test_safety_system()
    # test_forest_system()


