
from TrainTest import test_single_vehicle


def generat_oracle_data(env, oracle_vehicle, imitation_vehicle, steps):
    done = False
    state = env.reset()
    oracle_vehicle.plan_forest(env.env_map)
    # while not oracle_vehicle.plan(env.env_map):
    #     state = env.reset() 

    for n in range(steps):
        a = oracle_vehicle.plan_act(state)
        imitation_vehicle.save_step(state, a)

        s_prime, r, done, _ = env.step_plan(a)
        state = s_prime
        
        if done:
            env.render(wait=False)

            state = env.reset()
            oracle_vehicle.plan_forest(env.env_map)
            # while not oracle_vehicle.plan(env.env_map):
            #     state = env.reset()

        if n % 200 == 1:
            print(f"Filling buffer: {n}")
            
def generate_imitation_data(env, oracle_vehicle, imitation_vehicle, steps):
    done = False
    state = env.reset()
    oracle_vehicle.plan(env.env_map)
    # while not oracle_vehicle.plan(env.env_map):
    #     state = env.reset() 

    for n in range(steps):
        a = oracle_vehicle.plan_act(state)
        imitation_vehicle.save_step(state, a) # save oracle

        action = imitation_vehicle.plan_act(state) # act on imitation
        s_prime, r, done, _ = env.step_plan(action)
        state = s_prime
        
        if done:
            env.render(wait=False)

            state = env.reset()
            oracle_vehicle.plan(env.env_map)
            # while not oracle_vehicle.plan(env.env_map):
            #     state = env.reset()

        if n % 200 == 1:
            print(f"Filling buffer: {n}")



# train imitation vehicle
def train_imitation_vehicle(env, oracle_vehicle, imitation_vehicle, batches=50, steps=500):

    

    for i in range(batches):
        generate_imitation_data(env, oracle_vehicle, imitation_vehicle, steps)
        imitation_vehicle.train()
        imitation_vehicle.save()
        test_single_vehicle(env, imitation_vehicle, False, 20)


