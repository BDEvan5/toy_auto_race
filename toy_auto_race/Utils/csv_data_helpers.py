import csv 
import numpy as np 
import toy_auto_race.Utils.LibFunctions as lib
import matplotlib.pyplot as plt



"""Helpers"""
def save_csv_data(rewards, path, l=10):
    data = []
    for i in range(len(rewards)):
        data.append([i*l, rewards[i]])
    with open(path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["x", "y"])
        csvwriter.writerows(data)



def get_moving_avg(vehicle_name, show=False):
    # _{vehicle_name}
    path = 'Vehicles/' + vehicle_name + f"/training_data.csv"
    smoothpath = 'Vehicles/' + vehicle_name + f"/TrainingDataSmooth.csv"
    rewards = []
    with open(path, 'r') as csvfile:
        csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
        for lines in csvFile:  
            rewards.append(lines)
    rewards = np.array(rewards)[:, 1]

    smooth_rewards = lib.get_moving_average(100, rewards)

    new_rewards = []
    l = 30
    N = int(len(smooth_rewards) / l)
    for i in range(N):
        avg = np.mean(smooth_rewards[i*l:(i+1)*l])
        new_rewards.append(avg)
    smooth_rewards = np.array(new_rewards)

    save_csv_data(smooth_rewards, smoothpath, l)

    if show:
        lib.plot_no_avg(rewards, figure_n=1)
        lib.plot_no_avg(smooth_rewards, figure_n=2)
        plt.show()


# get_moving_avg("")
if __name__ == "__main__":
    get_moving_avg("ModRepeat_forest_3")

