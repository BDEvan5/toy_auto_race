from rockit import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import pi, cos, sin, tan, square
from casadi import vertcat, horzcat, sumsqr




import LibFunctions as lib
from TrajectoryPlanner import MinCurvatureTrajectory
from Simulator import ScanSimulator


Nsim    = 30            # how much samples to simulate
L       = 0.5             # bicycle model length
nx      = 3             # the system is composed of 3 states
nu      = 2             # the system has 2 control inputs
N       = 10            # number of control intervals

class RockitMPC:
    def __init__(self) -> None:
        self.ocp = Ocp(T=FreeTime(10.0))

        # Define states
        self.x     = self.ocp.state()
        self.y     = self.ocp.state()
        self.theta = self.ocp.state()
        self.X = vertcat(self.x, self.y, self.theta)

        # Defince controls
        self.delta = self.ocp.control()
        self.V     = self.ocp.control(order=0)

        # Define physical path parameter
        self.waypoints = self.ocp.parameter(2, grid='control')
        self.waypoint_last = self.ocp.parameter(2)
        self.p = vertcat(self.x, self.y)

        # Define parameter
        self.X_0 = self.ocp.parameter(nx)

        self.init_mpc()

    def init_mpc(self):
        # Specify ODE
        self.ocp.set_der(self.x,      self.V*cos(self.theta))
        self.ocp.set_der(self.y,      self.V*sin(self.theta))
        self.ocp.set_der(self.theta,  self.V/L*tan(self.delta))

        # Initial constraints
        self.ocp.subject_to(self.ocp.at_t0(self.X) == self.X_0)

        # Initial guess
        self.ocp.set_initial(self.x,      0)
        self.ocp.set_initial(self.y,      0)
        self.ocp.set_initial(self.theta,  0)

        self.ocp.set_initial(self.V,    0.5)

        # Path constraints
        max_v = 5
        self.ocp.subject_to( 0 <= (self.V <= max_v) )
        #ocp.subject_to( -0.3 <= (ocp.der(V) <= 0.3) )
        self.ocp.subject_to( -pi/6 <= (self.delta <= pi/6) )

        # Minimal time
        self.ocp.add_objective(0.50*self.ocp.T)

        self.ocp.add_objective(self.ocp.sum(sumsqr(self.p-self.waypoints), grid='control'))
        self.ocp.add_objective(sumsqr(self.ocp.at_tf(self.p)-self.waypoint_last))

        # Pick a solution method
        options = {"ipopt": {"print_level": 0}}
        options["expand"] = True
        options["print_time"] = False
        self.ocp.solver('ipopt', options)

        # Make it concrete for this ocp
        self.ocp.method(MultipleShooting(N=N, M=1, intg='rk', grid=FreeGrid(min=0.05, max=2)))

    def set_current_waypoints(self, current_waypoints):
        self.ocp.set_value(self.waypoints,current_waypoints[:,:-1])
        self.ocp.set_value(self.waypoint_last,current_waypoints[:,-1])

    def set_x0(self, current_X):
        self.ocp.set_value(self.X_0, current_X)

    def solve(self):
        sol = self.ocp.solve()

        t_sol, x_sol     = sol.sample(self.x,     grid='control')
        t_sol, y_sol     = sol.sample(self.y,     grid='control')
        t_sol, theta_sol = sol.sample(self.theta, grid='control')
        t_sol, delta_sol = sol.sample(self.delta, grid='control')
        t_sol, V_sol     = sol.sample(self.V,     grid='control')

        err = sol.value(self.ocp.objective)

        self.ocp.set_initial(self.x, x_sol)
        self.ocp.set_initial(self.y, y_sol)
        self.ocp.set_initial(self.theta, theta_sol)
        self.ocp.set_initial(self.delta, delta_sol)
        self.ocp.set_initial(self.V, V_sol)

        t = np.array(t_sol)
        x = np.array(x_sol)
        y = np.array(y_sol)
        th = np.array(theta_sol)
        d = np.array(delta_sol)
        v = np.array(V_sol)

        return t, x, y, th, d, v, err



class AgentMPC(RockitMPC):
    def __init__(self):
        self.name = "Optimal Agent:  MPC "
        self.env_map = None
        self.path_name = None
        self.wpts = None

        self.pind = 1
        self.target = None
        self.steps = 0

        self.tracking_error = []

        RockitMPC.__init__(self)

    def init_agent(self, env_map):
        self.env_map = env_map

        self.path_name = "DataRecords/" + self.env_map.name + "_path.npy" # move to setup call

        self.wpts = self.env_map.get_optimal_path()

        return self.wpts

    def act(self, obs):
        current_waypoints = self.get_current_wpts(obs)
        self.set_current_waypoints(current_waypoints)
        current_X = vertcat(obs[0], obs[1], obs[2])
        self.set_x0(current_X)
        
        t, x, y, th, d, v, err = self.solve()

        v_ref = v[0]
        d_ref = d[0]

        self.tracking_error.append(err)
        print(f'Tracking error f: {err}' )

        pts = np.concatenate([x[:, None], y[:, None]], axis=-1)

        return [v_ref, d_ref], pts, t, current_waypoints.T


    def get_current_wpts(self, obs):
        dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
        shift_distance = 1
        while dis_cur_target < shift_distance: # how close to say you were there
            if self.pind < len(self.wpts)-2:
                self.pind += 1
                dis_cur_target = lib.get_distance(self.wpts[self.pind], obs[0:2])
            else:
                self.pind = 0

        return self.wpts[self.pind + 1:self.pind + N +2].T

    def reset_lap(self):
        # for testing
        pass    

    def show_vehicle_history(self):
        pass

