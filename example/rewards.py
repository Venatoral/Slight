"""A series of reward functions."""

import numpy as np


def desired_velocity(env, fail=False, edge_list=None):
    r"""Encourage proximity to a desired velocity.

    This function measures the deviation of a system of vehicles from a
    user-specified desired velocity peaking when all vehicles in the ring
    are set to this desired velocity. Moreover, in order to ensure that the
    reward function naturally punishing the early termination of rollouts due
    to collisions or other failures, the function is formulated as a mapping
    :math:`r: \\mathcal{S} \\times \\mathcal{A}
    \\rightarrow \\mathbb{R}_{\\geq 0}`.
    This is done by subtracting the deviation of the system from the
    desired velocity from the peak allowable deviation from the desired
    velocity. Additionally, since the velocity of vehicles are
    unbounded above, the reward is bounded below by zero,
    to ensure nonnegativity.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system
    edge_list : list  of str, optional
        list of edges the reward is computed over. If no edge_list is defined,
        the reward is computed over all edges

    Returns
    -------
    float
        reward value
    """
    if edge_list is None:
        veh_ids = env.k.vehicle.get_ids()
    else:
        veh_ids = env.k.vehicle.get_ids_by_edge(edge_list)

    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_vehicles = len(veh_ids)

    if any(vel < -100) or fail or num_vehicles == 0:
        return 0.

    target_vel = env.env_params.additional_params['target_velocity']
    max_cost = np.array([target_vel] * num_vehicles)
    max_cost = np.linalg.norm(max_cost)

    cost = vel - target_vel
    cost = np.linalg.norm(cost)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    return max(max_cost - cost, 0) / (max_cost + eps)


def average_velocity(env, fail=False):
    """Encourage proximity to an average velocity.

    This reward function returns the average velocity of all
    vehicles in the system.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    fail : bool, optional
        specifies if any crash or other failure occurred in the system

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    if any(vel < -100) or fail:
        return 0.
    if len(vel) == 0:
        return 0.

    return np.mean(vel)


def rl_forward_progress(env, gain=0.1):
    """Rewared function used to reward the RL vehicles for travelling forward.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain : float
        specifies how much to reward the RL vehicles

    Returns
    -------
    float
        reward value
    """
    rl_velocity = env.k.vehicle.get_speed(env.k.vehicle.get_rl_ids())
    rl_norm_vel = np.linalg.norm(rl_velocity, 1)
    return rl_norm_vel * gain


def boolean_action_penalty(discrete_actions, gain=1.0):
    """Penalize boolean actions that indicate a switch."""
    return gain * np.sum(discrete_actions)


def min_delay(env):
    """Reward function used to encourage minimization of total delay.

    This function measures the deviation of a system of vehicles from all the
    vehicles smoothly travelling at a fixed speed to their destinations.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.network.speed_limit(edge)
        for edge in env.k.network.get_edge_list())
    time_step = env.sim_step

    max_cost = time_step * sum(vel.shape)

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    cost = time_step * sum((v_top - vel) / v_top)
    return max((max_cost - cost) / (max_cost + eps), 0)


def avg_delay_specified_vehicles(env, veh_ids):
    """Calculate the average delay for a set of vehicles in the system.

    Parameters
    ----------
    env: flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    veh_ids: a list of the ids of the vehicles, for which we are calculating
        average delay
    Returns
    -------
    float
        average delay
    """
    sum = 0
    for edge in env.k.network.get_edge_list():
        for veh_id in env.k.vehicle.get_ids_by_edge(edge):
            v_top = env.k.network.speed_limit(edge)
            sum += (v_top - env.k.vehicle.get_speed(veh_id)) / v_top
    time_step = env.sim_step
    try:
        cost = time_step * sum
        return cost / len(veh_ids)
    except ZeroDivisionError:
        return 0


def min_delay_unscaled(env):
    """Return the average delay for all vehicles in the system.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.

    Returns
    -------
    float
        reward value
    """
    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.network.speed_limit(edge)
        for edge in env.k.network.get_edge_list())
    time_step = env.sim_step

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    cost = time_step * sum((v_top - vel) / v_top)
    # print("min_delay_unscale")
    # print(cost / (env.k.vehicle.num_vehicles + eps))
    return cost / (env.k.vehicle.num_vehicles + eps)


def penalize_standstill(env, gain=1):
    """Reward function that penalizes vehicle standstill.

    Is it better for this to be:
        a) penalize standstill in general?
        b) multiplicative based on time that vel=0?

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    gain : float
        multiplicative factor on the action penalty

    Returns
    -------
    float
        reward value
    """
    veh_ids = env.k.vehicle.get_ids()
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    num_standstill = len(vel[vel == 0])
    penalty = gain * num_standstill
    # print("penalize_stanstill")
    # print(-penalty)
    return -penalty


def penalize_near_standstill(env, thresh=0.3, gain=1):
    """Reward function which penalizes vehicles at a low velocity.

    This reward function is used to penalize vehicles below a
    specified threshold. This assists with discouraging RL from
    gamifying a network, which can result in standstill behavior
    or similarly bad, near-zero velocities.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
    thresh : float
        the velocity threshold below which penalties are applied
    gain : float
        multiplicative factor on the action penalty
    """
    veh_ids = env.k.vehicle.get_ids()
    vel = np.array(env.k.vehicle.get_speed(veh_ids))
    penalize = len(vel[vel < thresh])
    penalty = gain * penalize
    return -penalty


def penalize_headway_variance(vehicles,
                              vids,
                              normalization=1,
                              penalty_gain=1,
                              penalty_exponent=1):
    """Reward function used to train rl vehicles to encourage large headways.

    Parameters
    ----------
    vehicles : flow.core.kernel.vehicle.KernelVehicle
        contains the state of all vehicles in the network (generally
        self.vehicles)
    vids : list of str
        list of ids for vehicles
    normalization : float, optional
        constant for scaling (down) the headways
    penalty_gain : float, optional
        sets the penalty for each vehicle between 0 and this value
    penalty_exponent : float, optional
        used to allow exponential punishing of smaller headways
    """
    headways = penalty_gain * np.power(
        np.array(
            [vehicles.get_headway(veh_id) / normalization
             for veh_id in vids]), penalty_exponent)
    return -np.var(headways)


def punish_rl_lane_changes(env, penalty=1):
    """Penalize an RL vehicle performing lane changes.

    This reward function is meant to minimize the number of lane changes and RL
    vehicle performs.

    Parameters
    ----------
    env : flow.envs.Env
        the environment variable, which contains information on the current
        state of the system.
    penalty : float, optional
        penalty imposed on the reward function for any rl lane change action
    """
    total_lane_change_penalty = 0
    for veh_id in env.k.vehicle.get_rl_ids():
        if env.k.vehicle.get_last_lc(veh_id) == env.timer:
            total_lane_change_penalty -= penalty

    return total_lane_change_penalty

def wait_time(env,distance=50):
    ids = env.k.vehicle.get_ids()
    return 



def wait_num_cars(env ,rl_actions, **kwargs):##不共享
##################################
    rew = {}
    num = 0
    total_rew = 0
    for rl_id in range(len(rl_actions)):
        traffics = gen_traffics(rl_id)
#        print("rl_id:"+rl_id)
 #       print("traffic")
 #       print(traffics)
        vehs_on_edge = env.k.vehicle.get_ids_by_edge(traffics)
        pos = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge
        ])
        speed = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge
        ])
        for i in range(len(pos)):
            if pos[i] > 170 and speed[i] < 1:
                num += 1
        rew[rl_id] = num
        total_rew += num
  #      print("rew{rl_id}")
  #      print(rew)
 #       print(num)
        rew[rl_id] = -rew[rl_id]
#        print("rew")
 #       print(rew)
######################################
    # total_rew + 1 / (1 + total_rew)
    return rew










def wait_num(env ,rl_actions, **kwargs):
##################################
    rew = {}
    up  = 3
    # print(rl_actions)
    if len(rl_actions) == 4:
        up = 2
    if env.discrete:
        num_agent = 3
    else:
        num_agent = len(rl_actions)
    for rl_id in range(num_agent):
        num_horizon1 = 0
        num_horizon2 = 0
        num_vertical1 = 0
        num_vertical2 = 0
        traffics = gen_traffics(rl_id,up)
        # print(rl_id)
        # print(traffics)

#        print("rl_id:"+rl_id)
 #       print("traffic")
 #       print(traffics)
        vehs_on_edge_horizon1 = env.k.vehicle.get_ids_by_edge(traffics[1])
        vehs_on_edge_horizon2 = env.k.vehicle.get_ids_by_edge(traffics[2])
        pos_horizon1 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_horizon1
        ])
        speed_horizon1 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_horizon1
        ])
        pos_horizon2 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_horizon2
        ])
        speed_horizon2 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_horizon2
        ])
        for i in range(len(pos_horizon1)):
            if pos_horizon1[i] > 0 and speed_horizon1[i] < 3:
                num_horizon1 += 1
        for i in range(len(pos_horizon2)):
            if pos_horizon2[i] > 0 and speed_horizon2[i] < 3:
                num_horizon2 += 1

        vehs_on_edge_vertical1 = env.k.vehicle.get_ids_by_edge(traffics[0])
        vehs_on_edge_vertical2 = env.k.vehicle.get_ids_by_edge(traffics[3])
        pos_vertical1 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_vertical1
        ])
        speed_vertical1 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_vertical1
        ])
        pos_vertical2 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_vertical2
        ])
        speed_vertical2 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_vertical2
        ])
        for i in range(len(pos_vertical1)):
            if pos_vertical1[i] > 0 and speed_vertical1[i] < 3:
                num_vertical1 += 1
        for i in range(len(pos_vertical2)):
            if pos_vertical2[i] > 0 and speed_vertical2[i] < 3:
                num_vertical2 += 1
        rew[rl_id] = max(num_horizon1,num_horizon2) ** 1.2 + max(num_vertical1,num_vertical2) ** 1.2
        # rew[rl_id] = (num_horizon1 + num_horizon2) ** 1.2 + (num_vertical1 + num_vertical2) ** 1.2
        rew[rl_id] = -1/2 * rew[rl_id] 
    return rew



def wait_mean_num(env ,rl_actions, **kwargs):
##################################
    # ob = 50   ##此为观察的车辆 和 state中观察的车辆对应
    rew =[[] for i in range(len(rl_actions))]
    # print(rl_actions)
    for rl_id in range(len(rl_actions)):
        num_horizon1 = 0
        num_horizon2 = 0
        num_vertical1 = 0
        num_vertical2 = 0
        up = 3
        if len(rl_actions) == 4:
            up = 2
        else:
            up = 3
            
        traffics = gen_traffics(rl_id,up)
        # print(rl_id)
        # print(traffics)
        # print(rl_id)
        # print(traffics)

#        print("rl_id:"+rl_id)
 #       print("traffic")
 #       print(traffics)
        vehs_on_edge_horizon1 = env.k.vehicle.get_ids_by_edge(traffics[1])
        vehs_on_edge_horizon2 = env.k.vehicle.get_ids_by_edge(traffics[2])
        pos_horizon1 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_horizon1
        ])
        speed_horizon1 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_horizon1
        ])
        pos_horizon2 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_horizon2
        ])
        speed_horizon2 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_horizon2
        ])
        for i in range(len(pos_horizon1)):
            if pos_horizon1[i] > 0 and speed_horizon1[i] < 3:
                num_horizon1 += 1
        for i in range(len(pos_horizon2)):
            if pos_horizon2[i] > 0 and speed_horizon2[i] < 3:
                num_horizon2 += 1

        vehs_on_edge_vertical1 = env.k.vehicle.get_ids_by_edge(traffics[0])
        vehs_on_edge_vertical2 = env.k.vehicle.get_ids_by_edge(traffics[3])
        pos_vertical1 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_vertical1
        ])
        speed_vertical1 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_vertical1
        ])
        pos_vertical2 = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge_vertical2
        ])
        speed_vertical2 = np.array([
            env.k.vehicle.get_speed(veh_id)
            for veh_id in vehs_on_edge_vertical2
        ])
        for i in range(len(pos_vertical1)):
            if pos_vertical1[i] > 0 and speed_vertical1[i] < 3:
                num_vertical1 += 1
        for i in range(len(pos_vertical2)):
            if pos_vertical2[i] > 0 and speed_vertical2[i] < 3:
                num_vertical2 += 1
        
        # if num_horizon1 > ob:
        #     num_horizon1 = ob
        # if num_horizon2 > ob:
        #     num_horizon2 = ob
        # if num_vertical1 > ob:
        #     num_vertical1 = ob
        # if num_vertical2 > ob:
        #     num_vertical2 = ob
        # print(num_horizon1,num_horizon2,num_vertical1,num_vertical2)
        rew[rl_id] = np.mean([num_horizon1,num_horizon2]) ** 1.2 + np.mean([num_vertical1,num_vertical2]) ** 1.2
        rew[rl_id] = -1/2 * rew[rl_id] 
    # print(rew)
    # print(rl_actions)
    return rew


def penalize_delay(env ,rl_actions, **kwargs):

    vel = np.array(env.k.vehicle.get_speed(env.k.vehicle.get_ids()))

    vel = vel[vel >= -1e-6]
    v_top = max(
        env.k.network.speed_limit(edge)
        for edge in env.k.network.get_edge_list())
    time_step = env.sim_step * env.EnvParams.sims_per_step

    # epsilon term (to deal with ZeroDivisionError exceptions)
    eps = np.finfo(np.float32).eps

    cost = time_step * sum((v_top - vel) / v_top)
    return cost / (env.k.vehicle.num_vehicles + eps)















def share_wait_num_cars(env ,rl_actions, **kwargs):##共享
##################################
    rew = {}
    num = 0
    for rl_id in rl_actions.keys():
        traffic_id = rl_id.split('center')[1]
        traffics = gen_traffics(traffic_id)
        print(traffic_id)
        print(traffics)
#        print("rl_id:"+rl_id)
 #       print("traffic")
 #       print(traffics)
        vehs_on_edge = env.k.vehicle.get_ids_by_edge(traffics)
        pos = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge
        ])
        for i in pos:
            if i > 170:
                num += 1
        rew[rl_id] = num
  #      print("rew{rl_id}")
  #      print(rew)
 #       print(num)
 #       rew[rl_id] = 1 / (1 + rew[rl_id])
#        print("rew")
 #       print(rew)
    rews = {}
    rews["center0"] = 0.7*rew["center0"] + 0.3*rew["center1"]
    rews["center1"] = 0.7*rew["center1"] + 0.15*rew["center0"]+ 0.15*rew["center2"]
    rews["center2"] = 0.7*rew["center2"] + 0.3*rew["center1"]    
    rews["center0"] = 1 / (1 + rews["center0"])
    rews["center1"] = 1 / (1 + rews["center1"])
    rews["center2"] = 1 / (1 + rews["center2"])
######################################
    
    return rews


def mix_webster_and_pnishNumWait(env):
    return

###################

def rlcar_wait_num_car(env ,rl_actions, **kwargs):
    ##################################
    rew = {}
    num = 0
    for rl_id in env.k.vehicle.get_rl_ids():
        traffics = gen_traffics(0)
        traffics += gen_traffics(1)
        traffics += gen_traffics(2)
    #        print("rl_id:"+rl_id)
    #       print("traffic")
    #       print(traffics)
        vehs_on_edge = env.k.vehicle.get_ids_by_edge(traffics)
        pos = np.array([
            env.k.vehicle.get_position(veh_id)
            for veh_id in vehs_on_edge
        ])
        for i in pos:
            if i > 170:
                num += 1
        rew[rl_id] = num
    #      print("rew{rl_id}")
    #      print(rew)
    #       print(num)
        rew[rl_id] = 1 / (1 + rew[rl_id])
    #        print("rew")
    #       print(rew)
    ######################################
    return rew
        
def gen_traffics(i,up = 3):
    i = int(i)
    x = int(i / up)
    y = i % up
    _x = x+1
    _y = y+1
    return ['bot{}_{}'.format(str(x),str(y)), 'right{}_{}'.format(str(x),str(y)),
                'left{}_{}'.format(str(_x),str(y)), 'top{}_{}'.format(str(x),str(_y))]
###########################