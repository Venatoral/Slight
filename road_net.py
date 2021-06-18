from our_env import SeqTrafficLightEnv
from flow.core.params import InFlows, SumoParams
from flow.core.params import NetParams, InitialConfig
from flow.core.params import EnvParams, SumoCarFollowingParams, VehicleParams
from flow.controllers import GridRouter
from flow.networks import TrafficLightGridNetwork
from flow.core.experiment import Experiment


# 路网参数
ROAD_PARAMS = dict(
    # 车辆进入速度
    v_enter=15,
    # 路网长度参数
    inner_length=300,
    long_length=500,
    short_length=300,
    # 路网行数
    n_rows=6,
    # 路网列数
    n_columns=6,
    # 各个方向车辆数量
    num_cars_left=20,
    num_cars_right=20,
    num_cars_top=20,
    num_cars_bot=20
)


# 总的车辆数
ROAD_PARAMS['total_cars'] = (
    (ROAD_PARAMS['num_cars_left'] +
     ROAD_PARAMS['num_cars_right']) * ROAD_PARAMS['n_columns']
    + (ROAD_PARAMS['num_cars_top'] +
       ROAD_PARAMS['num_cars_bot']) * ROAD_PARAMS['n_rows']
)

'''
根据num_row和num_col构造inflow的edges
'''
def get_edges(num_row: int, num_col: int):
    edges = []

    # 首先根据 left, right方向添加inflow
    for i in range(num_col):
        edges.extend(['left{}_{}'.format(num_row, i)])
        edges.extend(['right0_{}'.format(i)])
    # 然后是 top, bottom 方向
    for i in range(num_row):
        edges.extend(['bot{}_0'.format(i)])
        edges.extend(['top{}_{}'.format(i, num_col)])
    return edges


'''
获取flow_params
'''
def get_flow_params(num_row: int, num_col: int, additional_net_params):
    edges = get_edges(num_row=num_row, num_col=num_col)
    inflow = InFlows()
    # 添加inflow车流
    for i in range(len(edges)):
        inflow.add(
            veh_type='human',
            edge=edges[i],
            depart_lane='free',
            depart_speed=20,
            vehs_per_hour=800,
            # probability=0.1,
        )
    initial = InitialConfig(
        shuffle=True,
        spacing='custom',
        # float('inf') means to distribute cars on all lanes
        lanes_distribution=float('inf'),
    )
    net = NetParams(inflows=inflow, additional_params=additional_net_params)
    return initial, net


# 车辆信息
vehicles = VehicleParams()
vehicles.add(
    veh_id='human',
    routing_controller=(GridRouter, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        decel=7.5
    ),
    # num_vehicles=ROAD_PARAMS['total_cars']
    num_vehicles=0
)

# 交叉路口网络参数
grid_array = {
    "short_length": ROAD_PARAMS['short_length'],
    "inner_length": ROAD_PARAMS['inner_length'],
    "long_length": ROAD_PARAMS['long_length'],
    "row_num": ROAD_PARAMS['n_rows'],
    "col_num": ROAD_PARAMS['n_columns'],
    "cars_left": ROAD_PARAMS['num_cars_left'],
    "cars_right": ROAD_PARAMS['num_cars_right'],
    "cars_top": ROAD_PARAMS['num_cars_top'],
    "cars_bot": ROAD_PARAMS['num_cars_bot'],
}
# 网络一些其他参数
additional_net_params = {
    'grid_array': grid_array,
    'speed_limit': 35,
    'horizontal_lanes': 1,
    'vertical_lanes': 1,
    'tl_lights': True
}

init, net_params = get_flow_params(num_row=ROAD_PARAMS['n_rows'],
                                   num_col=ROAD_PARAMS['n_columns'],
                                   additional_net_params=additional_net_params
                                   )

# 训练环境参数
additional_env_params = {
        'target_velocity': 50,
        'switch_time': 3.0,
        'num_observed': 4,
        # 使用离散值来表示action_space
        'discrete': False,
        'tl_type': 'controlled',
        # number of nearest lights to observe, defaults to 4
        'num_local_lights': 4,
        # number of nearest edges to observe, defaults to 4
        'num_local_edges': 4,
    }

flow_params = dict(
    exp_tag='seq2seq_light_grid',
    # 如果用自己写的方法，需要改成TrafficLightGridPOEnv
    env_name=SeqTrafficLightEnv,
    network=TrafficLightGridNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=1,
        render=False,
        emission_path='data',
        restart_instance=True,
        print_warnings=False
    ),
    env=EnvParams(
        horizon=1000,
        additional_params=additional_env_params,
    ),
    net=net_params,
    veh=vehicles,
    initial=init
)


if __name__ == "__main__":
    exp = Experiment(flow_params=flow_params)
    exp.run(num_runs=1)
