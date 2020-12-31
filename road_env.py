from flow.core.params import InFlows, SumoParams
from flow.core.params import TrafficLightParams, NetParams, InitialConfig
from flow.core.params import EnvParams, SumoCarFollowingParams, VehicleParams
from flow.controllers import GridRouter
from flow.envs.ring.accel import AccelEnv, ADDITIONAL_ENV_PARAMS
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
    n_rows=2,
    # 路网列数
    n_columns=2,
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
            probability=0.25,
            depart_lane='free',
            depart_speed=20,
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
    num_vehicles=ROAD_PARAMS['total_cars']
)
# 交通灯
traffic_lights = TrafficLightParams()
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
    'vertical_lanes': 1
}

init, net_params = get_flow_params(num_row=ROAD_PARAMS['n_rows'],
                                   num_col=ROAD_PARAMS['n_columns'],
                                   additional_net_params=additional_net_params
                                   )


flow_params = dict(
    exp_tag='grid_inter',
    env_name=AccelEnv,
    network=TrafficLightGridNetwork,
    simulator='traci',
    sim=SumoParams(
        sim_step=0.1,
        render=True,
        emission_path='data'
    ),
    env=EnvParams(
        horizon=1500,
        additional_params=ADDITIONAL_ENV_PARAMS,
    ),
    net=net_params,
    veh=vehicles,
    initial=init,
    tls=traffic_lights,
)

if __name__ == "__main__":
    exp = Experiment(flow_params=flow_params)
    exp.run(num_runs=1)
