from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
from flow.core import rewards
from gym.spaces import Box, Discrete
import numpy as np

# center_[ID_IDX]
ID_IDX = 1


class SeqTrafficLightEnv(TrafficLightGridPOEnv):

    def __init__(self, env_params, sim_params, network, simulator):
        super().__init__(env_params, sim_params, network, simulator=simulator)
        # number of nearest lights to observe, defaults to 4
        self.num_local_lights = env_params.additional_params.get(
            "num_local_lights", 4)

        # number of nearest edges to observe, defaults to 4
        self.num_local_edges = env_params.additional_params.get(
            "num_local_edges", 4)
    
    
    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, local edge information, and traffic
        light state.
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(
                self.num_traffic_lights,
                3 * 4 * self.num_observed +
                2 * self.num_local_edges +
                2 * (1 + self.num_local_lights),
            ),
            dtype=np.float32)
        return tl_box

    # get state of the environment
    def get_state(self):
        """Observations for each traffic light agent.

        :return: dictionary which contains agent-wise observations as follows:
        - For the self.num_observed number of vehicles closest and incoming
        towards traffic light agent, gives the vehicle velocity, distance to
        intersection, edge number.
        - For edges in the network, gives the density and average velocity.
        - For the self.num_local_lights number of nearest lights (itself
        included), gives the traffic light information, including the last
        change time, light direction (i.e. phase), and a currently_yellow flag.
        """
        # Normalization factors
        max_speed = max(
            self.k.network.speed_limit(edge)
            for edge in self.k.network.get_edge_list())
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"], grid_array["long_length"],
                       grid_array["inner_length"])

        # TODO(cathywu) refactor TrafficLightGridPOEnv with convenience
        # methods for observations, but remember to flatten for single-agent

        # Observed vehicle information
        speeds = []
        dist_to_intersec = []
        edge_number = []
        all_observed_ids = []
        for _, edges in self.network.node_mapping:
            local_speeds = []
            local_dists_to_intersec = []
            local_edge_numbers = []
            for edge in edges:
                observed_ids = \
                    self.get_closest_to_intersection(edge, self.num_observed)
                all_observed_ids.append(observed_ids)

                # check which edges we have so we can always pad in the right
                # positions
                local_speeds.extend(
                    [self.k.vehicle.get_speed(veh_id) / max_speed for veh_id in
                     observed_ids])
                local_dists_to_intersec.extend([(self.k.network.edge_length(
                    self.k.vehicle.get_edge(
                        veh_id)) - self.k.vehicle.get_position(
                    veh_id)) / max_dist for veh_id in observed_ids])
                local_edge_numbers.extend([self._convert_edge(
                    self.k.vehicle.get_edge(veh_id)) / (
                    self.k.network.network.num_edges - 1) for veh_id in
                    observed_ids])

                if len(observed_ids) < self.num_observed:
                    diff = self.num_observed - len(observed_ids)
                    local_speeds.extend([1] * diff)
                    local_dists_to_intersec.extend([1] * diff)
                    local_edge_numbers.extend([0] * diff)

            speeds.append(local_speeds)
            dist_to_intersec.append(local_dists_to_intersec)
            edge_number.append(local_edge_numbers)

        # Edge information
        density = []
        velocity_avg = []
        for edge in self.k.network.get_edge_list():
            ids = self.k.vehicle.get_ids_by_edge(edge)
            if len(ids) > 0:
                # TODO(cathywu) Why is there a 5 here?
                density += [5 * len(ids) / self.k.network.edge_length(edge)]
                velocity_avg += [np.mean(
                    [self.k.vehicle.get_speed(veh_id) for veh_id in
                     ids]) / max_speed]
            else:
                density += [0]
                velocity_avg += [0]
        density = np.array(density)
        velocity_avg = np.array(velocity_avg)
        self.observed_ids = all_observed_ids

        # Traffic light information
        direction = self.direction.flatten()
        currently_yellow = self.currently_yellow.flatten()
        # This is a catch-all for when the relative_node method returns a -1
        # (when there is no node in the direction sought). We add a last
        # item to the lists here, which will serve as a default value.
        # TODO(cathywu) are these values reasonable?
        direction = np.append(direction, [0])
        currently_yellow = np.append(currently_yellow, [1])

        obs = []
        # obs -> [num_light, observation]
        node_to_edges = self.network.node_mapping
        tl_ids = ['center{}'.format(i) for i in range(self.num_traffic_lights)]
        for rl_id in tl_ids:
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_edges = node_to_edges[rl_id_num][1]
            local_edge_numbers = [self.k.network.get_edge_list().index(e)
                                  for e in local_edges]
            local_id_nums = [rl_id_num, self._get_relative_node(rl_id, "top"),
                             self._get_relative_node(rl_id, "bottom"),
                             self._get_relative_node(rl_id, "left"),
                             self._get_relative_node(rl_id, "right")]

            observation = np.array(np.concatenate(
                [speeds[rl_id_num], dist_to_intersec[rl_id_num],
                 edge_number[rl_id_num], density[local_edge_numbers],
                 velocity_avg[local_edge_numbers],
                 direction[local_id_nums], currently_yellow[local_id_nums]
                 ]))
            obs.append(observation)

        return obs

    # get the adj_matrix
    def get_adj_matrix(self):
        adj_matrix = np.identity(self.num_traffic_lights, dtype=np.float)
        directions = ['top', 'bottom', 'left', 'right']
        tl_ids = ['center{}'.format(i) for i in range(self.num_traffic_lights)]
        # set neighborhood as '1'
        for rl_id in tl_ids:
            rl_id_num = int(rl_id.split("center")[ID_IDX])
            local_id_nums = []
            # look up its neighborhood around the light
            for direction in directions:
                node = self._get_relative_node(rl_id, direction)
                # if exist, then append
                if node != -1:
                    local_id_nums.append(node)
            # set adj_matrix as 1.0
            print('{} : {}'.format(rl_id, local_id_nums))
            adj_matrix[rl_id_num][local_id_nums] = 1.0
        return adj_matrix

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        # Male Implemented
        # max or mean
        def max_queue_length(env):
            # now add in the density and average velocity on the edges
            density = []
            for edge in env.k.network.get_edge_list():
                ids = env.k.vehicle.get_ids_by_edge(edge)
                if len(ids) > 0:
                    vehicle_length = 5
                    # density += [vehicle_length * len(ids)]
                    density += [vehicle_length * len(ids) / env.k.network.edge_length(edge)]
                else:
                    density += [0]
            return np.max(density)


        def queue_length(env):
            # now add in the density and average velocity on the edges
            density = []
            for edge in env.k.network.get_edge_list():
                ids = env.k.vehicle.get_ids_by_edge(edge)
                if len(ids) > 0:
                    vehicle_length = 5
                    # density += [vehicle_length * len(ids)]
                    density += [vehicle_length * len(ids) / env.k.network.edge_length(edge)]
                else:
                    density += [0]
            return np.mean(density)

        if self.env_params.evaluate:
            return - rewards.min_delay_unscaled(self)
        else:
            # penalize_near_standstill ??
            max_speed = max([self.k.vehicle.get_speed(id) for id in self.k.vehicle.get_ids()])
            r = rewards.average_velocity(self) - rewards.min_delay_unscaled(self) - max_queue_length(self) \
                + rewards.penalize_near_standstill(self, gain=0.15)
            # return (- rewards.min_delay_unscaled(self) +
            #         rewards.penalize_standstill(self, gain=0.2))
            return r
