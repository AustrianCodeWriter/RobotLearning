import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import os


class GOEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self,
                 healthy_z_range=(0.15, 0.45),  #Adapt it for rewards change
                 reset_noise_scale=1e-2,
                # reset_noise_scale=0.0001,
                 terminate_when_unhealthy=True,
                 exclude_current_positions_from_observation=False,
                 frame_skip=40, #do not change
                 **kwargs,
                 ):
        if exclude_current_positions_from_observation:
            self.obs_dim = 17 + 18
        else:
            self.obs_dim = 19 + 18

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), 'go/scene.xml'),
                           frame_skip=frame_skip,
                           observation_space=observation_space,
                           **kwargs
                           )
        self.action_dim = 12
        self.action_space = Box(
            low=self.lower_limits, high=self.upper_limits, shape=(self.action_dim,), dtype=np.float64
        )

        self._reset_noise_scale = reset_noise_scale
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    @property
    def lower_limits(self):
        return np.array([-0.5, -0.6, -0.8] * 4)
      #  return np.array([-0.863, -0.686, -2.818]*4)

    @property
    def upper_limits(self):
       return np.array([0.5, 0.6, 0.8] * 4)
      #  return np.array([0.863, 4.501, -0.888]*4)

    @property
    def init_joints(self): #do not change #Change of robot body position
        return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0.0, 0.7, -1.4]*4)

    @property
    def base_rotation(self):
        """
        compute root (base) rotation of the robot. The rotation can be used for rewards
        :return: rotation of root in xyz direction
        """
        q = self.data.qpos[3:7].copy()
        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        x = np.arctan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = np.arcsin(np.clip(2 * (q1 * q3 + q0 * q2), -1, 1))
        z = np.arctan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))

        return [x, y, z]

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        return terminated

    def _get_obs(self):   ## GET STATE
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return np.concatenate([qpos, qvel])

    # ------------ reward functions----------------
    def _reward_healthy(self):
        return (self.is_healthy - 1) * 5

    def _reward_lin_vel(self, before_pos, after_pos): # Info: modified func.
        #Instead distance reward for velocity
        target_vel = np.array([0.5, 0, 0])
        lin_vel = (after_pos - before_pos) / self.dt
        # Check if the robot moved forward in the x-direction comparing the x-coordinates
        moved_forward_x= after_pos[0] > before_pos[0]
        moved_forward_y = after_pos[1] - before_pos[1] #0 - False
        moved_forward_z = after_pos[2] - before_pos[2] #

       # if moved_forward_x:
            # Reward for moving forward
        return np.exp(-10*np.linalg.norm(target_vel - lin_vel))
       # else:
            # Penalize if not moving forward
        #    return -1.0

        # 1. Velocity Tracking Reward
       # velocity_difference = np.abs(target_vel - lin_vel)
        #threshold = np.array([0.5, 0, 0])  # Adjust this threshold as needed
        #velocity_reward = 2.0 if np.linalg.norm(velocity_difference) > threshold else -1.0
        #print(velocity_reward)
        #return velocity_reward
        # print(f"lin reward: {np.exp(-0.1*np.linalg.norm(target_vel - lin_vel))}  ")
       # return np.exp(-0.1*np.linalg.norm(target_vel - lin_vel))  0.- 2
    # #Chaned from 10 to 0.1 - Increased reward -> better loss (lower)
        #Adjusting the factor -10 can change the rate at which the penalty decreases as the distance increases

    # Not in use - Calculate linear velocity based on the change in position over time
    def help_reward_lin_vel(self, previous_position, current_position):
        displacement = current_position - previous_position
        velocity = displacement / self.dt
        linear_velocity = np.linalg.norm(velocity)  # Euclidean distance as linear velocity
        return linear_velocity

    def _calculate_rotation_reward(self):
      #  x_rotation = self.base_rotation[0]
        # Set a threshold for considering forward movement based on rotation
       # forward_threshold = 0.1  # Adjust this threshold based on robot's orientation

        # Check if the robot is facing forward based on x-axis rotation
        #if abs(x_rotation) < forward_threshold:
         #   return 2.0
        #else:
            # Penalize rotation
         #   return -1.0
        desired_orientation = [0, 0, 0]  # desired orientation for forward movement - no rotation in any axis
        current_orientation = self.base_rotation  # Calculate current orientation using base_rotation
        # Calculate angular distance (difference) between desired and current orientations
        angular_distance = np.linalg.norm(np.array(desired_orientation) - np.array(current_orientation))
       # # Define a reward based on the angular distance from the desired orientation
        reward = max(0, 1 - angular_distance)  # Reward decreases as angular distance increases
        return reward

# Actor NN - get action
    def step(self, delta_q):   ## For Q calculation in rl agent?  #TODO correct reward function
        """ Apply actions, simulate, call self.post_physics_step()
       delta_q: Predicted action changes.
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
            1. Action: Combines delta_q with the current joint positions to get the action.
               • Clips the action values to be within the defined limits
            2. Uses the do_simulation method to perform a simulation step with the computed action.
            3. Reward Computation:• Computes rewards based on linear velocity tracking and healthiness.
                                 • Combines the rewards to get the total reward.
            4. Checks if the environment should be terminated based on healthiness.
        """
        #Action update
        # delta_q represents the predicted changes in action that the agent wants to apply.
        # The predicted action delta_q is added to the current joint positions (self.data.qpos[-12:]) to determine the new action.
        #delta_q = np.array([0, -1.0, 0] * 4)
        action = delta_q + self.data.qpos[-12:]
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)


        before_pos = self.data.qpos[:3].copy() #do not change

        self.do_simulation(action, self.frame_skip)
        after_pos = self.data.qpos[:3].copy()  #do not change

        lin_v_track_reward = self._reward_lin_vel(before_pos, after_pos)
        # Design reward based on linear velocity
     #   reward = lin_v_track_reward * 8  # Adjust scaling factor as needed

        # Update previous position for next iteration
      #  previous_position = after_pos



        healthy_reward = self._reward_healthy()
     #   print(healthy_reward)
        rotation_reward = self._calculate_rotation_reward()
     #   print(before_pos)
      #  print(lin_v_track_reward)
       # print(after_pos)
        # Include the linear velocity-based reward, healthy reward and rotation reward in the total reward
        total_rewards =  5.0*lin_v_track_reward + 2.0*healthy_reward + 2.0 * rotation_reward
        # print(f"total_rewards: {total_rewards}  ")
        total_rewards = np.array([total_rewards])
        mean_rewards = np.mean(total_rewards)
        std_rewards = np.std(total_rewards)
        epsilon = 1e-9
        total_rewards = (total_rewards - mean_rewards) / (std_rewards + epsilon) #Normalize total reward

        terminate = self.terminated
        observation = self._get_obs()
        info = {
            'total_reward': total_rewards,
            'lin_v_track_reward': lin_v_track_reward,
            "healthy_reward": healthy_reward,
            "traverse": self.data.qpos[0],
        }

        if self.render_mode == "human":
            self.render()
        return observation, total_rewards, terminate, info

    def reset_model(self):
        #adding noise to the selected actions promotes exploration while still benefiting from the policy's learned behavior.
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_joints + self.np_random.uniform(
           low=noise_low, high=noise_high, size=self.model.nq )
       # print(qpos)


        qvel = self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

