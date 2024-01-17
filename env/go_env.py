import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box
import os
import math


# computing rewards, taking simulation steps, and resetting the environment.


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
                 healthy_z_range=(0.15, 0.5),
                 reset_noise_scale=1e-2,
                 # cale of random noise added to initial joint positions and velocities during reset.
                 terminate_when_unhealthy=True,
                 exclude_current_positions_from_observation=False,
                 frame_skip=40,  # Number of simulation steps to skip for each environment step.
                 ang_vel_weight=1,
                 **kwargs,
                 ):
        if exclude_current_positions_from_observation:
            self.obs_dim = 17 + 18
        else:
            self.obs_dim = 19 + 18

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float64
        )  # qpos: position of joints, qvel: velocities of joints
        MujocoEnv.__init__(self,
                           model_path=os.path.join(os.path.dirname(__file__), 'go/scene.xml'),
                           frame_skip=frame_skip,
                           observation_space=observation_space,
                           **kwargs
                           )  #
        # action space with dimension of 12, representing actions of joints
        self.action_dim = 12
        self.action_space = Box(
            low=self.lower_limits, high=self.upper_limits, shape=(self.action_dim,), dtype=np.float64
        )

        self._reset_noise_scale = reset_noise_scale
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        self.ang_vel_weight = ang_vel_weight

        # self defined parameters
        self.joint_angles = self.data.qpos[7:]
        self.done = 0
        # rewards coefficients, Penalize the change in the action and encourage smooth actions.
        self.action_rate = -0.01
        # Early termination penalty.
        self.termination = -1.0

        self.stiffnes = 1
        self.damping = 0.3

    # limits for the joint actions.
    @property
    def lower_limits(self):
        # return np.array([-0.863, -0.686, -2.818] * 4) #default
        return np.array([-0.7, -1.0, -2] * 4)

    @property
    def upper_limits(self):
        return np.array([0.52, 2.1, 2.1] * 4)

    ''' Abduction Joint (FR_hip_joint):
        Joint Type: abduction
        Axis: [1, 0, 0]
        Damping: 1
        Range: [-0.863, 0.863]
        Hip Joint (FR_thigh_joint):

        Joint Type: hip
        Range: [-0.686, 4.501]
        Knee Joint (FR_calf_joint):

        Joint Type: knee
        Range: [-2.818, -0.888]
        Force Range: [-35.55, 35.55]
        Abduction Joint (FL_hip_joint):
    '''

    # 7 joints for trunk 12 joints for leg
    # Legs (FR, FL, RR, RL) abduction, hip, knee
    @property
    def init_joints(self):
        return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0, 0.7, -1.4] * 4)

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

    # if the robot's z-coordinate is within the specified healthy range.
    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        return terminated  # returns 1 or 0

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return np.concatenate([qpos, qvel])

    # ------------ reward functions----------------

    def _reward_healthy(self):
        return (self.is_healthy - 1) * 5

    'rewards for moving forward: for moving in x direction around target velocity gets the biggest reward, for moving ' \
    'in y,z direction the reward will be smaller'
    def _reward_lin_vel(self, before_pos, after_pos):
        target_vel = np.array([0.3, 0, 0]) #represents x,y,z of torso
        lin_vel = (after_pos - before_pos) / self.dt
        return np.exp(-0.1 * np.linalg.norm(target_vel - lin_vel))

    'check if the leg joints are out of the defined upper & lower limit'
    def reward_joint_limits(self):
        joint_out_of_limits= np.any(self.joint_angles < self.lower_limits) | np.any(
            self.joint_angles > self.upper_limits)
        if joint_out_of_limits:
            self.done = -1
        return self.done

    '''check the angular velocity. For higher angular velocity the return value (penalty) is higher.
        The angular velocity is represented by:  ang_vel = self.data.qvel[6:].copy() after simulation in step function
        QUESTION: is considering self.data.qvel[6:].copy() for this function correct?'''
    def reward_ang_vel(self, ang_vel):
        ang_vel_cost = np.square(np.linalg.norm(ang_vel))
        return -self.ang_vel_weight * ang_vel_cost

    'I think we should choose one function: this one or the calculate_rotation_reward, but not both on same time?'
    def heading_reward(self, after_pos):
        # heading reward
        cur_heading = after_pos
        cur_heading /= np.linalg.norm(cur_heading)
        error = np.linalg.norm(cur_heading - np.array([1, 0, 0]))
        return np.exp(-error)

    'compare desired orientation of torso [0,0,0] with current orientation' \
    'smaller differencies get higher rewards, the current_orientation is represented by self.base_rotation ' \
    'after simulation in step function '
    'QUESTION: do we need both of the functions calculate_roatation-reward and reward_ang_vel?'
    def _calculate_rotation_reward(self, current_orientation):
        desired_orientation = [1, 0, 0]  # desired orientation for forward movement
        # Calculate angular distance (difference) between desired and current orientations
        angular_distance = np.linalg.norm(np.array(desired_orientation) - np.array(current_orientation))
        # # Define a reward based on the angular distance from the desired orientation
        reward = max(0, 1 - angular_distance)  # Reward decreases as angular distance increases
        return reward


    def _calc_torque_reward(self, prev_torque, after_torque):
        penalty = 0.25 * (sum(np.abs(prev_torque - after_torque)) / len(after_torque))
        return np.exp(-penalty)

    '''this function calculate difference between current action and previous action to have smooth movements
        QUESTION: but is this not counterproductive for learning?'''
    def reward_action_rate(self, prev_action):
        #should we implement some tolerance
        target_action = self.data.qpos[-12:]
        penalty = 5 * sum(np.abs(prev_action - target_action)) / len(target_action)
        return np.exp(-penalty)
        # return self.action_rate * np.sum(np.square(delta_q))

    'calculate distance between right legs joints to avoid small distances (crossing) between legs on same side or' \
    'too large distances '
    def _calc_feet_right_separation_reward(self):
        # feet y-separation cost

        # FR, FL, BR, BL
        '''frfoot_pos = self.data.qpos[7]
        brfoot_pos = self.data.qpos[13]

        # front movement
        foot_dist_site_r = np.abs(frfoot_pos - brfoot_pos)
        error_r = 5 * np.square(foot_dist_site_r - 0.6)

        if foot_dist_site_r < 0.80 and foot_dist_site_r > 0.20:
            error_r = 0'''

        frfoot_pos = self.data.qpos[9]
        brfoot_pos = self.data.qpos[15]

        # front movement
        foot_dist_site_r = np.abs(frfoot_pos - brfoot_pos)
        error_r = 5 * np.square(foot_dist_site_r - 0.6)

        if foot_dist_site_r < 0.80 and foot_dist_site_r > 0.20:
            error_r = 0
        return np.exp(- error_r)

    'calculate distance between left legs joints to avoid small distances (crossing) between legs on same side or' \
    'too large distances '
    def _calc_feet_left_separation_reward(self):
        # feet y-separation cost

        # FR, FL, BR, BL

        '''flfoot_pos = self.data.qpos[10]
        blfoot_pos = self.data.qpos[16]

        # front movement
        foot_dist_site_l = np.abs(flfoot_pos - blfoot_pos)
        error_l = 5 * np.square(foot_dist_site_l - 0.6)

        # print(f'ditance back: {foot_dist_site_l }')

        if foot_dist_site_l < 0.80 and foot_dist_site_l > 0.20:
            error_l = 0'''

        flfoot_pos = self.data.qpos[12]
        blfoot_pos = self.data.qpos[18]

        # front movement
        foot_dist_site_l = np.abs(flfoot_pos - blfoot_pos)
        error_l = 5 * np.square(foot_dist_site_l - 0.6)

        # print(f'ditance back: {foot_dist_site_l }')

        if foot_dist_site_l < 0.80 and foot_dist_site_l > 0.20:
            error_l = 0
        return np.exp(-error_l)

    'calculate distance between front legs joints to avoid small distances (crossing) between legs  or' \
    'too large distances '
    def _calc_feet_front_separation_reward(self):
        # feet y-separation cost

        # FR, FL, BR, BL
        '''frfoot_pos = self.data.qpos[8]
        flfoot_pos = self.data.qpos[11]

        # front movement
        foot_dist_site = np.abs(frfoot_pos - flfoot_pos)
        target_distance_legs = 0.5
        error_r = 5 * np.square(foot_dist_site - target_distance_legs)  #for larger distances the error quadraticly increases

        # gives us some tolerance range, if the distance is in this range the error is set to 0 => no penalty
        if foot_dist_site < 0.8 and foot_dist_site > 0.2:
            error_r = 0
        '''

        frfoot_pos = self.data.qpos[10]
        flfoot_pos = self.data.qpos[13]

        # front movement
        foot_dist_site = np.abs(frfoot_pos - flfoot_pos)
        target_distance_legs = 0.5
        error_r = 5 * np.square(
            foot_dist_site - target_distance_legs)  # for larger distances the error quadraticly increases

        # gives us some tolerance range, if the distance is in this range the error is set to 0 => no penalty
        if foot_dist_site < 0.8 and foot_dist_site > 0.2:
            error_r = 0
        return np.exp(- error_r)

    'calculate distance between back legs joints to avoid small distances (crossing) between legs  or' \
    'too large distances '
    def _calc_feet_back_separation_reward(self):
        # feet y-separation cost

        # FR, FL, BR, BL
        ''' brfoot_pos = self.data.qpos[14]
        blfoot_pos = self.data.qpos[17]
        # front movement
        foot_dist_site = np.abs(brfoot_pos - blfoot_pos)
        # print(f'ditance back: {foot_dist_site }')
        error_r = 5 * np.square(foot_dist_site - 0.5)

        if foot_dist_site < 0.8 and foot_dist_site > 0.2:
            error_r = 0'''

        brfoot_pos = self.data.qpos[16]
        blfoot_pos = self.data.qpos[18]
        # front movement
        foot_dist_site = np.abs(brfoot_pos - blfoot_pos)
        # print(f'ditance back: {foot_dist_site }')
        error_r = 5 * np.square(foot_dist_site - 0.5)

        if foot_dist_site < 0.8 and foot_dist_site > 0.2:
            error_r = 0
        return np.exp(- error_r)

    def step(self, delta_q):  # SIMULATION
        action = delta_q + self.data.qpos[-12:]
        # The resulting action is then clipped to ensure it falls within specified lower and upper limits.
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        #influence the flexibility of joints movement
        prev_torque = self.stiffnes * delta_q - self.damping * self.data.qvel[6:]

        #save before position and before(previous action)
        before_pos = self.data.qpos[:3].copy()
        prev_action = action
        self.do_simulation(action, self.frame_skip)

        #save 'after' data
        after_pos = self.data.qpos[:3].copy()
        after_torque = self.stiffnes * delta_q - self.damping * self.data.qvel[6:]
        ang_vel = self.data.qvel[6:].copy() # to calculate angular_velocity_reward
        current_orientation = self.base_rotation # to check if the robot is oriented in x direction


        #calculate reward functions and sum up to reward_total
        lin_v_track_reward = self._reward_lin_vel(before_pos, after_pos)
        healthy_reward = self._reward_healthy()
        # reward_standing = self.reward_standing();
        reward_ang_vel = self.reward_ang_vel(ang_vel)
        heading_reward = self.heading_reward(after_pos)
        rotation_reward = self._calculate_rotation_reward(current_orientation)
        reward_joint_limits = self.reward_joint_limits()
        reward_action_rate = self.reward_action_rate(prev_action)
        torque_reward = self._calc_torque_reward(prev_torque, after_torque)
        reward_foot_separation_left = self._calc_feet_left_separation_reward()
        reward_foot_separation_right = self._calc_feet_right_separation_reward()
        feet_back_separation_reward = self._calc_feet_back_separation_reward()
        feet_front_separation_reward = self._calc_feet_front_separation_reward()

        #total_rewards = 5.0 * lin_v_track_reward + 1.5 * healthy_reward + reward_ang_vel + 2 *heading_reward + 2.4*rotation_reward + reward_joint_limits + 1.5*reward_action_rate + 1.8*torque_reward
        #+ 1.5*reward_foot_separation_left +  1.5*reward_foot_separation_right + 1.5*feet_back_separation_reward + 1.5*feet_front_separation_reward

        total_rewards = 7.0 * lin_v_track_reward + 1.5 * healthy_reward + reward_ang_vel + + 1.5 * rotation_reward + reward_joint_limits  + 1 * torque_reward
        + 3*reward_foot_separation_left +  3*reward_foot_separation_right + 3*feet_back_separation_reward + 3*feet_front_separation_reward

        total_rewards = np.array([total_rewards])
        #    mean_rewards = np.mean(total_rewards)
        #   std_rewards = np.std(total_rewards)
        #  epsilon = 1e-9
        # total_rewards = (total_rewards - mean_rewards) / (std_rewards + epsilon)  # Normalize total reward

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

    # randomly initializes joint positions and velocities with added noise.
    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_joints + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
