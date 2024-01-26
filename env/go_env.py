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
                 healthy_y_range=(-0.25, 0.25),  # define healthy y range
                 healthy_x_range=(-0.2,),
                 reset_noise_scale=1e-2,
                 # cale of random noise added to initial joint positions and velocities during reset.
                 terminate_when_unhealthy=True,
                 exclude_current_positions_from_observation=False,
                 frame_skip=40,  # Number of simulation steps to skip for each environment step.
                 ang_vel_weight=1,
                 ctrl_cost_weight=0.1,
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
        self._healthy_y_range = healthy_y_range
        self._healthy_x_range = healthy_x_range
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

        self.stiffnes = 0.6
        self.damping = 0.1

        self.foot_radius = 0.023
        #self.feet_air_time = np.zeros(4)
        self.qpos_history = np.zeros(15 * 12)
        #self.qpos_history_counter = 0
        self._ctrl_cost_weight = ctrl_cost_weight
        self.init_foot_quat = np.array([-0.24135508, -0.24244352, -0.66593612, 0.66294642])
        #self.time_counter_FR_foot_in_ar = 0.0
        #self.time_counter_FL_foot_in_ar = 0.0
        #self.time_counter_BR_foot_in_ar = 0.0
        #self.time_counter_BL_foot_in_ar = 0.0


    # limits for the joint actions.
    @property
    def lower_limits(self):
        return np.array([-0.5, -0.4, -1.4] * 4)

    @property
    def upper_limits(self):
        return np.array([0.5, 3, -1] * 4)

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
        is_healthy = (min_z < self.data.qpos[2] < max_z)

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
        return (self.is_healthy - 1) * 15

    def angular_velocity_penalty(self):
        angular_velocity = self.data.qvel[6:].copy()
        return -0.1 * self.dt * np.linalg.norm(angular_velocity[:2]) ** 2


    def _calc_feet_right_separation_reward(self, foot_pos_after):
        # feet x-separation cost

        # FR, FL, BR, BL
        foot_pos_x = foot_pos_after[:, 0]
        frfoot_pos = foot_pos_x[0]
        brfoot_pos = foot_pos_x[2]

        # front movement
        foot_dist_site_r = np.abs(frfoot_pos - brfoot_pos)
        error_r = 5 * np.square(foot_dist_site_r - 0.65)

        if foot_dist_site_r < 0.80 and foot_dist_site_r > 0.4:
            error_r = 0
        return np.exp(- error_r)

    'calculate distance between left legs joints to avoid small distances (crossing) between legs on same side or' \
    'too large distances '

    def _calc_feet_left_separation_reward(self, foot_pos_after):
        # feet x-separation cost

        # FR, FL, BR, BL
        foot_pos_x = foot_pos_after[:, 0]
        flfoot_pos = foot_pos_x[1]
        blfoot_pos = foot_pos_x[3]

        # front movement
        foot_dist_site_l = np.abs(flfoot_pos - blfoot_pos)
        error_l = 5 * np.square(foot_dist_site_l - 0.65)

        # print(f'ditance back: {foot_dist_site_l }')

        if foot_dist_site_l < 0.80 and foot_dist_site_l > 0.4:
            error_l = 0

        return np.exp(-error_l)


    def _calc_torque_reward(self, prev_torque, after_torque):
        penalty = 0.25 * (sum(np.abs(prev_torque - after_torque)) / len(after_torque))
        return np.exp(-penalty)

    '''check the angular velocity. For higher angular velocity the return value (penalty) is higher.
            The angular velocity is represented by:  ang_vel = self.data.qvel[6:].copy() after simulation in step function
        '''



    '---------------The following reward functions are not used in final result, but were used for testing to obtian ' \
    '---------------optimal result ---------------------------------------------------------------------------------'

    def reward_ang_vel(self, ang_vel):
        ang_vel_cost = np.square(np.linalg.norm(ang_vel))
        return -self.ang_vel_weight * ang_vel_cost

    'rewards for moving forward: for moving in x direction around target velocity gets the biggest reward, for moving ' \
    'in y,z direction the reward will be smaller'
    def _reward_lin_vel(self, before_pos, after_pos):
        target_vel = np.array([0.4, 0, 0])  # represents x,y,z of torso
        lin_vel = (after_pos - before_pos) / self.dt
        return np.exp(-10.0 * np.linalg.norm(target_vel - lin_vel))

    'check if the leg joints are out of the defined upper & lower limit'
    def reward_joint_limits(self):
        joint_out_of_limits = np.any(self.joint_angles < self.lower_limits) | np.any(
            self.joint_angles > self.upper_limits)
        if joint_out_of_limits:
            self.done = -1
        return self.done

    def heading_reward(self):
        # heading reward
        cur_heading = self.data.qvel[:3]
        cur_heading /= np.linalg.norm(cur_heading)
        error = np.linalg.norm(cur_heading - np.array([1, 0, 0]))
        return np.exp(-error)



    'compare desired orientation of torso [0,0,0] with current orientation' \
    'smaller differencies get higher rewards, the current_orientation is represented by self.base_rotation ' \
    'after simulation in step function '
    def _calculate_rotation_reward(self, current_orientation):
        desired_orientation = [0, 0, 0]  # desired orientation for forward movement
        # Calculate angular distance (difference) between desired and current orientations
        angular_distance = np.linalg.norm(np.array(desired_orientation) - np.array(current_orientation))
        # # Define a reward based on the angular distance from the desired orientation
        reward = max(0, 1 - angular_distance)  # Reward decreases as angular distance increases
        return reward

    'calculate distance between front legs joints to avoid small distances (crossing) between legs  or' \
    'too large distances '

    def _calc_feet_front_separation_reward(self, foot_pos_after):
        # feet y-separation cost

        # FR, FL, BR, BL
        foot_pos_y = foot_pos_after[:, 1]
        frfoot_pos = foot_pos_y[0]
        flfoot_pos = foot_pos_y[1]

        # front movement
        foot_dist_site = np.abs(frfoot_pos - flfoot_pos)
        target_distance_legs = 0.08
        error_r = 5 * np.square(
            foot_dist_site - target_distance_legs)  # for larger distances the error quadratically increases

        # gives us some tolerance range, if the distance is in this range the error is set to 0 => no penalty
        if foot_dist_site < 0.1 and foot_dist_site > 0.05:
            error_r = 0
        return np.exp(- error_r)

    'calculate distance between back legs joints to avoid small distances (crossing) between legs  or' \
    'too large distances'
    def _calc_feet_back_separation_reward(self, foot_pos_after):
        # feet y-separation cost
        # FR, FL, BR, BL
        foot_pos_y = foot_pos_after[:, 1]
        brfoot_pos = foot_pos_y[2]
        blfoot_pos = foot_pos_y[3]
        # front movement
        foot_dist_site = np.abs(brfoot_pos - blfoot_pos)
        # print(f'ditance back: {foot_dist_site }')
        error_r = 5 * np.square(foot_dist_site - 0.08)

        if foot_dist_site < 0.1 and foot_dist_site > 0.05:
            error_r = 0
        return np.exp(- error_r)

    def penalty_foot_to_high(self, foot_pos_z):
        pos_z = foot_pos_z - self.foot_radius

        if np.any( pos_z > 0.1):
            return -5
        else:
            return 1

    def _calc_height_reward(self, foot_pos_after):
        contact_point = 0
        goal_height_ref = 0.45
        goal_speed_ref = 0.2
        current_height = self.data.qpos[2]
        relative_height = current_height - contact_point
        error = np.abs(relative_height - goal_height_ref)
        deadzone_size = 0.01 + 0.05 * goal_speed_ref
        if error < deadzone_size:
            error = 0
        #  print(np.exp(-40 * np.square(error)))
        return np.exp(-40 * np.square(error))


    """
    encourage the learning algorithm to find control policies that achieve the task with smoother and less aggressive control actions. 
    *ctrl_cost*: A negative reward for penalising the humanoid if it has too
    large of a control force. If there are *nu* actuators/controls, then the control has
    shape  `nu x 1`. It is measured as *`ctrl_cost_weight` * sum(control<sup>2</sup>)*.
    *contact_cost*: A negative reward for penalising the robot if the external contact force is too large. It is calculated by clipping
    *contact_cost_weight` * sum(external contact force<sup>2</sup>)* to the interval specified by `contact_cost_range`."""

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
        return control_cost

    # Not used in final result
    def quat_distance(self, q1, q2):
        # Ensure quaternions are unit quaternions
        q1 /= np.linalg.norm(q1)
        q2 /= np.linalg.norm(q2)

        # Calculate the dot product
        dot_product = np.dot(q1, q2)

        # Ensure the dot product is within the valid range [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate the angular distance
        angle = 2.0 * np.arccos(dot_product)

        return angle



    def get_joint_qvel(self):
        return self.data.qvel[6:].copy()

    def get_joint_torques(self):
        return self.data.qvel[6:].copy()

    def _calc_body_orient_reward(self, quat_ref=[1, 0, 0, 0]):
        # body orientation reward
        body_quat = self.data.body('trunk').xquat
        target_quat = np.array(quat_ref)
        error = 10 * (1 - np.inner(target_quat, body_quat)**2)
        return np.exp(-error)

    # Not used in final result
    def count_airtime(self, foot_position):
        if foot_position[0]:
            self.time_counter_FR_foot_in_ar += 1
        else:
            self.time_counter_FR_foot_in_ar = 0

        if foot_position[1]:
            self.time_counter_FL_foot_in_ar += 1
        else:
            self.time_counter_FL_foot_in_ar = 0

        if foot_position[2]:
            self.time_counter_BR_foot_in_ar += 1
        else:
            self.time_counter_BR_foot_in_ar = 0

        if foot_position[3]:
            self.time_counter_BL_foot_in_ar += 1
        else:
            self.time_counter_BL_foot_in_ar = 0

    # Not used in final result
    def penalty_airtime(self):
        if self.time_counter_FR_foot_in_ar > 100 or self.time_counter_FL_foot_in_ar > 100 or self.time_counter_BR_foot_in_ar > 100 or self.time_counter_BL_foot_in_ar > 100:
            return -5
        else:
            return 1

    # Not used in final result
    def penalty_pitch(self):
        pitch = abs(math.degrees(self.base_rotation[2]))
        if pitch > 1:
            return -5
        else:
            return 1

    # Not used in final result
    def _calculate_stability_penalty(self):
        # Get the orientation of the robot's base or trunk
        base_orientation = self.data.body('trunk').xquat

        # Define the desired orientation (e.g., flat on the ground)
        desired_orientation = np.array([1, 0, 0, 0])  # Assuming no rotation

        # Calculate the angular deviation between the current and desired orientations
        angular_deviation = np.arccos(np.clip(2 * np.dot(desired_orientation, base_orientation) ** 2 - 1, -1, 1))

        # Penalize large deviations from the desired orientation
        stability_penalty = 1.25 * angular_deviation

        return stability_penalty

    '''
    #Not used in final result 
    def penalty_contact_foot(self,foot_in_contact_before):
        condition_a = foot_in_contact_before[0] and foot_in_contact_before[3] and  foot_in_contact_before[1] and  \
            foot_in_contact_before[2]

        condition_b = not foot_in_contact_before[1] and not foot_in_contact_before[2] and not foot_in_contact_before[0] and not \
            foot_in_contact_before[3]

        if condition_a or condition_b:
            return -1.5
        else:
            return 0
    '''

    '''
    #Not used in final result 
    def back_feet_x_position_penalty(self, foot_pos_x):
            # feet to far away from center of torso
            if np.abs(foot_pos_x[2] - self.data.qpos[0]) > 0.3 or np.abs(foot_pos_x[3] - self.data.qpos[0]) > 0.3:
                return -5
            # feet to close to center of torso
            if np.abs(foot_pos_x[2] - self.data.qpos[0]) < 0.1 or np.abs(foot_pos_x[3] - self.data.qpos[0]) <0.1:
                 return -5

            return 1
    '''

    '''
    #Not used in final result 
    def front_feet_x_position_penalty(self, foot_pos_x):
            # feet to far away from center of torso
            if np.abs(foot_pos_x[0] - self.data.qpos[0]) > 0.3 or np.abs(foot_pos_x[1] - self.data.qpos[0]) > 0.3:
                return -5


            # feet to close to center of torso

            if np.abs(foot_pos_x[0] - self.data.qpos[0]) < 0.1 or np.abs(foot_pos_x[1] - self.data.qpos[0]) < 0.1:
                return -5

            return 1
    '''


    '''
    def pair_leg_movement_reward(self, foot_in_contact, qvel):
        pair_right_in_air = (not foot_in_contact[0] and not foot_in_contact[2]) and ( foot_in_contact[1] and  foot_in_contact[3])
        pair_right_moving_forward = (qvel[0] > 0 and qvel[1] > 0 and qvel[2] > 0) and (qvel[6] > 0 and qvel[7] > 0 and qvel[8] > 0)
        pair_right_same_angles = np.isclose(qvel[0], qvel[6], rtol=0.1, atol=0.1) and  np.isclose(qvel[1], qvel[7], rtol=0.1, atol=0.1) and np.isclose(qvel[2], qvel[7], rtol=0.1, atol=0.1)
        pair_left_not_moving_forward = (qvel[3] == 0 and qvel[4]== 0 and qvel[5] == 0) and (qvel[9] == 0 and qvel[10] == 0 and qvel[11] == 0)
        condition_right = pair_right_in_air and   pair_right_moving_forward and pair_right_same_angles and pair_left_not_moving_forward

        pair_left_in_air = (foot_in_contact[0] and  foot_in_contact[2]) and (
                   not foot_in_contact[1] and not foot_in_contact[3])
        pair_left_moving_forward = (qvel[3] > 0 and qvel[4] > 0 and qvel[5] > 0) and (
                    qvel[9] > 0 and qvel[10] > 0 and qvel[11] > 0)
        pair_left_same_angles = np.isclose(qvel[3], qvel[9], rtol=0.1, atol=0.1) and np.isclose(qvel[4], qvel[10],
                                                                                                 rtol=0.1,
                                                                                                 atol=0.1) and np.isclose(
            qvel[5], qvel[11], rtol=0.1, atol=0.1)
        pair_right_not_moving_forward = (qvel[0] == 0 and qvel[1] == 0 and qvel[2] == 0) and (
                    qvel[6] == 0 and qvel[7] == 0 and qvel[8] == 0)
        condition_left = pair_left_in_air and pair_left_moving_forward and pair_left_same_angles and pair_left_not_moving_forward




        if condition_left or condition_right:
            return 2
        else:
            return -1
    '''

    '''--------------------------------------------------------------------------------------------------------'''
    '''--------------------------------------------------------------------------------------------------------'''


    def step(self, delta_q):  # SIMULATION
        action = delta_q + self.data.qpos[-12:]
        # The resulting action is then clipped to ensure it falls within specified lower and upper limits.
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        # influence the flexibility of joints movement
        prev_torque = self.stiffnes * delta_q - self.damping * self.data.qvel[6:]
        before_pos = self.data.qpos[:3].copy()
        ctrl_cost = self.control_cost(action)

        self.qpos_history = self.data.qpos[-12:]
        self.qpos_history_counter += 1
        self.do_simulation(action, self.frame_skip)

        # save 'after' data
        after_pos = self.data.qpos[:3].copy()
        after_torque = self.stiffnes * delta_q - self.damping * self.data.qvel[6:]
        current_orientation = self.base_rotation  # to check if the robot is oriented in x direction
        FR_foot_after = self.data.site("FR").xpos
        FL_foot_after = self.data.site("FL").xpos
        BR_foot_after = self.data.site("RR").xpos
        BL_foot_after = self.data.site("RL").xpos
        foot_pos_after = np.vstack([FR_foot_after, FL_foot_after, BR_foot_after, BL_foot_after])

        '''target_angular_velocity = 45.0 / 180.0 * np.pi * (target_delta_yaw / np.pi)
        #foot_pos_z_after = foot_pos_after[:, 2]
        #foot_contact_z_after = foot_pos_z_after - self.foot_radius
        #foot_in_contact_after = foot_contact_z_after < 0.001  # a mm or less off the floor
        #foot_pos_x = foot_pos_after[:, 0]
        #z_penalty = -3.*z_velocity
        #y_penalty = -y_velocity
        #left_foot_contact_force = np.sum(
            #np.square(self.data.cfrc_ext[12] + self.data.cfrc_ext[13])
        #)
        # - np.abs(self.data.qpos[1]) * 10.0 - self.quat_distance(self.init_foot_quat,frfoot_pos) * 5- self.quat_distance(self.init_foot_quat, brfoot_pos) * 5)
        '''


        'calculate reward functions and sum up to reward_total'
        xyz_velocity = (after_pos - before_pos) / self.dt
        x_velocity, y_velocity, z_velocity = xyz_velocity
        forward_reward = 1.25 * x_velocity
        lin_v_track_reward = forward_reward

        angular_velocity_penalty = self.angular_velocity_penalty();
        healthy_reward = self._reward_healthy()
        rotation_reward = self._calculate_rotation_reward(current_orientation)
        torque_reward = self._calc_torque_reward(prev_torque, after_torque)
        reward_foot_separation_left = self._calc_feet_left_separation_reward(foot_pos_after)
        reward_foot_separation_right = self._calc_feet_right_separation_reward(foot_pos_after)

        '''
                lin_v_track_reward = self._reward_lin_vel(before_pos, after_pos)
                heading_reward = self.heading_reward(after_pos)
                reward_ang_vel = self.reward_ang_vel(ang_vel)
                reward_joint_limits = self.reward_joint_limits()
                feet_back_separation_reward = self._calc_feet_back_separation_reward(foot_pos_after)
                feet_front_separation_reward = self._calc_feet_front_separation_reward(foot_pos_after)
                pair_leg_movement_reward = self.pair_leg_movement_reward(foot_in_contact_before)
                height_reward = self._calc_height_reward(foot_pos_after)
                _calc_body_orient_reward =  self._calc_body_orient_reward()
                penalty_contact_foot = self.penalty_contact_foot(foot_in_contact_after)
                heading_reward = self.heading_reward()
                penalty_foot_to_high = self.penalty_foot_to_high(foot_pos_z_after)
                back_feet_x_position_penalty = self.back_feet_x_position_penalty(foot_pos_x)
                front_feet_x_position_penalty = self.front_feet_x_position_penalty(foot_pos_x)
                self.count_airtime(foot_contact_z_after)
                penalty_airtime = self.penalty_airtime()
                penalty_pitch = self.penalty_pitch()
                pair_leg_movement_reward = self.pair_leg_movement_reward(foot_in_contact_after, ang_vel)'''

        ''' local_velocity = Robot.get_3d_local_velocity()
        # linear_velocity = self.data.qpos[:3].copy()
        # roll, pitch, yaw = self.base_rotation
        # target_velocity = np.array([0.5, 0, 0])
        # joint_motion_penalty = -0.001 *  self.dt * (np.linalg.norm(self.get_joint_qacc())**2 + np.linalg.norm(self.get_joint_qvel())**2)
        # joint_torque_penalty = -0.00002 *  self.dt * np.linalg.norm(self.get_joint_torques())**2
        # print(angular_velocity_penalty)
        # keep_leg_rotation_reward_knee = self.keep_leg_rotation_reward_knee(before_joints, after_joints)
        # keep_leg_rotation_reward_hip =self.keep_leg_rotation_reward_hip(before_joints, after_joints) '''
        '''
        FR_qvel = self.data.qvel[6:9]
        FL_qvel = self.data.qvel[9:12]
        # FL_qvel[0] *= -1
        RR_qvel = self.data.qvel[12:15]
        RL_qvel = self.data.qvel[15:18]
        # RL_qvel[0] *= -1
        diagonal_difference = np.linalg.norm(FR_qvel[1:] - RL_qvel[1:]) + np.linalg.norm(FL_qvel[1:] - RR_qvel[1:])
        shoulder_difference = np.linalg.norm(FR_qvel[1] - FL_qvel[1]) + 2 * np.linalg.norm(
            RR_qvel[1] - RL_qvel[1])  # penalize harder on the back legs
        shoulder_center = np.linalg.norm(FR_qvel[1:] + FL_qvel[1:]) + 2 * np.linalg.norm(RR_qvel[1:] + RL_qvel[1:])
        diagonal_difference_foots = np.abs(FR_foot_after[1] - BL_foot_after[1]) + np.abs(
            FL_foot_after[1] - BR_foot_after[1])
        shoulder_difference_foots = - np.linalg.norm(FR_foot_after - FL_foot_after) + 2 + np.linalg.norm(
            BL_foot_after - BR_foot_after)

        diagonal_difference_penalty = np.exp(-0.01 * diagonal_difference)
        shoulder_difference_penalty = - np.exp(-0.01 * shoulder_difference)
        shoulder_difference_penalty_foots = - np.exp(-0.01 * shoulder_difference_foots)
        diagonal_difference_penalty_foots = 5 * np.square(diagonal_difference_foots - 0.25)
        '''

        rewards = 1.5 * forward_reward * 2.5 + healthy_reward + 1.5 * reward_foot_separation_left + 1.5 * reward_foot_separation_right + 1.5 * rotation_reward + 1 * torque_reward + angular_velocity_penalty

        # BEST 2 reward:
        # rewards = 1.5 * forward_reward * 2.5 + healthy_reward + 1.5 * reward_foot_separation_left + 1.5 * reward_foot_separation_right + 1.5 * rotation_reward + 1 * torque_reward + penalty_airtime
        # +  1.5*diagonal_difference + 1.5*diagonal_difference_foots

        terminate = self.terminated
        observation = self._get_obs()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.data.cfrc_ext, -1, 1)))

        total_rewards = rewards - ctrl_cost * 0.5 - contact_cost

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

        self.qpos_history = np.zeros(15 * 12)
        self.qpos_history_counter = 0
        qpos = self.init_joints + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
