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
                 healthy_z_range=(0.15, 0.5),
                 reset_noise_scale=1e-2,
                 terminate_when_unhealthy=True,
                 exclude_current_positions_from_observation=False,
                 frame_skip=40,
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
        return np.array([-0.863, -0.686, -2.818]*4)

    @property
    def upper_limits(self):
        return np.array([0.863, 4.501, -0.888]*4)

    @property
    def init_joints(self):
        return np.array([0, 0, 0.37, 1, 0, 0, 0] + [0, 0.7, -1.4]*4)

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

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        return np.concatenate([qpos, qvel])

    # ------------ reward functions----------------
    def _reward_healthy(self):
        return (self.is_healthy - 1) * 5

    def _reward_lin_vel(self, before_pos, after_pos):
        target_vel = np.array([0.5, 0, 0])
        lin_vel = (after_pos - before_pos) / self.dt
        return np.exp(-10*np.linalg.norm(target_vel - lin_vel))

    def step(self, delta_q):
        action = delta_q + self.data.qpos[-12:]
        action = np.clip(action, a_min=self.lower_limits, a_max=self.upper_limits)

        before_pos = self.data.qpos[:3].copy()
        self.do_simulation(action, self.frame_skip)
        after_pos = self.data.qpos[:3].copy()

        lin_v_track_reward = self._reward_lin_vel(before_pos, after_pos)
        healthy_reward = self._reward_healthy()
        total_rewards = 5.0*lin_v_track_reward + 1.0*healthy_reward

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