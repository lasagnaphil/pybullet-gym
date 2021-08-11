# import os
# os.system("Xvfb :1 -screen 0 1024x768x24 &")
# os.environ['DISPLAY'] = ':1'

from gym.wrappers.monitoring.video_recorder import VideoRecorder

def record_video(env, model, name, video_length=500):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    """
    video_recorder = VideoRecorder(env, path="./videos/{}.mp4".format(name))
    obs = env.reset()
    steps = 0
    episodes = 0
    epi_length = 0
    epi_length_history = []
    total_reward = 0.
    total_reward_history = []

    while True:
        steps += 1
        epi_length += 1
        env.render(mode='human')
        video_recorder.capture_frame()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode: {0}\tLength: {1}\tScore: {2}"
                  .format(episodes, epi_length, total_reward))
            episodes += 1
            epi_length_history.append(epi_length)
            epi_length = 0
            total_reward_history.append(total_reward)
            total_reward = 0.
            env.reset()
        if steps >= video_length:
            if episodes == 0:
                episodes = 1
                epi_length_history.append(epi_length)
                total_reward_history.append(total_reward)
            break
    print()
    print("Average reward: {0}".format(sum(total_reward_history) / len(total_reward_history)))
    print("Average episode length: {0}".format(sum(epi_length_history) / len(epi_length_history)))

    video_recorder.close()

import gym

import pybullet
import pybulletgym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import SubprocVecEnv
from pybulletgym.envs.mujoco.envs.pendulum.inverted_pendulum_env import InvertedPendulumMuJoCoEnv
from pybulletgym.envs.mujoco.envs.pendulum.inverted_double_pendulum_env import InvertedDoublePendulumMuJoCoEnv
from pybulletgym.envs.mujoco.envs.locomotion.walker2d_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.walker2d import Walker2D

class InvertedPendulumCustomEnv(InvertedPendulumMuJoCoEnv):
    def __init__(self, render=False):
        InvertedPendulumMuJoCoEnv.__init__(self, render)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()

        # Step 1
        reward = 1.0
        # Step 2
        # theta = state[1]
        # reward = np.exp(-10.0 * theta**2)

        done = not np.isfinite(state).all() or np.abs(state[1]) > .2

        return state, reward, done, {}

class InvertedPendulumCustomTargetEnv(InvertedPendulumMuJoCoEnv):
    def __init__(self, render=False):
        InvertedPendulumMuJoCoEnv.__init__(self, render)
        self.counter = 0
        self.reset_target()
        self.observation_space = gym.spaces.Box(-np.inf*np.ones([5]), np.inf*np.ones([5]))

    def reset_target(self):
        self.target_pos = np.random.uniform(-1.0, 1.0)
        if self.isRender:
            print("Changed target position to ", self.target_pos)

        if hasattr(self, "_p"):
            if not hasattr(self, "target_visual_shape"):
                self.target_visual_shape = self._p.createVisualShape(shapeType=pybullet.GEOM_SPHERE, radius=0.05,
                                                                     rgbaColor=[1, 0, 0, 1])
                self.target_col_shape = self._p.createCollisionShape(shapeType=pybullet.GEOM_SPHERE, radius=0.001)
            # if hasattr(self, "target_body"):
            #     self._p.removeBody(self.target_body) # removeBody() seems to be buggy
            self.target_body = self._p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                                       baseCollisionShapeIndex=self.target_col_shape,
                                                       baseVisualShapeIndex=self.target_visual_shape,
                                                       basePosition=[self.target_pos, 0, 0],
                                                       useMaximalCoordinates=True)

    def reset(self):
        state = InvertedPendulumMuJoCoEnv.reset(self)
        self.reset_target()
        return np.concatenate((state, [self.target_pos]))

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        x = state[0]
        state = np.concatenate((state, [self.target_pos]))
        reward = np.exp(-10.0 * np.square(x - self.target_pos))

        done = not np.isfinite(state).all() or np.abs(state[1]) > .3
        self.counter += 1
        if done or self.counter % 100 == 0:
            self.reset_target()
            self.counter = 0
        return state, reward, done, {}

class InvertedDoublePendulumCustomEnv(InvertedDoublePendulumMuJoCoEnv):
    def __init__(self, render=False):
        InvertedDoublePendulumMuJoCoEnv.__init__(self, render)

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()
        # upright position: 0.6 (one pole) + 0.6 (second pole) * 0.5 (middle of second pole) = 0.9
        pos_x, _, pos_y = self.robot.pole2.pose().xyz()
        v1, v2 = self.robot.j1.get_state()[1], self.robot.j2.get_state()[1]

        # alive_bonus = 10
        # dist_penalty = 0.01 * pos_x ** 2 + (pos_y - 1.7) ** 2
        # vel_penalty = 1e-3 * v1 ** 2 + 5e-3 * v2 ** 2
        pos_reward = np.exp(-1.0 * pos_x**2) * np.exp(-10.0 * (pos_y - 0.9)**2)
        vel_reward = np.exp(-0.2 * v1**2) * np.exp(-0.2 * v2**2)

        done = pos_y + 0.3 <= 1
        # reward = alive_bonus - dist_penalty - vel_penalty
        reward = pos_reward * vel_reward

        return state, reward, done, {}

class InvertedDoublePendulumCustomTargetEnv(InvertedDoublePendulumMuJoCoEnv):
    def __init__(self, render=False):
        InvertedDoublePendulumMuJoCoEnv.__init__(self, render)
        self.counter = 0
        self.reset_target()
        self.observation_space = gym.spaces.Box(-np.inf*np.ones([12]), np.inf*np.ones([12]))

    def reset_target(self):
        self.target_pos = np.random.uniform(-1.0, 1.0)
        if self.isRender:
            print("Changed target position to {}".format(self.target_pos))

        if hasattr(self, "_p"):
            if not hasattr(self, "target_visual_shape"):
                self.target_visual_shape = self._p.createVisualShape(shapeType=pybullet.GEOM_SPHERE, radius=0.05,
                                                                     rgbaColor=[1, 0, 0, 1])
                self.target_col_shape = self._p.createCollisionShape(shapeType=pybullet.GEOM_SPHERE, radius=0.001)
            # if hasattr(self, "target_body"):
            #     self._p.removeBody(self.target_body) # removeBody() seems to be buggy
            self.target_body = self._p.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0],
                                                       baseCollisionShapeIndex=self.target_col_shape,
                                                       baseVisualShapeIndex=self.target_visual_shape,
                                                       basePosition=[self.target_pos, 0, 0],
                                                       useMaximalCoordinates=True)

    def reset(self):
        state = InvertedDoublePendulumMuJoCoEnv.reset(self)
        # self.reset_target()
        return np.concatenate((state, [self.target_pos]))

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()
        state = self.robot.calc_state()

        pos_x, _, pos_y = self.robot.pole2.pose().xyz()
        v1, v2 = self.robot.j1.get_state()[1], self.robot.j2.get_state()[1]

        pos_reward = np.exp(-10.0 * (pos_x - self.target_pos)**2) * np.exp(-10.0 * (pos_y - 0.9)**2)

        state = np.concatenate((state, [self.target_pos]))
        reward = pos_reward
        done = pos_y + 0.3 <= 1

        self.counter += 1
        if done or self.counter % 100 == 0:
            self.reset_target()
            self.counter = 0

        return state, reward, done, {}

class Walker2DCustomEnv(WalkerBaseMuJoCoEnv):
    def __init__(self, render=False):
        self.robot = Walker2D()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot, render)

        self.x_after = 0

    def step(self, a):
        self.robot.apply_action(a)
        self.scene.global_step()

        alive_bonus = 1.0

        pos_before = self.x_after
        self.x_after = self.robot_body.get_pose()[0]
        x_vel = (self.x_after - pos_before) / self.scene.dt

        power_cost = -1e-3 * np.square(a).sum()

        state = self.robot.calc_state()

        height, ang = state[0], state[1]

        done = not (np.isfinite(state).all() and
                    (np.abs(state[2:]) < 100).all() and
                    (height > -0.2 and height < 1.0) and # height starts at 0 in pybullet
                    (ang > -1.0 and ang < 1.0))

        reward = alive_bonus + x_vel + power_cost
        self.reward += reward

        return state, reward, bool(done), {}

from stable_baselines3.common.env_util import make_vec_env

def train(cls, path, name, video_length=500, total_timesteps=100000, resume_from=None):
    env = make_vec_env(cls, n_envs=32, vec_env_cls=SubprocVecEnv)
    if resume_from:
        model = PPO.load(resume_from, env)
    else:
        model = PPO(MlpPolicy, env, verbose=0, n_steps=256, tensorboard_log='./tensorboard')
    # Use a separate environement for evaluation
    eval_env = cls(render=True)
    # eval_env = cls()
    # Random Agent, before training
    record_video(eval_env, model, name="{}-before".format(name))
    model.learn(total_timesteps=total_timesteps)
    record_video(eval_env, model, name="{}-after".format(name))
    model.save(path="{}-trained.zip".format(name))

def view_model(cls, path, name, video_length):
    eval_env = cls(render=True)
    model = PPO.load(path)
    record_video(eval_env, model, name=name, video_length=video_length)

import sys
if __name__ == "__main__":
    if sys.argv[1] == "train":
        resume_from = sys.argv[3] if len(sys.argv) >= 4 else None
        if sys.argv[2] == "inverted-pendulum":
            train(InvertedPendulumCustomEnv, "inverted-pendulum-trained.zip",
                  name="inverted-pendulum", total_timesteps=100000, resume_from=resume_from)
        elif sys.argv[2] == "inverted-pendulum-target":
            train(InvertedPendulumCustomTargetEnv, "inverted-pendulum-target-trained.zip",
                  name="inverted-pendulum-target", total_timesteps=300000, resume_from=resume_from)
        elif sys.argv[2] == "inverted-double-pendulum":
            train(InvertedDoublePendulumCustomEnv, "inverted-double-pendulum-trained.zip",
                  name="inverted-double-pendulum", total_timesteps=300000, resume_from=resume_from)
        elif sys.argv[2] == "inverted-double-pendulum-target":
            train(InvertedDoublePendulumCustomTargetEnv, "inverted-double-pendulum-target-trained.zip",
                  name="inverted-double-pendulum-target", total_timesteps=300000, resume_from=resume_from)
        elif sys.argv[2] == "walker2d":
            train(Walker2DCustomEnv, "walker2d-trained.zip",
                  name="walker2d", total_timesteps=500000, resume_from=resume_from)
    elif sys.argv[1] == "test":
        if sys.argv[2] == "inverted-pendulum":
            view_model(InvertedPendulumCustomEnv, "inverted-pendulum-trained.zip",
                       name="inverted-pendulum-after", video_length=1000)
        elif sys.argv[2] == "inverted-pendulum-target":
            view_model(InvertedPendulumCustomTargetEnv, "inverted-pendulum-target-trained.zip",
                       name="inverted-pendulum-target-after", video_length=1000)
        elif sys.argv[2] == "inverted-double-pendulum":
            view_model(InvertedDoublePendulumCustomEnv, "inverted-double-pendulum-trained.zip",
                        name="inverted-double-pendulum", video_length=1000)
        elif sys.argv[2] == "inverted-double-pendulum-target":
            view_model(InvertedDoublePendulumCustomTargetEnv, "inverted-double-pendulum-target-trained.zip",
                       name="inverted-double-pendulum-target", video_length=1000)
        elif sys.argv[2] == "walker2d":
            view_model(Walker2DCustomEnv, "walker2d-trained.zip",
                       name="walker2d-after", video_length=1000)

