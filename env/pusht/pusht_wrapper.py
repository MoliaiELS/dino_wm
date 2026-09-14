import os
import numpy as np
import gym
from env.pusht.pusht_env import PushTEnv
from utils import aggregate_dct

class PushTWrapper(PushTEnv):
    def __init__(
            self, 
            with_velocity=True,
            with_target=True,
        ):
        super().__init__(
            with_velocity=with_velocity,
            with_target=with_target, 
        )
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        rs = np.random.RandomState(seed)
        
        def generate_state():
            if self.with_velocity:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                        0,
                        0,  # agent velocities default 0
                    ]
                )
            else:
                return np.array(
                    [
                        rs.randint(50, 450),
                        rs.randint(50, 450),
                        rs.randint(100, 400),
                        rs.randint(100, 400),
                        rs.randn() * 2 * np.pi - np.pi,
                    ]
                )
        
        init_state = generate_state()
        goal_state = generate_state()
        
        return init_state, goal_state
    
    def update_env(self, env_info):
        self.shape = env_info['shape']
    
    def eval_state(self, goal_state, cur_state):
        """
        Evaluate object-goal coverage while ignoring the agent's final pose.

        ``goal_state`` and ``cur_state`` follow the legacy state layout:
        [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
        """
        goal_state = np.asarray(goal_state)
        cur_state = np.asarray(cur_state)
        task_metrics = self.evaluate_task(
            block_pose=cur_state[2:5],
            goal_pose=goal_state[2:5],
        )
        return {
            "success": task_metrics["success"],
            "coverage": task_metrics["coverage"],
            "position_error": task_metrics["position_error"],
            "angle_error": task_metrics["angle_error"],
        }

    def eval_task_state(self, cur_state):
        """Evaluate a legacy state against the environment's configured target."""
        cur_state = np.asarray(cur_state)
        return self.evaluate_task(block_pose=cur_state[2:5])

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.seed(seed)
        self.reset_to_state = init_state
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
