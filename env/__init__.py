"""Lightweight Gym registrations for the environments used by DINO-WM.

Keep this module free of simulator imports.  In particular, importing the
PushT environment must not require MuJoCo, which is an optional dependency
used by PointMaze only.
"""

from gym.envs.registration import register


# Duplicated here deliberately so registering the lazy PointMaze entry point
# does not import env.pointmaze (and therefore mujoco_py).
U_MAZE_SPEC = "#####\\#GOO#\\###O#\\#OOO#\\#####"


register(
    id="pusht",
    entry_point="env.pusht.pusht_wrapper:PushTWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
register(
    id='point_maze',
    entry_point='env.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec': U_MAZE_SPEC,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 23.85,
        'ref_max_score': 161.86,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5'
    }
)
register(
    id="wall",
    entry_point="env.wall.wall_env_wrapper:WallEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)

register(
    id="deformable_env",
    entry_point="env.deformable_env.FlexEnvWrapper:FlexEnvWrapper",
    max_episode_steps=300,
    reward_threshold=1.0,
)
