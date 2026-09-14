import torch
from einops import rearrange

class Preprocessor:
    def __init__(self, 
        action_mean,
        action_std,
        state_mean,
        state_std,
        proprio_mean,
        proprio_std,
        transform,
        action_low=None,
        action_high=None,
    ):
        self.action_mean = action_mean
        self.action_std = action_std
        self.state_mean = state_mean
        self.state_std = state_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std
        self.transform = transform
        if (action_low is None) != (action_high is None):
            raise ValueError("action_low and action_high must be provided together")
        self.action_low = action_low
        self.action_high = action_high

    @staticmethod
    def _expand_action_parameter(parameter, actions):
        parameter = torch.as_tensor(
            parameter, device=actions.device, dtype=actions.dtype
        ).flatten()
        if actions.shape[-1] % parameter.numel() != 0:
            raise ValueError(
                f"Action dimension {actions.shape[-1]} is not divisible by "
                f"parameter dimension {parameter.numel()}"
            )
        return parameter.repeat(actions.shape[-1] // parameter.numel())

    def clamp_normalized_actions(self, actions):
        """Clamp normalized actions to the simulator's command-space bounds.

        The final dimension may contain one action or a flattened frame-skip
        chunk such as ``(dx_0, dy_0, dx_1, dy_1, ...)``.
        """
        if self.action_low is None:
            return actions

        mean = self._expand_action_parameter(self.action_mean, actions)
        std = self._expand_action_parameter(self.action_std, actions)
        low = self._expand_action_parameter(self.action_low, actions)
        high = self._expand_action_parameter(self.action_high, actions)
        normalized_low = (low - mean) / std
        normalized_high = (high - mean) / std
        return torch.maximum(
            torch.minimum(actions, normalized_high), normalized_low
        )

    def normalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        return (actions - self.action_mean) / self.action_std

    def denormalize_actions(self, actions):
        '''
        actions: (b, t, action_dim)  
        '''
        return actions * self.action_std + self.action_mean
    
    def normalize_proprios(self, proprio):
        '''
        input shape (..., proprio_dim)
        '''
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_states(self, state):
        '''
        input shape (..., state_dim)
        '''
        return (state - self.state_mean) / self.state_std

    def preprocess_obs_visual(self, obs_visual):
        return rearrange(obs_visual, "b t h w c -> b t c h w") / 255.0

    def transform_obs_visual(self, obs_visual):
        transformed_obs_visual = torch.tensor(obs_visual)
        transformed_obs_visual = self.preprocess_obs_visual(transformed_obs_visual)
        transformed_obs_visual = self.transform(transformed_obs_visual)
        return transformed_obs_visual
    
    def transform_obs(self, obs):
        '''
        np arrays to tensors
        '''
        transformed_obs = {}
        transformed_obs['visual'] = self.transform_obs_visual(obs['visual'])
        transformed_obs['proprio'] = self.normalize_proprios(torch.tensor(obs['proprio']))
        return transformed_obs
