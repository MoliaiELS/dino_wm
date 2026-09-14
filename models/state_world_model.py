"""Action-conditioned oracle-state world model for Experiment A."""

from __future__ import annotations

import torch
from torch import nn

from models.proprio import ProprioceptiveEmbedding
from models.vit import ViTPredictor


class StateWorldModel(nn.Module):
    """Causal transformer dynamics model over state/action tokens.

    States are expected to be normalized by training-split statistics. Actions
    remain in the simulator's bounded command space. At token ``t`` the model
    receives ``(state_t, action_t)`` and predicts ``state_{t+1}`` as a residual.
    """

    def __init__(
        self,
        state_dim=11,
        action_dim=2,
        max_context=20,
        model_dim=128,
        state_emb_dim=64,
        action_emb_dim=32,
        depth=4,
        heads=4,
        mlp_dim=256,
        dim_head=32,
        dropout=0.1,
    ):
        super().__init__()
        if max_context < 1:
            raise ValueError("max_context must be positive")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.max_context = int(max_context)
        self.state_encoder = ProprioceptiveEmbedding(
            num_frames=max_context,
            in_chans=state_dim,
            emb_dim=state_emb_dim,
        )
        self.action_encoder = ProprioceptiveEmbedding(
            num_frames=max_context,
            in_chans=action_dim,
            emb_dim=action_emb_dim,
        )
        self.input_projection = nn.Sequential(
            nn.Linear(state_emb_dim + action_emb_dim, model_dim),
            nn.LayerNorm(model_dim),
        )
        # This is the same causal ViT predictor family used by DINO-WM, with
        # one state/action token in place of a grid of visual patch tokens.
        self.predictor = ViTPredictor(
            num_patches=1,
            num_frames=max_context,
            dim=model_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=dropout,
        )
        self.state_head = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, state_dim),
        )

    def predict_next(self, states, actions):
        """Predict the next state at every causal position in a sequence."""

        if states.ndim != 3 or actions.ndim != 3:
            raise ValueError("states and actions must have shape [batch, time, dim]")
        if states.shape[:2] != actions.shape[:2]:
            raise ValueError("states and actions must be aligned in batch and time")
        if states.shape[-1] != self.state_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("state/action feature dimensions do not match the model")
        if states.shape[1] < 1 or states.shape[1] > self.max_context:
            raise ValueError(
                f"Context length must be in [1, {self.max_context}], got {states.shape[1]}"
            )
        state_emb = self.state_encoder(states)
        action_emb = self.action_encoder(actions)
        tokens = self.input_projection(torch.cat([state_emb, action_emb], dim=-1))
        predicted_tokens = self.predictor(tokens)
        return states + self.state_head(predicted_tokens)

    def forward(self, states, actions):
        """Teacher-forced predictions for a window with one extra state."""

        if states.shape[1] != actions.shape[1] + 1:
            raise ValueError("A training window must contain T actions and T+1 states")
        predicted = self.predict_next(states[:, :-1], actions)
        target = states[:, 1:]
        loss = nn.functional.mse_loss(predicted, target)
        return predicted, loss

    def rollout(self, initial_states, actions, history_actions=None):
        """Autoregressively roll out a future command-space action sequence.

        ``initial_states`` can contain a history. If it contains more than one
        state, the actions between those historical states must be provided as
        ``history_actions``. Returned states contain only future predictions.
        """

        if initial_states.ndim != 3 or actions.ndim != 3:
            raise ValueError("initial_states and actions must be batched sequences")
        if initial_states.shape[0] != actions.shape[0]:
            raise ValueError("State and action batches must match")
        history_length = initial_states.shape[1]
        if history_length < 1:
            raise ValueError("At least one initial state is required")
        if history_actions is None:
            if history_length != 1:
                raise ValueError("Historical actions are required for multiple initial states")
            history_actions = actions.new_empty(actions.shape[0], 0, self.action_dim)
        expected_history_actions = history_length - 1
        if history_actions.shape != (
            actions.shape[0],
            expected_history_actions,
            self.action_dim,
        ):
            raise ValueError("history_actions must connect the supplied initial states")

        state_history = initial_states
        action_history = history_actions
        predictions = []
        for step in range(actions.shape[1]):
            current_action = actions[:, step : step + 1]
            aligned_actions = torch.cat([action_history, current_action], dim=1)
            context_length = min(state_history.shape[1], self.max_context)
            context_states = state_history[:, -context_length:]
            context_actions = aligned_actions[:, -context_length:]
            next_state = self.predict_next(context_states, context_actions)[:, -1:]
            predictions.append(next_state)
            state_history = torch.cat([state_history, next_state], dim=1)
            action_history = torch.cat([action_history, current_action], dim=1)
        if not predictions:
            return initial_states.new_empty(
                initial_states.shape[0], 0, self.state_dim
            )
        return torch.cat(predictions, dim=1)

    def config_dict(self):
        attention = self.predictor.transformer.layers[0][0]
        feed_forward = self.predictor.transformer.layers[0][1]
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_context": self.max_context,
            "model_dim": self.predictor.pos_embedding.shape[-1],
            "state_emb_dim": self.state_encoder.emb_dim,
            "action_emb_dim": self.action_encoder.emb_dim,
            "depth": len(self.predictor.transformer.layers),
            "heads": attention.heads,
            "mlp_dim": feed_forward.net[1].out_features,
            "dim_head": attention.to_qkv.out_features // (3 * attention.heads),
            "dropout": float(self.predictor.dropout.p),
        }
