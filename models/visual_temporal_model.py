"""Action-conditioned temporal adapter over frozen DINO patch tokens."""

from __future__ import annotations

import torch
from torch import nn

from models.vit import ViTPredictor


class VisualTemporalWorldModel(nn.Module):
    """Predict frozen visual tokens while exposing a contextual representation.

    The frozen DINO encoder is deliberately outside this module. Each source
    frame contributes spatial visual tokens, normalized proprioception and the
    outgoing action. The causal ViT is the same predictor family used by the
    original DINO-WM implementation.
    """

    def __init__(
        self,
        patch_count=16,
        visual_dim=384,
        proprio_dim=4,
        action_dim=2,
        max_context=3,
        model_dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        dim_head=32,
        dropout=0.1,
        proprio_loss_weight=0.25,
    ):
        super().__init__()
        if patch_count < 1 or max_context < 1:
            raise ValueError("patch_count and max_context must be positive")
        self.patch_count = int(patch_count)
        self.visual_dim = int(visual_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.max_context = int(max_context)
        self.model_dim = int(model_dim)
        self.proprio_loss_weight = float(proprio_loss_weight)
        self.visual_projection = nn.Sequential(
            nn.Linear(visual_dim, model_dim), nn.LayerNorm(model_dim)
        )
        self.proprio_projection = nn.Sequential(
            nn.Linear(proprio_dim, model_dim), nn.LayerNorm(model_dim)
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, model_dim), nn.LayerNorm(model_dim)
        )
        self.spatial_embedding = nn.Parameter(
            torch.zeros(1, 1, patch_count, model_dim)
        )
        nn.init.normal_(self.spatial_embedding, std=0.02)
        self.predictor = ViTPredictor(
            num_patches=patch_count + 2,
            num_frames=max_context,
            dim=model_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=dropout,
        )
        self.visual_head = nn.Sequential(
            nn.Linear(model_dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, visual_dim)
        )
        self.proprio_head = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, proprio_dim),
        )

    def _validate(self, visual, proprio, actions):
        if visual.ndim != 4 or proprio.ndim != 3 or actions.ndim != 3:
            raise ValueError("visual/proprio/actions must be batched sequences")
        if visual.shape[:2] != proprio.shape[:2] or visual.shape[:2] != actions.shape[:2]:
            raise ValueError("visual, proprio and action time axes must align")
        if visual.shape[2:] != (self.patch_count, self.visual_dim):
            raise ValueError(f"Unexpected visual token shape {visual.shape[2:]}")
        if proprio.shape[-1] != self.proprio_dim or actions.shape[-1] != self.action_dim:
            raise ValueError("Unexpected proprio/action feature dimensions")
        if not 1 <= visual.shape[1] <= self.max_context:
            raise ValueError("Context length exceeds configured maximum")

    def contextualize(self, visual, proprio, actions):
        self._validate(visual, proprio, actions)
        visual_tokens = self.visual_projection(visual) + self.spatial_embedding
        proprio_token = self.proprio_projection(proprio).unsqueeze(2)
        action_token = self.action_projection(actions).unsqueeze(2)
        tokens = torch.cat([visual_tokens, proprio_token, action_token], dim=2)
        batch, time, token_count, dim = tokens.shape
        contextual = self.predictor(tokens.reshape(batch, time * token_count, dim))
        return contextual.reshape(batch, time, token_count, dim)

    def predict_next(self, visual, proprio, actions):
        contextual = self.contextualize(visual, proprio, actions)
        visual_context = contextual[:, :, : self.patch_count]
        proprio_context = contextual[:, :, self.patch_count]
        visual_next = visual + self.visual_head(visual_context)
        proprio_next = proprio + self.proprio_head(proprio_context)
        return visual_next, proprio_next, contextual

    def forward(self, visual, proprio, actions):
        if visual.shape[1] != actions.shape[1] + 1:
            raise ValueError("Training windows require T actions and T+1 frames")
        predicted_visual, predicted_proprio, _ = self.predict_next(
            visual[:, :-1], proprio[:, :-1], actions
        )
        visual_loss = nn.functional.mse_loss(predicted_visual, visual[:, 1:])
        proprio_loss = nn.functional.mse_loss(predicted_proprio, proprio[:, 1:])
        loss = visual_loss + self.proprio_loss_weight * proprio_loss
        return {
            "visual": predicted_visual,
            "proprio": predicted_proprio,
            "loss": loss,
            "visual_loss": visual_loss,
            "proprio_loss": proprio_loss,
        }

    def contextual_representation(self, visual, proprio, history_actions):
        """Represent the latest observation without using a future action."""

        if visual.shape[1] != history_actions.shape[1] + 1:
            raise ValueError("History actions must connect supplied observations")
        zero_action = history_actions.new_zeros(
            history_actions.shape[0], 1, self.action_dim
        )
        aligned_actions = torch.cat([history_actions, zero_action], dim=1)
        contextual = self.contextualize(visual, proprio, aligned_actions)
        last = contextual[:, -1]
        return torch.cat(
            [last[:, : self.patch_count].mean(dim=1), last[:, self.patch_count]],
            dim=-1,
        )

    def rollout(self, visual_history, proprio_history, future_actions, history_actions=None):
        if visual_history.shape[:2] != proprio_history.shape[:2]:
            raise ValueError("Visual and proprio histories must align")
        batch, history_length = visual_history.shape[:2]
        if history_actions is None:
            if history_length != 1:
                raise ValueError("Multiple observations require connecting actions")
            history_actions = future_actions.new_empty(batch, 0, self.action_dim)
        if history_actions.shape != (batch, history_length - 1, self.action_dim):
            raise ValueError("history_actions have the wrong shape")
        visual_states = visual_history
        proprio_states = proprio_history
        action_history = history_actions
        visual_predictions = []
        proprio_predictions = []
        for step in range(future_actions.shape[1]):
            action = future_actions[:, step : step + 1]
            aligned_actions = torch.cat([action_history, action], dim=1)
            context_length = min(visual_states.shape[1], self.max_context)
            visual_next, proprio_next, _ = self.predict_next(
                visual_states[:, -context_length:],
                proprio_states[:, -context_length:],
                aligned_actions[:, -context_length:],
            )
            visual_next = visual_next[:, -1:]
            proprio_next = proprio_next[:, -1:]
            visual_predictions.append(visual_next)
            proprio_predictions.append(proprio_next)
            visual_states = torch.cat([visual_states, visual_next], dim=1)
            proprio_states = torch.cat([proprio_states, proprio_next], dim=1)
            action_history = torch.cat([action_history, action], dim=1)
        if not visual_predictions:
            return (
                visual_history.new_empty(batch, 0, self.patch_count, self.visual_dim),
                proprio_history.new_empty(batch, 0, self.proprio_dim),
            )
        return torch.cat(visual_predictions, dim=1), torch.cat(
            proprio_predictions, dim=1
        )

    def config_dict(self):
        attention = self.predictor.transformer.layers[0][0]
        feed_forward = self.predictor.transformer.layers[0][1]
        return {
            "patch_count": self.patch_count,
            "visual_dim": self.visual_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "max_context": self.max_context,
            "model_dim": self.model_dim,
            "depth": len(self.predictor.transformer.layers),
            "heads": attention.heads,
            "mlp_dim": feed_forward.net[1].out_features,
            "dim_head": attention.to_qkv.out_features // (3 * attention.heads),
            "dropout": float(self.predictor.dropout.p),
            "proprio_loss_weight": self.proprio_loss_weight,
        }
