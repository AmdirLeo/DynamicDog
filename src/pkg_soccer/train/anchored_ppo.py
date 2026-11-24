"""SB3 PPO with a frozen reference actor in every minibatch loss.

The reference constrains cumulative drift, unlike target_kl which only compares
against the last rollout policy. The critic and exploration distribution remain
trainable. The reference must share observation/action semantics with the actor.
The PPO update follows Stable-Baselines3 2.9's continuous-action implementation.
"""
import copy

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class AnchoredPPO(PPO):
    def set_reference(self, path, obs_dim, action_scale, contact_offset, role,
                      coefficient=2.0):
        from .train_ppo import load_initial_weights
        if not np.isfinite(coefficient) or coefficient <= 0:
            raise ValueError('anchor coefficient must be positive')
        # Validate/copy using exactly the same contract as the deployment loader.
        self.reference_policy = copy.deepcopy(self.policy)
        holder = type('PolicyHolder', (), {'policy': self.reference_policy})()
        load_initial_weights(holder, path, obs_dim, action_scale, contact_offset, role)
        self.reference_policy.requires_grad_(False)
        self.reference_policy.set_training_mode(False)
        self.anchor_coefficient = float(coefficient)

    def _excluded_save_params(self):
        # Deployment only needs the normal policy. Training resume must explicitly
        # attach the reference again; never silently continue unconstrained.
        return super()._excluded_save_params() + ['reference_policy']

    def anchor_loss(self, observations):
        latent = self.policy.mlp_extractor.forward_actor(
            self.policy.extract_features(observations))
        mean = self.policy.action_net(latent)
        with torch.no_grad():
            ref = self.reference_policy
            target = ref.action_net(ref.mlp_extractor.forward_actor(
                ref.extract_features(observations)))
        # Action matching preserves physical commands; logits also constrain
        # saturated tanh outputs whose action gradients alone nearly vanish.
        action_error = torch.tanh(mean) - torch.tanh(target)
        loss = action_error.square().mean() + .05 * (mean-target).square().mean()
        return loss, action_error.detach().abs().mean()

    def train(self):
        if not hasattr(self, 'reference_policy'):
            raise RuntimeError('AnchoredPPO requires set_reference before learning')
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_vf = (None if self.clip_range_vf is None else
                   self.clip_range_vf(self._current_progress_remaining))
        stats = {key: [] for key in ('entropy_loss', 'policy_gradient_loss',
                                    'value_loss', 'clip_fraction', 'approx_kl',
                                    'anchor_loss', 'anchor_action_mae', 'loss')}
        keep_training = True
        for epoch in range(self.n_epochs):
            for batch in self.rollout_buffer.get(self.batch_size):
                values, log_prob, entropy = self.policy.evaluate_actions(
                    batch.observations, batch.actions)
                advantages = batch.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages-advantages.mean()) / (advantages.std()+1e-8)
                ratio = torch.exp(log_prob-batch.old_log_prob)
                policy_loss = -torch.min(advantages*ratio,
                    advantages*torch.clamp(ratio, 1-clip_range, 1+clip_range)).mean()
                values = values.flatten()
                if clip_vf is not None:
                    values = batch.old_values + torch.clamp(values-batch.old_values,
                                                           -clip_vf, clip_vf)
                value_loss = F.mse_loss(batch.returns, values)
                entropy_loss = log_prob.mean() if entropy is None else -entropy.mean()
                anchor_loss, action_mae = self.anchor_loss(batch.observations)
                loss = (policy_loss + self.ent_coef*entropy_loss + self.vf_coef*value_loss
                        + self.anchor_coefficient*anchor_loss)
                with torch.no_grad():
                    log_ratio = log_prob-batch.old_log_prob
                    kl = (torch.exp(log_ratio)-1-log_ratio).mean().item()
                terms = dict(entropy_loss=entropy_loss.item(),
                             policy_gradient_loss=policy_loss.item(),
                             value_loss=value_loss.item(),
                             clip_fraction=((ratio-1).abs()>clip_range).float().mean().item(),
                             approx_kl=kl, anchor_loss=anchor_loss.item(),
                             anchor_action_mae=action_mae.item(), loss=loss.item())
                for name, value in terms.items():
                    stats[name].append(value)
                if self.target_kl is not None and kl > 1.5*self.target_kl:
                    keep_training = False
                    if self.verbose:
                        print(f'Early stopping at epoch {epoch}: KL={kl:.6f}')
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not keep_training:
                break
        for name, values in stats.items():
            self.logger.record('train/'+name, np.mean(values))
        self.logger.record('train/explained_variance', explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()))
        self.logger.record('train/std', torch.exp(self.policy.log_std).mean().item())
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/clip_range', clip_range)
