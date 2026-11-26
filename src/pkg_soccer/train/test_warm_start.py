"""Actor 热启动必须保留部署动作，或在写入前拒绝不兼容权重。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from pkg_soccer.rl_control import Actor
from .export_actor import export
from .train_ppo import build_env_class, load_initial_weights, make_policy_class


class WarmStartContract(unittest.TestCase):
    def model(self, depth=2, width=32):
        torch.set_num_threads(1)
        return PPO(make_policy_class(), build_env_class()(), n_steps=8,
                   batch_size=8, device='cpu', seed=17,
                   policy_kwargs=dict(net_arch=[width] * depth,
                                      activation_fn=torch.nn.Tanh))

    def test_same_contract_preserves_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.model()
            source.save(str(Path(tmp) / 'source.zip'))
            path = str(Path(tmp) / 'actor.npz')
            export(str(Path(tmp) / 'source.zip'), path, obs_dim=41)
            target = self.model()
            with torch.no_grad():
                target.policy.action_net.bias.add_(1.)
            load_initial_weights(target, path, 41, [.45, .2, .7])
            actor = Actor(path)
            for obs in np.random.default_rng(5).uniform(-2, 2, (20, 41)).astype(np.float32):
                np.testing.assert_allclose(target.predict(obs, deterministic=True)[0],
                                           actor(obs), atol=1e-6)

    def test_incompatible_transfer_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.model()
            source.save(str(Path(tmp) / 'source.zip'))
            path = str(Path(tmp) / 'actor.npz')
            export(str(Path(tmp) / 'source.zip'), path, obs_dim=41)
            for target, kwargs in ((self.model(depth=6), {}),
                                   (self.model(width=64), {}),
                                   (self.model(), {'contact_offset': (0., 0.)}),
                                   (self.model(), {'action_scale': [.9, .4, .7]})):
                before = {k: v.clone() for k, v in target.policy.state_dict().items()}
                with self.assertRaises(ValueError):
                    load_initial_weights(target, path, 41, **kwargs)
                for key, value in target.policy.state_dict().items():
                    self.assertTrue(torch.equal(before[key], value), key)
            with np.load(path) as source_weights:
                payload = dict(source_weights)
            payload['role'] = np.str_('shared_2v2')
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, '合约'):
                load_initial_weights(self.model(), path, 41)

    def test_anchor_has_student_gradients_and_frozen_teacher(self):
        from .anchored_ppo import AnchoredPPO
        with tempfile.TemporaryDirectory() as tmp:
            source = self.model()
            source.save(str(Path(tmp) / 'source.zip'))
            path = str(Path(tmp) / 'teacher.npz')
            export(str(Path(tmp) / 'source.zip'), path, obs_dim=41)
            model = AnchoredPPO(make_policy_class(), build_env_class()(),
                                n_steps=8, batch_size=8, n_epochs=2, device='cpu',
                                policy_kwargs=dict(net_arch=[32]*2,
                                                   activation_fn=torch.nn.Tanh))
            load_initial_weights(model, path, 41, [.45, .2, .7])
            model.set_reference(path, 41, [.45, .2, .7], [.1, 0], 'attacker')
            observation = torch.randn(16, 41)
            self.assertEqual(model.anchor_loss(observation)[0].item(), 0.)
            teacher_before = {k: v.clone() for k, v in model.reference_policy.state_dict().items()}
            with torch.no_grad():
                model.policy.action_net.bias.add_(.3)
            loss, mae = model.anchor_loss(observation)
            self.assertGreater(mae.item(), .1)
            loss.backward()
            self.assertGreater(model.policy.action_net.bias.grad.abs().sum().item(), 0.)
            self.assertTrue(all(p.grad is None for p in model.reference_policy.parameters()))
            model.learn(16)
            self.assertIn('train/anchor_loss', model.logger.name_to_value)
            for k, v in model.reference_policy.state_dict().items():
                self.assertTrue(torch.equal(v, teacher_before[k]), k)
            saved = str(Path(tmp) / 'anchored.zip')
            model.save(saved)
            # Standard PPO deployment can read the checkpoint without teacher files.
            exported = str(Path(tmp) / 'exported.npz')
            export(saved, exported, obs_dim=41)
            restored = AnchoredPPO.load(saved, device='cpu')
            with self.assertRaisesRegex(RuntimeError, 'set_reference'):
                restored.train()


if __name__ == '__main__':
    unittest.main()
