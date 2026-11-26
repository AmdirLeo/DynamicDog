"""Opt-in teammate prediction matching the teammate_memory_cv_v1 experiment.

Only this peer's missing-packet prediction changes. Own-body, ball, opponent
prediction and role selection retain their existing runtime behavior; this
does not establish full simulation-to-hardware equivalence.
"""
from .rl_control import RLController


class VelocityTeammateController(RLController):
    def _predict_missing(self, now):
        super()._predict_missing(now)
        if 'teammate' not in self._predicted:
            return
        raw = self._observed.get('teammate')
        track = self.tracks.get('teammate')
        if raw is None or track is None or not track.valid(now):
            self.teammate = None
            self._predicted.discard('teammate')
            return
        age = now - raw.received
        vx, vy, wz = track.velocity
        # Always start from the last received packet, never a predicted pose.
        # Keep the superclass's receipt time and source/sequence metadata.
        self.teammate = self.teammate._replace(
            x=raw.x + vx*age, y=raw.y + vy*age, yaw=raw.yaw + wz*age)


def controller_class(prediction_model='legacy_acceleration'):
    if prediction_model == 'legacy_acceleration':
        return RLController
    if prediction_model == 'regression_velocity':
        return VelocityTeammateController
    raise ValueError('rl_teammate_prediction_model 必须为 '
                     'legacy_acceleration 或 regression_velocity')
