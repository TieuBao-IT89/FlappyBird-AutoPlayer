"""Deterministic DQN browser controller with the existing telemetry interface."""
from __future__ import annotations

import time
import hashlib
from pathlib import Path

import numpy as np

from ..strategy import Strategy
from .browser_adapter import BrowserObservationAdapter


def browser_policy_hash():
    """Include both browser RL files, which the legacy package hash omits."""
    digest = hashlib.sha256()
    for name in ('browser_adapter.py', 'browser_policy.py'):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n'))
    return digest.hexdigest()


class DQNController(Strategy):
    """Reuse visual physics telemetry, never the rule-based action planner."""

    def __init__(self, model, limits=None, *, clock=time.perf_counter):
        super().__init__()
        self.model = model
        self.adapter = BrowserObservationAdapter(limits)
        self.clock = clock
        self.last_decision = float('-inf')
        self.policy_decision = None

    def new_run(self):
        controller = DQNController(self.model, self.adapter.limits, clock=self.clock)
        controller.pipe_speed = self.pipe_speed
        return controller

    def action_is_fresh(self, captured_at, now, key_delay=0.):
        return 0 <= now + key_delay - captured_at <= self.adapter.limits.max_frame_age

    def should_flap(self, obs, now):
        state = self.adapter.update(obs, now, now=self.clock())
        # This learns motion only for vision association and existing diagnostic
        # fields. DQN velocity is independently measured by the adapter.
        if state is not None:
            self._observe(obs, now)
        self.filtered_velocity = self.adapter.velocity
        self.policy_decision = {'policy': 'dqn', 'observation': state.tolist() if state is not None else None,
                                'action': None, 'inference_ms': 0., 'reason': self.adapter.reason}
        if state is None:
            return False, f'dqn: {self.adapter.reason}'
        # The trained environment makes one decision per 1/60 second. Faster
        # capture must not repeat an action; slower capture is never replayed.
        if now - self.last_decision < 1 / 60 - .0001:
            self.policy_decision['reason'] = 'decision cadence'
            return False, 'dqn: decision cadence'
        self.last_decision = now
        start = self.clock()
        action, _ = self.model.predict(state, deterministic=True)
        finished = self.clock()
        self.policy_decision['inference_ms'] = (finished-start)*1000
        values = np.asarray(action)
        if values.size != 1 or values.item() not in (0, 1):
            raise ValueError('DQN returned an invalid action; expected scalar 0 or 1')
        action = int(values.item())
        self.policy_decision['action'] = action
        if not self.action_is_fresh(now, finished):
            self.policy_decision['reason'] = 'inference exceeded frame age'
            return False, 'dqn: inference exceeded frame age'
        self.policy_decision['reason'] = 'ready'
        if action:
            self.last_flap = now
        return bool(action), f'dqn: {"flap" if action else "coast"}'
