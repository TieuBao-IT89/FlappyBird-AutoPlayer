"""A conservative threshold controller with a short-term motion estimate."""
from __future__ import annotations

from dataclasses import dataclass
from .vision import Bird, Observation


@dataclass
class Strategy:
    min_flap_interval: float = .18
    deadband_fraction: float = .035
    anticipation_seconds: float = .11
    target_offset_fraction: float = .02
    last_bird: Bird | None = None
    last_seen: float | None = None
    last_flap: float = float('-inf')
    filtered_velocity: float = 0.0

    def should_flap(self, obs: Observation, now: float) -> tuple[bool, str]:
        bird = obs.bird
        if bird is None:
            self.last_bird = None
            self.last_seen = None
            return False, 'bird not detected'
        if self.last_bird is not None and self.last_seen is not None:
            dt = now - self.last_seen
            if .015 <= dt <= .5:
                velocity = (bird.y - self.last_bird.y) / dt
                self.filtered_velocity = self.filtered_velocity * .52 + velocity * .48
        self.last_bird, self.last_seen = bird, now
        target = (obs.pipe.gap_center if obs.pipe else obs.height * .47) + obs.height * self.target_offset_fraction
        horizon = self.anticipation_seconds
        predicted = bird.y + self.filtered_velocity * horizon
        # When far from a pipe, fly near the middle instead of chasing a distant gap.
        if obs.pipe and obs.pipe.left - bird.x > obs.width * .35:
            target = obs.height * .47
        # When entering a gap, be a bit less aggressive with flap spamming.
        threshold = target + obs.height * self.deadband_fraction
        if now - self.last_flap < self.min_flap_interval:
            return False, 'cooldown'
        if predicted > threshold:
            self.last_flap = now
            return True, f'flap: y={bird.y:.0f}, target={target:.0f}, vy={self.filtered_velocity:.0f}'
        return False, f'coast: y={bird.y:.0f}, target={target:.0f}, vy={self.filtered_velocity:.0f}'
