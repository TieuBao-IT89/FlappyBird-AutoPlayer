"""Opt-in real Chromium + screenshot + CLI loop integration (no game JS reads)."""
import json
import os

import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('FLAPPY_BROWSER_SMOKE') != '1', reason='set FLAPPY_BROWSER_SMOKE=1; needs Chromium')
def test_dqn_cli_drives_real_offline_browser(tmp_path, monkeypatch):
    pytest.importorskip('gymnasium')
    from flappy_auto.__main__ import make_parser, run

    calls = []

    class Model:
        def predict(self, obs, deterministic):
            assert obs.shape == (5,) and obs.dtype == np.float32 and deterministic
            calls.append(obs.copy())
            return np.array(1 if len(calls) % 8 == 0 else 0), None

    monkeypatch.setattr('flappy_auto.rl.dqn.load_checkpoint', lambda path: (Model(), {'test': True}))
    args = make_parser().parse_args(['--demo', '--headless', '--seconds', '4', '--runs', '1',
                                     '--policy', 'dqn', '--model', 'mock.zip', '--save-debug', str(tmp_path)])
    assert run(args) == 0
    assert calls
    directory, = tmp_path.glob('session_*')
    results = json.loads((directory/'results.json').read_text())
    assert results['policy'] == 'dqn' and results['runs'][0]['inferences'] == len(calls)
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    decisions = [e for e in events if e.get('policy_decision', {}).get('action') == 1]
    assert decisions and any(e['action_sent'] for e in decisions)
    assert all(e['observation']['fresh'] for e in decisions if e['action_sent'])
