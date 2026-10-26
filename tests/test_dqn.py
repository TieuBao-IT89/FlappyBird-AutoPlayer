from dataclasses import asdict, replace
import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from flappy_auto.rl.dqn import DQNConfig, SeedSchedule, _replace_file, load_checkpoint
from flappy_auto.rl.environment import FlappyEnv
from flappy_auto.rl.evaluation import evaluate, random_policy, run_episode, summarize


@pytest.mark.parametrize('values', [
    {'version': 2}, {'seed': True}, {'total_timesteps': 0}, {'total_timesteps': 10}, {'threads': 0},
    {'batch_size': 50_001}, {'learning_rate': float('nan')}, {'gamma': 1.1},
    {'hidden_layers': []}, {'training_seeds': []}, {'training_seeds': [1, 1]}, {'test_seeds': [True]},
    {'training_seeds': [1000]}, {'test_seeds': [0]}, {'test_seeds': [1000]}, {'unknown': 1},
])
def test_reject_invalid_configurations_and_overlapping_seed_splits(values):
    with pytest.raises(ValueError):
        DQNConfig.from_dict(values)


def test_configuration_round_trip():
    config = DQNConfig()
    assert DQNConfig.from_dict(json.loads(json.dumps(asdict(config)))) == config


def test_training_schedule_never_uses_validation_or_test_seeds():
    env = SeedSchedule(FlappyEnv(max_steps=1), [10, 11, 12])
    actual = []
    for index in range(7):
        _, info = env.reset(seed=42 if index == 0 else None)
        actual.append(info['episode_seed'])
        assert env.step(0)[-1]['episode_seed'] == info['episode_seed']
    assert actual == [10, 11, 12, 10, 11, 12, 10]
    assert env.reset(seed=42)[1]['episode_seed'] == 10


def test_capped_episode_and_failed_policy_are_not_final_scores():
    capped = run_episode(lambda obs: 0, 2000, 2)
    assert capped['status'] == 'time_limit' and capped['final_score'] is None
    assert capped['steps'] == 2 and capped['reward'] == pytest.approx(.02)
    failed = run_episode(lambda obs: 2, 2000, 2)
    assert failed['status'] == 'error' and failed['final_score'] is None
    assert failed['error'] and failed['steps'] == 0


def test_seeded_episodes_are_deterministic_and_trace_records_actions():
    trace = io.StringIO()
    first = run_episode(random_policy(8), 2000, 100, trace=csv.writer(trace))
    second = run_episode(random_policy(8), 2000, 100)
    assert first['trajectory_sha256'] == second['trajectory_sha256']
    assert first['score_so_far'] == second['score_so_far'] and first['steps'] == second['steps']
    assert len(trace.getvalue().splitlines()) == first['steps']


class CoastModel:
    def predict(self, obs, deterministic):
        assert deterministic
        return np.array(0), None


def test_evaluation_compares_all_seed_policy_pairs_and_keeps_censoring(tmp_path):
    metadata = {'config': asdict(DQNConfig()), 'model_sha256': 'test'}
    result = evaluate(CoastModel(), [2000, 2001], 2, tmp_path, metadata)
    assert len(result['rows']) == 6
    assert {r['seed'] for r in result['rows']} == {2000, 2001}
    assert all(r['status'] == 'time_limit' and r['final_score'] is None for r in result['rows'])
    assert all(g['completed_final_scores']['n'] == 0 for g in result['summary'])
    output = Path(result['output_directory'])
    assert len((output/'trials.csv').read_text().splitlines()) == 7
    assert json.loads((output/'report.json').read_text())['checkpoint']['model_sha256'] == 'test'


@pytest.mark.parametrize('seeds', [[0], [1000], [2000, 2000], [], [-1], [True]])
def test_evaluation_rejects_non_holdout_seeds_before_creating_report(tmp_path, seeds):
    with pytest.raises(ValueError, match='disjoint'):
        evaluate(CoastModel(), seeds, 2, tmp_path, {'config': asdict(DQNConfig())})
    assert not list(tmp_path.iterdir())


def test_summary_does_not_mix_time_caps_with_completed_scores():
    rows = [{'policy': 'dqn', 'status': status, 'score_so_far': score, 'final_score': final,
             'survival_seconds': 10, 'reward': 1}
            for status, score, final in [('completed', 10, 10), ('completed', 20, 20),
                                          ('time_limit', 100, None), ('error', 0, None)]]
    group, = summarize(rows)
    assert group['completed_final_scores'] == {'n': 2, 'mean': 15, 'median': 15, 'min': 10}
    assert group['thresholds_strictly_over']['20'] == {
        'observed_over': 1, 'known_not_over': 2, 'unknown': 1, 'observed_lower_bound_rate': .25}


def test_interrupted_evaluation_exports_unstarted_trials(tmp_path):
    class InterruptedModel:
        def predict(self, *args, **kwargs):
            raise KeyboardInterrupt

    result = evaluate(InterruptedModel(), [2000, 2001], 2, tmp_path,
                      {'config': asdict(DQNConfig()), 'model_sha256': 'test'})
    assert [r['status'] for r in result['rows']] == ['interrupted']+['not_run']*5
    assert result['summary'][0]['attempted'] == 1
    assert result['summary'][1]['attempted'] == 0
    assert (Path(result['output_directory'])/'report.json').exists()


def test_checkpoint_replacement_retries_transient_windows_locks(tmp_path, monkeypatch):
    source, target = tmp_path/'new.zip', tmp_path/'last.zip'
    source.write_bytes(b'new')
    target.write_bytes(b'old')
    original, attempts = Path.replace, []

    def locked_once(self, destination):
        attempts.append(True)
        if len(attempts) < 3:
            assert target.read_bytes() == b'old'
            raise PermissionError('temporary file lock')
        return original(self, destination)

    monkeypatch.setattr(Path, 'replace', locked_once)
    monkeypatch.setattr('flappy_auto.rl.dqn.time.sleep', lambda seconds: None)
    _replace_file(source, target)
    assert target.read_bytes() == b'new' and len(attempts) == 3


def test_missing_checkpoint_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match='metadata'):
        load_checkpoint(tmp_path/'missing.zip')


def test_invalid_checkpoint_suffix(tmp_path):
    with pytest.raises(ValueError, match='.zip'):
        load_checkpoint(tmp_path/'model')
