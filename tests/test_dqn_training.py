from dataclasses import replace
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

pytest.importorskip('stable_baselines3', reason='Install the dqn extra to test real PyTorch training')

from flappy_auto.rl.dqn import DQNConfig, evaluate_main, load_checkpoint, play_main, train, train_main
from flappy_auto.rl.evaluation import model_policy, run_episode


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    config = DQNConfig(total_timesteps=64, learning_starts=8, buffer_size=128, batch_size=8,
                       hidden_layers=(8, 8), max_steps=8, eval_freq=32,
                       training_seeds=(1, 2), validation_seeds=(101,), test_seeds=(201, 202))
    result = train(config, tmp_path_factory.mktemp('dqn'))
    return config, result, Path(result['directory'])


def test_real_gradient_updates_and_lossless_model_round_trip(trained):
    config, result, directory = trained
    assert result['status'] == 'completed' and result['timesteps'] == 64
    assert result['gradient_updates'] > 0 and not result['test_seeds_used_for_selection']
    model, metadata = load_checkpoint(directory/'last.zip')
    initial, _ = load_checkpoint(directory/'best.zip')  # All validation runs are equally capped at 8 steps.
    assert metadata['timesteps'] == 64 and metadata['gradient_updates'] > 0
    assert any(not np.array_equal(a.detach().numpy(), b.detach().numpy())
               for a, b in zip(model.q_net.parameters(), initial.q_net.parameters()))
    second, _ = load_checkpoint(directory/'last.zip')
    assert run_episode(model_policy(model), 201, 20)['trajectory_sha256'] == run_episode(model_policy(second), 201, 20)['trajectory_sha256']
    validation = list(csv.DictReader((directory/'validation.csv').open()))
    assert {int(row['seed']) for row in validation} == {101}
    with (directory/'episodes.monitor.csv').open() as monitor:
        next(monitor)
        assert {int(row['episode_seed']) for row in csv.DictReader(monitor)} <= {1, 2}


@pytest.mark.parametrize('field,value', [('model_sha256', 'bad'), ('environment_sha256', 'wrong'),
                                       ('actions', 3), ('version', 2)])
def test_reject_tampered_or_incompatible_checkpoint(trained, tmp_path, field, value):
    _, _, directory = trained
    for suffix in ('.zip', '.json'):
        shutil.copyfile(directory/('last'+suffix), tmp_path/('last'+suffix))
    sidecar = tmp_path/'last.json'
    metadata = json.loads(sidecar.read_text())
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path/'last.zip')


def test_play_and_evaluation_cli_load_frozen_model(trained, tmp_path):
    _, _, directory = trained
    path = str(directory/'last.zip')
    assert play_main([path, '--max-steps', '5', '--trace', str(tmp_path/'trace.csv')]) == 0
    assert len((tmp_path/'trace.csv').read_text().splitlines()) == 6
    assert evaluate_main([path, '--output', str(tmp_path/'eval'), '--max-steps', '5']) == 0
    assert evaluate_main([path, '--output', str(tmp_path/'bad'), '--seeds', '1']) == 1
    assert evaluate_main([path, '--max-steps', '0']) == 1


def test_training_interruption_saves_partial_checkpoint(tmp_path, monkeypatch, trained):
    from stable_baselines3 import DQN

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(DQN, 'learn', interrupt)
    result = train(trained[0], tmp_path)
    assert result['status'] == 'interrupted'
    _, metadata = load_checkpoint(Path(result['directory'])/'last.zip')
    assert metadata['status'] == 'interrupted' and metadata['timesteps'] == 0


def test_training_failure_is_reported_with_partial_checkpoint(tmp_path, monkeypatch, trained):
    from stable_baselines3 import DQN

    def fail(*args, **kwargs):
        raise RuntimeError('training test failure')

    monkeypatch.setattr(DQN, 'learn', fail)
    with pytest.raises(RuntimeError, match='training test failure'):
        train(trained[0], tmp_path)
    path, = tmp_path.glob('*/training.json')
    assert json.loads(path.read_text())['status'] == 'error'
    assert (path.parent/'last.zip').exists()


def test_training_cli_round_trip_config(trained, tmp_path):
    _, _, directory = trained
    assert train_main(['--config', str(directory/'config.json'), '--timesteps', '16',
                       '--learning-starts', '4', '--output', str(tmp_path)]) == 0
    assert train_main(['--timesteps', '1']) == 1
