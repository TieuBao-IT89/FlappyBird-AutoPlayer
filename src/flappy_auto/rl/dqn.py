"""CPU DQN training, validation-only checkpoint selection and verified loading."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import sys
import time

import gymnasium as gym

from .environment import FlappyEnv
from .evaluation import evaluate, model_policy, run_episode, summarize, write_trials
from ..provenance import code_info
from ..replay import report_directory
from ..session_format import write_json


@dataclass(frozen=True)
class DQNConfig:
    version: int = 1
    total_timesteps: int = 200_000
    seed: int = 42
    training_seeds: tuple[int, ...] = tuple(range(32))
    validation_seeds: tuple[int, ...] = (1000, 1001, 1002, 1003)
    test_seeds: tuple[int, ...] = tuple(range(2000, 2020))
    max_steps: int = 3600
    eval_freq: int = 20_000
    learning_rate: float = .0005
    buffer_size: int = 50_000
    learning_starts: int = 2000
    batch_size: int = 64
    gamma: float = .99
    train_freq: int = 4
    target_update_interval: int = 1000
    exploration_fraction: float = .35
    exploration_final_eps: float = .05
    hidden_layers: tuple[int, ...] = (64, 64)
    threads: int = 1

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1:
            raise ValueError('Unsupported DQN configuration version')
        for key in ('total_timesteps', 'max_steps', 'eval_freq', 'buffer_size', 'batch_size',
                    'train_freq', 'target_update_interval', 'threads'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'{key} must be a positive integer')
        for key in ('seed', 'learning_starts'):
            if type(getattr(self, key)) is not int or not 0 <= getattr(self, key) < 2**32:
                raise ValueError(f'{key} must be a nonnegative 32-bit integer')
        if self.learning_starts >= self.total_timesteps or self.batch_size > self.buffer_size:
            raise ValueError('total_timesteps must exceed learning_starts; batch_size must fit buffer_size')
        for key in ('learning_rate', 'gamma', 'exploration_fraction', 'exploration_final_eps'):
            value = getattr(self, key)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{key} must be finite and in (0, 1]')
        for key in ('training_seeds', 'validation_seeds', 'test_seeds', 'hidden_layers'):
            value = getattr(self, key)
            if (not isinstance(value, (list, tuple)) or not value
                    or any(type(v) is not int or v < (1 if key == 'hidden_layers' else 0) for v in value)):
                raise ValueError(f'{key} must contain valid integers')
            object.__setattr__(self, key, tuple(value))
            if key != 'hidden_layers' and len(set(value)) != len(value):
                raise ValueError(f'{key} contains duplicate seeds')
        pools = [set(getattr(self, k)) for k in ('training_seeds', 'validation_seeds', 'test_seeds')]
        if any(pools[a] & pools[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError('Training, validation and test seeds must be disjoint')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data)-set(cls.__dataclass_fields__):
            raise ValueError('Unknown DQN configuration fields')
        return cls(**data)


class SeedSchedule(gym.Wrapper):
    """Cycle only the configured training levels across SB3 automatic resets."""

    def __init__(self, env, seeds):
        super().__init__(env)
        if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
            raise ValueError('SeedSchedule requires unique nonnegative seeds')
        self.seeds, self.index, self.episode_seed = tuple(seeds), 0, None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.index = seed % len(self.seeds)
        self.episode_seed = self.seeds[self.index % len(self.seeds)]
        self.index += 1
        obs, info = self.env.reset(seed=self.episode_seed, options=options)
        return obs, {**info, 'episode_seed': self.episode_seed}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, {**info, 'episode_seed': self.episode_seed}


def _libraries(threads=1):
    try:
        import torch
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise RuntimeError('Install training dependencies: python -m pip install -e ".[dev,rl,dqn]"') from exc
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return torch, DQN


def environment_hash():
    return hashlib.sha256(Path(__file__).with_name('environment.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def training_provenance():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package.glob('*.py')):
        digest.update(source.name.encode())
        digest.update(source.read_bytes().replace(b'\r\n', b'\n'))
    return {**code_info(), 'rl_source_sha256': digest.hexdigest(),
            **{key: version(key) for key in ('gymnasium', 'stable-baselines3', 'torch')}}


def _replace_file(source, destination):
    # Windows scanners can briefly hold a newly written checkpoint without
    # delete sharing. Retry the atomic replacement; never remove the good copy.
    for attempt in range(20):
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05)


def save_checkpoint(model, path, config, provenance, *, status, selection=None):
    path = Path(path)
    temporary = path.with_name(path.stem+'.pending.zip')
    model.save(temporary)
    _replace_file(temporary, path)
    metadata = {'kind': 'flappy-dqn-checkpoint', 'version': 1, 'algorithm': 'DQN', 'policy': 'MlpPolicy',
                'environment_version': 1, 'environment_sha256': environment_hash(),
                'observation_shape': [5], 'actions': 2, 'device': 'cpu', 'config': asdict(config),
                'timesteps': model.num_timesteps, 'gradient_updates': model._n_updates,
                'status': status, 'selection': selection, 'code': provenance,
                'model_sha256': file_hash(path)}
    sidecar = path.with_suffix('.json')
    write_json(sidecar.with_suffix('.pending.json'), metadata)
    _replace_file(sidecar.with_suffix('.pending.json'), sidecar)
    return metadata


def load_checkpoint(path):
    path = Path(path)
    if path.suffix != '.zip':
        raise ValueError('Checkpoint path must end in .zip and have its matching .json metadata')
    try:
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if (not isinstance(metadata, dict) or type(metadata.get('version')) is not int or metadata['version'] != 1
                or metadata.get('kind') != 'flappy-dqn-checkpoint' or metadata.get('algorithm') != 'DQN'
                or metadata.get('environment_version') != 1 or metadata.get('environment_sha256') != environment_hash()
                or metadata.get('observation_shape') != [5] or metadata.get('actions') != 2):
            raise ValueError('Checkpoint environment/schema is incompatible')
        config = DQNConfig.from_dict(metadata['config'])
        if metadata.get('model_sha256') != file_hash(path):
            raise ValueError('Checkpoint SHA256 does not match metadata')
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError(f'Cannot load checkpoint metadata: {exc}') from exc
    _, DQN = _libraries(config.threads)
    model = DQN.load(path, device='cpu')
    if model.observation_space.shape != (5,) or model.action_space.n != 2:
        raise ValueError('Saved model spaces differ from FlappyEnv')
    return model, metadata


def train(config, output):
    torch, DQN = _libraries(config.threads)
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    directory = report_directory(output, 'dqn')
    provenance = training_provenance()
    write_json(directory/'config.json', asdict(config))
    write_json(directory/'code.json', provenance)
    env = Monitor(SeedSchedule(FlappyEnv(max_steps=config.max_steps), config.training_seeds),
                  filename=str(directory/'episodes.monitor.csv'), info_keywords=('score', 'episode_seed'))
    validation_rows = []
    started = time.perf_counter()

    class Validation(BaseCallback):
        best = None
        best_path = None
        last_evaluated = None

        def assess(self):
            rows = [run_episode(model_policy(self.model), seed, config.max_steps) for seed in config.validation_seeds]
            if any(r['status'] == 'error' for r in rows):
                raise RuntimeError('Validation failed: '+str([r['error'] for r in rows if r['error']]))
            validation_rows.extend({**row, 'timesteps': self.num_timesteps} for row in rows)
            write_trials(directory/'validation.csv', validation_rows)
            group = summarize(rows)[0]
            metric = (group['observed_scores_including_censored']['mean'], group['rewards']['mean'])
            selection = {'split': 'validation', 'timesteps': self.num_timesteps,
                         'criterion': 'mean score_so_far, then mean reward', 'summary': group}
            checkpoint = directory/f'checkpoint_{self.num_timesteps:09d}.zip'
            save_checkpoint(self.model, checkpoint, config, provenance, status='checkpoint', selection=selection)
            if self.best is None or metric > self.best:
                self.best = metric
                self.best_path = checkpoint
                write_json(directory/'selection.json', {'checkpoint': checkpoint.name, **selection})
            self.last_evaluated = self.num_timesteps
            print(f'[TRAIN] steps={self.num_timesteps} updates={self.model._n_updates} '
                  f'validation_score={metric[0]:.2f} reward={metric[1]:.2f} elapsed={time.perf_counter()-started:.1f}s', flush=True)

        def _on_training_start(self):
            self.assess()

        def _on_step(self):
            if self.num_timesteps % config.eval_freq == 0:
                self.assess()
            return True

    status, error, model, callback = 'completed', None, None, None
    try:
        model = DQN('MlpPolicy', env, learning_rate=config.learning_rate, buffer_size=config.buffer_size,
                    learning_starts=config.learning_starts, batch_size=config.batch_size, gamma=config.gamma,
                    train_freq=config.train_freq, gradient_steps=1, target_update_interval=config.target_update_interval,
                    exploration_fraction=config.exploration_fraction, exploration_final_eps=config.exploration_final_eps,
                    policy_kwargs={'net_arch': list(config.hidden_layers)}, seed=config.seed, device='cpu', verbose=0)
        callback = Validation()
        print(f'[TRAIN] directory={directory.resolve()}', flush=True)
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
        if callback.last_evaluated != model.num_timesteps:
            callback.assess()
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception as exc:
        status, error = 'error', str(exc)
    finally:
        try:
            if model is not None:
                save_checkpoint(model, directory/'last.zip', config, provenance, status=status)
            if callback is not None and callback.best_path is not None:
                # Immutable periodic files avoid overwriting ZIPs that Windows
                # scanners or readers may still hold. Publish aliases only once.
                shutil.copyfile(callback.best_path, directory/'best.zip')
                shutil.copyfile(callback.best_path.with_suffix('.json'), directory/'best.json')
            result = {'status': status, 'error': error, 'directory': str(directory.resolve()),
                      'timesteps': model.num_timesteps if model is not None else 0,
                      'gradient_updates': model._n_updates if model is not None else 0,
                      'wall_seconds': time.perf_counter()-started, 'test_seeds_used_for_selection': False,
                      'config': asdict(config), 'code': provenance}
            write_json(directory/'training.json', result)
        finally:
            env.close()
    if error:
        raise RuntimeError(error)
    return result


def train_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/dqn'))
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--learning-starts', type=int)
    parser.add_argument('--eval-freq', type=int)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args(argv)
    try:
        config = DQNConfig.from_dict(json.loads(args.config.read_text(encoding='utf-8'))) if args.config else DQNConfig()
        changes = {'total_timesteps': args.timesteps, 'learning_starts': args.learning_starts,
                   'eval_freq': args.eval_freq, 'seed': args.seed}
        config = replace(config, **{k: v for k, v in changes.items() if v is not None})
        result = train(config, args.output)
        print(f'[TRAIN] status={result["status"]} timesteps={result["timesteps"]} output={result["directory"]}', flush=True)
        return 0 if result['status'] == 'completed' else 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluate a frozen DQN on held-out Gymnasium seeds and baselines.')
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/dqn-evaluation'))
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args(argv)
    try:
        model, metadata = load_checkpoint(args.model)
        result = evaluate(model, args.seeds if args.seeds is not None else metadata['config']['test_seeds'],
                          metadata['config']['max_steps'] if args.max_steps is None else args.max_steps, args.output, metadata)
        print(f'[EVAL] report={result["output_directory"]}')
        return 130 if any(r['status'] == 'interrupted' for r in result['rows']) else int(any(r['status'] == 'error' for r in result['rows']))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


def play_main(argv=None):
    import csv

    parser = argparse.ArgumentParser(description='Load a trained DQN and play one headless Gymnasium episode.')
    parser.add_argument('model', type=Path)
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--trace', type=Path, help='Export every observation, action and reward to CSV')
    args = parser.parse_args(argv)
    file = None
    try:
        model, metadata = load_checkpoint(args.model)
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            file = args.trace.open('x', encoding='utf-8', newline='')
            writer = csv.writer(file)
            writer.writerow(('step', 'bird_y', 'velocity', 'pipe_distance', 'gap_top', 'gap_bottom',
                             'action', 'reward', 'score_so_far', 'terminated', 'truncated'))
        else:
            writer = None
        max_steps = metadata['config']['max_steps'] if args.max_steps is None else args.max_steps
        if args.seed < 0 or max_steps < 1:
            raise ValueError('Seed must be nonnegative; max_steps must be positive')
        row = run_episode(model_policy(model), args.seed, max_steps, trace=writer)
        print(json.dumps(row, ensure_ascii=False))
        return int(row['status'] == 'error')
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1
    finally:
        if file is not None:
            file.close()
