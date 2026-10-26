from dataclasses import asdict, replace
import json
import math
import subprocess

import pytest

from flappy_auto.benchmark import BenchmarkConfig, paired_comparisons, run_benchmark, summarize
from flappy_auto.provenance import code_info
from flappy_auto.simulation import ReactivePolicy, SimulationConfig, _collision, make_policy, run_trial


class NeverFlap:
    def should_flap(self, obs, now):
        return False, 'no input'


def trial(config=None, policy=None, **kwargs):
    return run_trial(config or SimulationConfig(), policy or NeverFlap(),
                     **{'seed': 5, 'trial': 0, 'policy_name': 'test', 'mode': 'cold', **kwargs})


def test_no_input_hits_floor_at_analytic_time():
    config = SimulationConfig(first_pipe_x=1000)
    result = trial(config)
    expected = math.sqrt(2*(config.height*.88-config.radius-config.start_y)/config.gravity)
    assert result['status'] == 'completed' and result['reason'] == 'floor collision'
    assert result['survival_seconds'] == pytest.approx(expected, abs=1e-7)
    assert result['final_score'] == result['score_so_far'] == 0
    assert not result['censored']


def test_budget_and_requested_stop_are_censored():
    result = trial(SimulationConfig(seconds=.2))
    assert result['status'] == 'time_limit' and result['censored']
    assert result['final_score'] is None and result['survival_seconds'] == pytest.approx(.2)
    result = trial(stop_requested=lambda t: t >= .1)
    assert result['status'] == 'stopped' and result['final_score'] is None
    assert .1 <= result['survival_seconds'] < .11


@pytest.mark.parametrize('name', ['reactive', 'predictive'])
def test_seed_repeats_trajectory_independent_of_measured_cpu_time(name):
    config = SimulationConfig(seconds=3)
    first, second = trial(config, make_policy(name)), trial(config, make_policy(name))
    assert first['trajectory_sha256'] == second['trajectory_sha256']
    for key in ('status', 'reason', 'score_so_far', 'survival_seconds', 'flaps', 'learned_physics'):
        assert first[key] == second[key]
    assert first['trajectory_sha256'] != trial(config)['trajectory_sha256']
    assert first['survival_seconds'] > trial(config)['survival_seconds']
    assert first['configured_fps'] == 30 and first['decision_ms_p95'] >= 0


def test_input_delay_changes_world_trajectory():
    config = SimulationConfig(seconds=3)
    immediate = trial(replace(config, input_delay=0), ReactivePolicy())
    delayed = trial(replace(config, input_delay=.1), ReactivePolicy())
    assert immediate['trajectory_sha256'] != delayed['trajectory_sha256']


def test_collision_checks_swept_pipe_and_parabolic_apex():
    config = SimulationConfig(first_pipe_x=210, pipe_width=1)
    assert _collision(100, 0, 0, .3, config, [300]).startswith('pipe')
    # Both endpoints are safe but the apex passes through the ceiling.
    config = SimulationConfig()
    assert _collision(30, -300, 0, 600/config.gravity, config, []) == 'ceiling collision'


@pytest.mark.parametrize('change', [{'version': 2}, {'seeds': []}, {'seeds': [True]}, {'seeds': [[1]]},
                                    {'seeds': [1, 1]}, {'policies': ['predictive']}, {'runs_per_seed': 0},
                                    {'modes': ['other']}, {'unknown': 1}, {'simulation': {'gravity': float('nan')}},
                                    {'simulation': {'capture_fps': 241}}, {'simulation': {'radius': 0}},
                                    {'simulation': {'pipe_spacing': 20}}, {'simulation': {'seconds': .001}}])
def test_reject_malformed_benchmark_configuration(change):
    with pytest.raises(ValueError):
        BenchmarkConfig.from_dict(change)


def test_configuration_round_trip_and_warm_reset_retains_learned_state():
    config = BenchmarkConfig.from_dict({'seeds': [4, 9], 'simulation': {'bird_noise': 2}})
    assert BenchmarkConfig.from_dict(json.loads(json.dumps(asdict(config)))) == config
    policy = make_policy('predictive')
    trial(SimulationConfig(seconds=2), policy)
    warm = policy.new_run()
    assert warm.gravity == policy.gravity > 0
    assert warm.jump_speed == policy.jump_speed > 0
    assert warm.last_seen is None and warm.last_flap == float('-inf')


def row(policy, status, score, *, final=None, seed=1, seconds=4):
    return {'policy': policy, 'mode': 'cold', 'seed': seed, 'trial': 0, 'status': status,
            'score_so_far': score, 'final_score': final, 'survival_seconds': seconds}


def test_report_excludes_censored_scores_and_retains_failed_trials():
    rows = [row('reactive', 'completed', 21, final=21), row('reactive', 'completed', 3, final=3, seed=2),
            row('reactive', 'time_limit', 100, seed=3), row('reactive', 'time_limit', 20, seed=4),
            row('reactive', 'error', None, seed=5, seconds=None), row('reactive', 'not_run', None, seed=6, seconds=None)]
    group, = summarize(rows)
    assert group['planned'] == 6 and group['attempted'] == 5
    assert group['completed_scores'] == {'n': 2, 'mean': 12, 'median': 12, 'min': 3}
    assert group['thresholds_strictly_over']['20'] == {
        'observed_over': 2, 'known_not_over': 1, 'unknown': 2,
        'observed_lower_bound_rate': .4, 'completed_only_rate': .5}
    assert group['thresholds_strictly_over']['50']['unknown'] == 2
    assert group['observed_survival_capped']['n'] == 4
    assert group['completed_survival']['n'] == 2


def test_paired_final_score_differences_require_two_completed_runs():
    rows = [row('reactive', 'completed', 4, final=4), row('predictive', 'time_limit', 9),
            row('reactive', 'completed', 3, final=3, seed=2), row('predictive', 'completed', 8, final=8, seed=2)]
    first, second = paired_comparisons(rows)
    assert first['final_score_delta'] is None and not first['both_completed']
    assert first['observed_score_delta_not_final'] == 5
    assert second['final_score_delta'] == 5 and second['both_completed']


def test_real_cold_and_warm_workers_export_every_trial_and_reproducible_config(tmp_path):
    config = BenchmarkConfig(simulation=SimulationConfig(seconds=.1), seeds=(1,), runs_per_seed=2)
    result = run_benchmark(config, tmp_path)
    assert len(result['trials']) == 8
    for policy in config.policies:
        cold = [r for r in result['trials'] if r['mode'] == 'cold' and r['policy'] == policy]
        warm = [r for r in result['trials'] if r['mode'] == 'warm' and r['policy'] == policy]
        assert cold[0]['process_id'] != cold[1]['process_id']
        assert warm[0]['process_id'] == warm[1]['process_id']
        assert warm[0]['trajectory_sha256'] == cold[0]['trajectory_sha256']
        assert all(r['status'] == 'time_limit' and r['final_score'] is None for r in cold+warm)
    from pathlib import Path
    directory = Path(result['output_directory'])
    assert BenchmarkConfig.from_dict(json.loads((directory/'config.json').read_text())) == config
    assert len((directory/'trials.csv').read_text().splitlines()) == 9
    assert len((directory/'comparisons.csv').read_text().splitlines()) == 5
    assert json.loads((directory/'results.json').read_text())['code']['source_sha256']
    assert 'python -m flappy_auto.benchmark --config' in (directory/'report.md').read_text(encoding='utf-8')


def test_worker_failures_are_reported_not_discarded(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError('worker unavailable')

    monkeypatch.setattr(subprocess, 'Popen', unavailable)
    config = BenchmarkConfig(seeds=(1,), policies=('reactive',), modes=('warm',), runs_per_seed=3)
    result = run_benchmark(config, tmp_path)
    assert [r['status'] for r in result['trials']] == ['error', 'not_run', 'not_run']
    assert result['summary'][0]['completed_scores']['n'] == 0
    assert result['summary'][0]['thresholds_strictly_over']['20']['unknown'] == 1


def test_worker_interruption_preserves_all_planned_cases(tmp_path, monkeypatch):
    provenance = code_info()
    monkeypatch.setattr('flappy_auto.benchmark.code_info', lambda: provenance)
    class Interrupted:
        def communicate(self, *args, **kwargs):
            if not hasattr(self, 'terminated'):
                raise KeyboardInterrupt
            return '', ''

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: Interrupted())
    config = BenchmarkConfig(seeds=(1, 5), policies=('reactive',), modes=('cold',), runs_per_seed=2)
    result = run_benchmark(config, tmp_path)
    assert [r['status'] for r in result['trials']] == ['interrupted']+['not_run']*3


def test_worker_timeout_is_an_error_with_unstarted_cases_retained(tmp_path, monkeypatch):
    provenance = code_info()
    monkeypatch.setattr('flappy_auto.benchmark.code_info', lambda: provenance)
    class TimedOut:
        def communicate(self, *args, **kwargs):
            if not hasattr(self, 'terminated'):
                raise subprocess.TimeoutExpired('worker', .01)
            return '', ''

        def terminate(self):
            self.terminated = True

    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: TimedOut())
    config = BenchmarkConfig(seeds=(1,), policies=('reactive',), modes=('warm',), runs_per_seed=2)
    result = run_benchmark(config, tmp_path)
    assert [r['status'] for r in result['trials']] == ['error', 'not_run']
    assert result['trials'][0]['reason'] == 'worker wall-clock timeout'


@pytest.mark.parametrize('output', ['[]', '{"status": "completed"}', 'invalid JSON'])
def test_invalid_worker_output_is_kept_as_failed_trial(tmp_path, monkeypatch, output):
    provenance = code_info()
    monkeypatch.setattr('flappy_auto.benchmark.code_info', lambda: provenance)

    class Invalid:
        returncode = 0

        def communicate(self, *args, **kwargs):
            return output, ''

    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: Invalid())
    config = BenchmarkConfig(seeds=(1,), policies=('reactive',), modes=('cold',), runs_per_seed=1)
    result = run_benchmark(config, tmp_path)
    assert result['trials'][0]['status'] == 'error'
    assert result['summary'][0]['planned'] == 1
