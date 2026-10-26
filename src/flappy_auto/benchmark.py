"""Seeded closed-loop controller benchmarks, including cold processes and warm runs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .provenance import code_info
from .replay import report_directory
from .session_format import finite_number, write_json
from .simulation import SimulationConfig, make_policy, run_trial


@dataclass(frozen=True)
class BenchmarkConfig:
    version: int = 1
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    seeds: tuple[int, ...] = (1, 5, 12)
    runs_per_seed: int = 2
    policies: tuple[str, ...] = ('reactive', 'predictive')
    modes: tuple[str, ...] = ('cold', 'warm')

    def __post_init__(self):
        if type(self.version) is not int or self.version != 1 or not isinstance(self.simulation, SimulationConfig):
            raise ValueError('Unsupported benchmark configuration version')
        if type(self.runs_per_seed) is not int or self.runs_per_seed < 1:
            raise ValueError('runs_per_seed must be a positive integer')
        for key in ('seeds', 'policies', 'modes'):
            value = getattr(self, key)
            if not isinstance(value, (tuple, list)) or not value:
                raise ValueError(f'{key} must be a nonempty list')
            if any(type(v) is not (int if key == 'seeds' else str) for v in value):
                raise ValueError(f'Invalid entry in {key}')
            object.__setattr__(self, key, tuple(value))
            if len(set(value)) != len(value):
                raise ValueError(f'{key} contains duplicate entries')
        if any(type(s) is not int or s < 0 for s in self.seeds):
            raise ValueError('Seeds must be nonnegative integers')
        if set(self.policies)-{'reactive', 'predictive'} or 'reactive' not in self.policies:
            raise ValueError('Choose reactive/predictive policies and include the reactive baseline')
        if set(self.modes)-{'cold', 'warm'}:
            raise ValueError('Choose cold/warm modes')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data)-set(cls.__dataclass_fields__):
            raise ValueError('Unknown benchmark configuration fields')
        return cls(**{**data, 'simulation': SimulationConfig.from_dict(data.get('simulation', {}))})


def _stats(values):
    values = [v for v in values if v is not None]
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None, 'min': min(values) if values else None}


def summarize(trials):
    groups = []
    for policy, mode in sorted({(r['policy'], r['mode']) for r in trials}):
        rows = [r for r in trials if (r['policy'], r['mode']) == (policy, mode)]
        attempted = [r for r in rows if r['status'] != 'not_run']
        completed = [r for r in rows if r['status'] == 'completed']
        thresholds = {}
        for threshold in (20, 50):
            successes = sum(r.get('score_so_far') is not None and r['score_so_far'] > threshold for r in attempted)
            failures = sum(r.get('final_score') is not None and r['final_score'] <= threshold for r in completed)
            thresholds[str(threshold)] = {
                'observed_over': successes, 'known_not_over': failures,
                'unknown': len(attempted)-successes-failures,
                'observed_lower_bound_rate': successes/len(attempted) if attempted else None,
                'completed_only_rate': sum(r['final_score'] > threshold for r in completed)/len(completed) if completed else None}
        groups.append({'policy': policy, 'mode': mode, 'planned': len(rows), 'attempted': len(attempted),
                       'statuses': dict(Counter(r['status'] for r in rows)),
                       'completed_scores': _stats([r['final_score'] for r in completed]),
                       'observed_scores_including_censored': _stats([r.get('score_so_far') for r in attempted]),
                       'observed_survival_capped': _stats([r.get('survival_seconds') for r in attempted]),
                       'completed_survival': _stats([r.get('survival_seconds') for r in completed]),
                       'processing_fps': _stats([r.get('processing_fps') for r in attempted]),
                       'image_age_ms': _stats([r.get('image_age_ms_mean') for r in attempted]),
                       'decision_ms': _stats([r.get('decision_ms_mean') for r in attempted]),
                       'decision_ms_p95': _stats([r.get('decision_ms_p95') for r in attempted]),
                       'thresholds_strictly_over': thresholds})
    return groups


def paired_comparisons(trials):
    baseline = {(r['seed'], r['trial'], r['mode']): r for r in trials if r['policy'] == 'reactive'}
    comparisons = []
    for row in trials:
        if row['policy'] == 'reactive':
            continue
        base = baseline.get((row['seed'], row['trial'], row['mode']))
        if base is None:
            continue
        completed = row['status'] == base['status'] == 'completed'
        comparisons.append({'seed': row['seed'], 'trial': row['trial'], 'mode': row['mode'], 'policy': row['policy'],
                            'baseline_status': base['status'], 'candidate_status': row['status'],
                            'both_completed': completed,
                            'final_score_delta': row['final_score']-base['final_score'] if completed else None,
                            'observed_score_delta_not_final': row['score_so_far']-base['score_so_far']
                            if row.get('score_so_far') is not None and base.get('score_so_far') is not None else None,
                            'observed_survival_delta_capped': row['survival_seconds']-base['survival_seconds']
                            if row.get('survival_seconds') is not None and base.get('survival_seconds') is not None else None})
    return comparisons


def _csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in rows)


def export_report(directory, config, trials, provenance):
    groups, comparison = summarize(trials), paired_comparisons(trials)
    result = {'version': 1, 'kind': 'closed-loop-simulation-benchmark', 'environment_version': 1,
              'perception': 'synthetic noisy geometry, no browser/OpenCV detection',
              'timing': 'Simulated input/decision delays are fixed; measured CPU time never alters the physics.',
              'config': asdict(config), 'code': provenance, 'trials': trials, 'summary': groups, 'comparisons': comparison}
    write_json(directory/'config.json', asdict(config))
    write_json(directory/'results.json', result)
    _csv(directory/'trials.csv', trials)
    _csv(directory/'comparisons.csv', comparison)
    lines = ['# Benchmark điều khiển có phản hồi', '',
             'Mô phỏng dùng quan sát hình học có nhiễu, **không đo độ chính xác OpenCV hoặc điểm website**. '
             'Mỗi seed/lần thử có cùng trạng thái đầu, vật lý, nhiễu, độ trễ và ngân sách cho mọi bộ điều khiển.', '',
             '`cold`: tiến trình Python mới cho từng ván. `warm`: một tiến trình cho nhóm ván cùng seed/bộ điều khiển; '
             'giữ thông số đã học qua `new_run()`. Mọi lần thử, cả lỗi và lần chưa chạy, nằm trong trials.csv.', '',
             '**Điểm hoàn thành** chỉ dùng ván va chạm thật. Điểm lúc hết giờ/dừng là cận dưới, không được tính thành điểm cuối ván. '
             'Trung bình/trung vị/thấp nhất và số mẫu hoàn thành được tách riêng. Tỷ lệ vượt 20/50 dùng dấu `>`; '
             'cận dưới đã quan sát tính trên mọi lần thử đã khởi động, kèm số chưa biết.', '',
             '| Bộ điều khiển | Chế độ | Hoàn thành / đã thử | Điểm cuối TB / trung vị / thấp nhất | Hết giờ | Lỗi/dừng/chưa chạy |',
             '| --- | --- | --- | --- | --- | --- |']
    for group in groups:
        scores, counts = group['completed_scores'], group['statuses']
        values = ' / '.join('—' if scores[k] is None else f'{scores[k]:.2f}' for k in ('mean', 'median', 'min'))
        lines.append(f'| {group["policy"]} | {group["mode"]} | {scores["n"]} / {group["attempted"]} | {values} | '
                     f'{counts.get("time_limit", 0)} | {sum(n for s, n in counts.items() if s not in ("completed", "time_limit"))} |')
    lines += ['', 'Chi tiết thời gian sống (có giới hạn), FPS xử lý, tuổi ảnh, độ trễ cấu hình, thời gian quyết định và '
              'các ngưỡng 20/50 nằm trong results.json và trials.csv. FPS xử lý là tốc độ tính offline theo đồng hồ máy; '
              'observations_per_sim_second và configured_fps mô tả lịch quan sát trong mô phỏng. '
              'Các số đo CPU có thể thay đổi giữa các lần chạy; trajectory_sha256 và kết quả vật lý có thể đối chiếu tái lập.', '',
              'comparisons.csv ghép đúng seed/lần thử/chế độ với reactive baseline. Chỉ có final_score_delta khi cả hai ván hoàn thành; '
              'các chênh lệch điểm quan sát hoặc thời gian sống bị giới hạn có nhãn riêng.', '',
              'Chạy lại bằng một lệnh (cấu hình đã lưu đầy đủ):', '',
              f'`python -m flappy_auto.benchmark --config "{directory.resolve()/"config.json"}" --output artifacts/benchmarks`', '',
              f'Mã nguồn: `{provenance.get("git_revision")}`; SHA256: `{provenance["source_sha256"]}`.', '']
    (directory/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def _failure(case, status, reason):
    return {**case, 'scenario_seed': case['seed']+case['trial']*1_000_003, 'status': status, 'reason': reason,
            'process_id': None, 'score_so_far': None, 'final_score': None, 'censored': True,
            'survival_seconds': None, 'trajectory_sha256': None}


def _valid_worker_row(row, case):
    if (not isinstance(row, dict) or any(row.get(k) != v for k, v in case.items())
            or row.get('status') not in ('completed', 'time_limit', 'stopped', 'error')):
        return False
    score, final = row.get('score_so_far'), row.get('final_score')
    if score is not None and (type(score) is not int or score < 0):
        return False
    for key in ('survival_seconds', 'processing_fps', 'image_age_ms_mean', 'decision_ms_mean', 'decision_ms_p95'):
        if row.get(key) is not None and not finite_number(row[key]):
            return False
    if row['status'] != 'error' and (type(score) is not int or score < 0 or not finite_number(row.get('survival_seconds'))):
        return False
    if row['status'] == 'completed':
        if type(final) is not int or final != score:
            return False
    elif final is not None:
        return False
    json.dumps(row, allow_nan=False)
    return True


def run_benchmark(config, output, *, worker_timeout=300):
    if not isinstance(worker_timeout, (int, float)) or not 0 < worker_timeout < float('inf'):
        raise ValueError('Worker timeout must be finite and positive')
    directory = report_directory(output, 'benchmark')
    provenance = code_info()
    groups = []
    for policy in config.policies:
        for mode in config.modes:
            for seed in config.seeds:
                cases = [{'policy': policy, 'mode': mode, 'seed': seed, 'trial': i} for i in range(config.runs_per_seed)]
                groups.extend([[case] for case in cases] if mode == 'cold' else [cases])
    rows, interrupted = [], False
    for cases in groups:
        if interrupted:
            rows.extend(_failure(c, 'not_run', 'benchmark interrupted before this group') for c in cases)
            continue
        print(f'[BENCH] {cases[0]["policy"]}/{cases[0]["mode"]} seed={cases[0]["seed"]} trials={[c["trial"] for c in cases]}', flush=True)
        payload = json.dumps({'simulation': asdict(config.simulation), 'cases': cases})
        process, stdout, failure_status, failure_reason = None, '', 'error', 'worker returned no result'
        try:
            process = subprocess.Popen([sys.executable, '-m', 'flappy_auto.benchmark', '--worker'],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate(payload, timeout=worker_timeout)
            if process.returncode:
                failure_reason = f'worker exit {process.returncode}: {stderr[-500:]}'
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            failure_status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
            failure_reason = 'benchmark interrupted' if isinstance(exc, KeyboardInterrupt) else 'worker wall-clock timeout'
            interrupted = isinstance(exc, KeyboardInterrupt)
            if process is not None:
                process.terminate()
                try:
                    stdout, _ = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    stdout, _ = process.communicate()
        except OSError as exc:
            failure_reason = str(exc)
        received = []
        try:
            for line in stdout.splitlines():
                row = json.loads(line)
                case = cases[len(received)]
                if not _valid_worker_row(row, case):
                    raise ValueError('worker result does not match planned case')
                received.append(row)
        except (ValueError, IndexError, TypeError):
            failure_reason = 'invalid worker output after completed result prefix'
        rows.extend(received)
        for index, case in enumerate(cases[len(received):]):
            rows.append(_failure(case, failure_status if index == 0 else 'not_run', failure_reason))
        # A durable partial report includes all cases still planned, so a
        # failure never silently disappears from the denominator.
        pending_cases = [c for group in groups for c in group][len(rows):]
        export_report(directory, config, rows+[_failure(c, 'not_run', 'pending') for c in pending_cases], provenance)
    result = export_report(directory, config, rows, provenance)
    result['output_directory'] = str(directory.resolve())
    return result


def _worker():
    payload = json.load(sys.stdin)
    config = SimulationConfig.from_dict(payload['simulation'])
    policy = None
    for case in payload['cases']:
        try:
            policy = make_policy(case['policy']) if policy is None else policy.new_run()
            row = run_trial(config, policy, seed=case['seed'], trial=case['trial'],
                            policy_name=case['policy'], mode=case['mode'])
        except Exception as exc:
            row = _failure(case, 'error', str(exc))
            policy = None
        print(json.dumps(row, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/benchmarks'))
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--runs', type=int)
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--fps', type=int)
    parser.add_argument('--policies', choices=('reactive', 'predictive'), nargs='+')
    parser.add_argument('--modes', choices=('cold', 'warm'), nargs='+')
    parser.add_argument('--worker-timeout', type=float, default=300)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        _worker()
        return 0
    try:
        config = BenchmarkConfig.from_dict(json.loads(args.config.read_text(encoding='utf-8'))) if args.config else BenchmarkConfig()
        overrides = {k: v for k, v in {'seeds': args.seeds, 'runs_per_seed': args.runs,
                                      'policies': args.policies, 'modes': args.modes}.items() if v is not None}
        simulation = replace(config.simulation, **{k: v for k, v in {'seconds': args.seconds, 'capture_fps': args.fps}.items() if v is not None})
        config = replace(config, simulation=simulation, **overrides)
        result = run_benchmark(config, args.output, worker_timeout=args.worker_timeout)
        print(f'[BENCH] report={result["output_directory"]}', flush=True)
        return int(any(r['status'] in ('error', 'interrupted', 'not_run') for r in result['trials']))
    except (OSError, ValueError, TypeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
