"""Matched fresh-browser rule/DQN trials; keep every failure and capped game."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

from flappy_auto.profiles import GameProfile
from flappy_auto.provenance import code_info
from flappy_auto.replay import report_directory
from flappy_auto.session_format import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=30, help='Per-trial session cap')
    parser.add_argument('--fps', type=float, default=60)
    parser.add_argument('--site', action='store_true', help='Explicitly run flappybird.io instead of offline demo')
    parser.add_argument('--headed', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts/browser-evaluation'))
    args = parser.parse_args()
    if args.trials < 1 or not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.fps)):
        parser.error('trials, seconds and fps must be positive')
    # Validate checkpoint before launching any trials (including rule trials).
    from flappy_auto.rl.dqn import load_checkpoint
    from flappy_auto.rl.browser_policy import browser_policy_hash
    _, checkpoint = load_checkpoint(args.model)
    directory = report_directory(args.output, 'browser')
    write_json(directory/'config.json', {**vars(args), 'model': str(args.model.resolve()), 'output': str(args.output),
                                        'code': code_info(), 'checkpoint': checkpoint,
                                        'policy_source_sha256': browser_policy_hash(),
                                        'vision_profile': asdict(GameProfile()),
                                        'order': 'alternate rule/dqn; reverse on odd trial index'})
    rows = []
    for trial in range(args.trials):
        for policy in (('rule', 'dqn') if trial % 2 == 0 else ('dqn', 'rule')):
            destination = directory/f'{trial+1:02d}-{policy}'
            command = [sys.executable, '-u', '-m', 'flappy_auto', '--policy', policy,
                       '--seconds', str(args.seconds), '--runs', '1', '--fps', str(args.fps),
                       '--save-debug', str(destination)]
            if not args.site:
                command.append('--demo')
            if not args.headed:
                command.append('--headless')
            if policy == 'dqn':
                command.extend(['--model', str(args.model.resolve())])
            print(f'[TRIAL {trial+1}] {policy}', flush=True)
            destination.mkdir(parents=True)
            write_json(destination/'command.json', command)
            with (destination/'console.log').open('w', encoding='utf-8') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.seconds+90)
                    error = None if process.returncode == 0 else f'exit {process.returncode}'
                except subprocess.TimeoutExpired:
                    error = 'browser process timeout'
            results = list(destination.glob('session_*/results.json'))
            result = json.loads(results[0].read_text(encoding='utf-8')) if results else {}
            run = (result.get('runs') or [{}])[0]
            row = {'trial': trial+1, 'source': 'website' if args.site else 'demo', 'policy': policy,
                   'status': 'error' if error else run.get('status', 'error'),
                   'score_so_far': run.get('score_so_far'), 'final_score': run.get('final_score'),
                   'seconds': run.get('seconds'), 'fps': run.get('fps'),
                   'detection_failures': run.get('detection_failures'), 'frames': run.get('frames'),
                   'stale_frames': run.get('stale_frames'), 'inferences': run.get('inferences'),
                   'blocked_decisions': run.get('blocked_decisions'),
                   'mean_frame_age_ms': run.get('mean_frame_age_ms'),
                   'mean_inference_ms': run.get('mean_inference_ms'),
                   'reason': error or run.get('reason', 'missing results'),
                   'session': str(results[0].relative_to(directory)) if results else None}
            rows.append(row)
            with (directory/'trials.csv').open('w', newline='', encoding='utf-8') as output:
                writer = csv.DictWriter(output, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            write_json(directory/'trials.json', rows)
            print(json.dumps(row), flush=True)
    print(f'[DONE] {directory.resolve()}', flush=True)
    return int(any(row['status'] == 'error' for row in rows))


if __name__ == '__main__':
    sys.exit(main())
