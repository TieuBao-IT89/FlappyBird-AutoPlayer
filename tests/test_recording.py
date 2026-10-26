import json

import numpy as np

from flappy_auto.recording import Recorder
from flappy_auto.strategy import Strategy
from flappy_auto.vision import Bird, Observation


def test_repeated_sessions_preserve_previous_results_and_recorded_images(tmp_path):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    obs = Observation(Bird(30, 50, 6), None, 100, 100)
    first = Recorder(tmp_path)
    first.record(1, .1, .02, 0, frame, obs, False, Strategy())
    first.finish_run(1)
    first.close({'runs': [{'score': 0}]})
    second = Recorder(tmp_path)
    second.close({'runs': [{'score': 2}]})
    assert first.directory != second.directory
    assert json.loads((first.directory / 'results.json').read_text())['runs'][0]['score'] == 0
    assert json.loads((second.directory / 'results.json').read_text())['runs'][0]['score'] == 2
    assert list((first.directory / 'run_001').glob('*.jpg'))
    assert len((first.directory / 'telemetry.csv').read_text().splitlines()) == 2
