from dataclasses import asdict, replace
import json

import pytest

from flappy_auto.__main__ import make_parser, resolve_profile
from flappy_auto.profiles import GameProfile, HSVRange, VisionConfig


def test_profile_round_trip_unicode_path(tmp_path):
    profile = GameProfile(source='demo', roi=(10, 20, 300, 500), reference_size=(360, 600),
                          vision=VisionConfig(bird=HSVRange((175, 100, 90), (8, 255, 255)), use_eye=False))
    path = tmp_path / 'hồ sơ' / 'chim đỏ.json'
    profile.save(path)
    assert GameProfile.load(path) == profile
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize('field,value', [
    ('version', 2), ('version', True), ('source', 'other'), ('url', 'file:///secret'),
    ('selector', ''), ('viewport', [True, 720]), ('viewport', [480, 0]),
    ('reference_size', [360.5, 600]), ('region', 'screen'), ('roi', [0, 0, 361, 600]),
    ('roi', [-1, 0, 20, 20]), ('roi', [0, 0, 0, 20]),
])
def test_reject_malformed_profile(tmp_path, field, value):
    data = asdict(GameProfile(reference_size=(360, 600)))
    data[field] = value
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Cannot load profile'):
        GameProfile.load(path)


@pytest.mark.parametrize('data', ['{broken', '[]', '{}', '{"version":1,"unknown":true}'])
def test_reject_invalid_schema(tmp_path, data):
    path = tmp_path / 'invalid.json'
    path.write_text(data, encoding='utf-8')
    with pytest.raises(ValueError):
        GameProfile.load(path)


@pytest.mark.parametrize('lower,upper', [((180, 0, 0), (179, 255, 255)),
                                        ((0, 256, 0), (179, 255, 255)),
                                        ((0, 100, 0), (179, 50, 255)),
                                        ((0, 0, 200), (179, 255, 100)),
                                        ((1.5, 0, 0), (10, 255, 255))])
def test_reject_invalid_color_ranges(lower, upper):
    with pytest.raises(ValueError):
        HSVRange(lower, upper)


def test_reject_invalid_nested_vision_and_missing_roi_reference(tmp_path):
    with pytest.raises(ValueError, match='reference_size'):
        GameProfile(roi=(0, 0, 20, 20))
    data = asdict(GameProfile())
    data['vision']['use_beak'] = 'false'
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='booleans'):
        GameProfile.load(path)


def test_cli_precedence_and_demo_source(tmp_path):
    path = tmp_path / 'demo.json'
    saved = GameProfile(source='demo', selector='#game', viewport=(700, 900),
                        roi=(10, 20, 300, 500), reference_size=(360, 600))
    saved.save(path)
    args = make_parser().parse_args(['--profile', str(path)])
    assert resolve_profile(args) == saved
    assert args.demo and args.selector == '#game' and args.viewport == (700, 900)
    assert args.roi == saved.roi
    args = make_parser().parse_args(['--profile', str(path), '--url', 'https://example.com/game',
                                    '--selector', '#new', '--viewport', '900', '1000', '--full-frame'])
    resolved = resolve_profile(args)
    assert not args.demo and args.url == 'https://example.com/game'
    assert args.selector == '#new' and args.viewport == (900, 1000) and args.roi is None
    assert resolved.reference_size == saved.reference_size  # Overrides retain the layout guard.
    args = make_parser().parse_args(['--profile', str(path), '--roi', '20', '30', '200', '400'])
    resolve_profile(args)
    assert args.roi == (20, 30, 200, 400)


def test_cli_original_defaults_and_new_calibration_profile(tmp_path):
    args = make_parser().parse_args([])
    assert resolve_profile(args) is None
    assert not args.demo and args.url == 'https://flappybird.io/'
    assert args.selector == 'canvas' and args.viewport == (480, 720) and args.roi is None
    path = tmp_path / 'new.json'
    args = make_parser().parse_args(['--calibrate', '--demo', '--profile', str(path)])
    assert resolve_profile(args) is None and args.demo
    args = make_parser().parse_args(['--profile', str(path)])
    with pytest.raises(ValueError, match='Cannot load profile'):
        resolve_profile(args)


def test_calibrate_loads_existing_default_profile_and_retains_explicit_roi(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    saved = GameProfile(source='demo', reference_size=(360, 600), vision=VisionConfig(use_eye=False))
    saved.save(tmp_path / 'profiles/demo.json')
    args = make_parser().parse_args(['--demo', '--calibrate', '--roi', '1', '2', '300', '500'])
    assert resolve_profile(args) == saved
    assert args.roi_explicit and args.roi == (1, 2, 300, 500)


def test_reference_dimensions_and_coordinate_origin_must_match():
    profile = GameProfile(reference_size=(360, 600))
    profile.validate_capture((360, 600), 'element')
    with pytest.raises(ValueError, match='calibrate'):
        profile.validate_capture((720, 1200), 'element')
    with pytest.raises(ValueError, match='calibrate'):
        profile.validate_capture((360, 600), 'viewport')
