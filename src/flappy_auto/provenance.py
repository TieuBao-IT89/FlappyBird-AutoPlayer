"""Small, explicit provenance records for reproducible offline experiments."""
from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess

import cv2
import numpy as np


def code_info():
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source in sorted(package.glob('*.py')):
        digest.update(source.name.encode())
        digest.update(source.read_bytes().replace(b'\r\n', b'\n'))
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=package, capture_output=True,
                                  text=True, timeout=3, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=package,
                                    capture_output=True, text=True, timeout=3, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    try:
        package_version = version('flappybird-autoplayer')
    except PackageNotFoundError:
        package_version = 'source checkout'
    return {'git_revision': revision, 'git_dirty': dirty, 'source_sha256': digest.hexdigest(),
            'package_version': package_version, 'python': platform.python_version(),
            'platform': platform.platform(), 'numpy': np.__version__, 'opencv': cv2.__version__}
