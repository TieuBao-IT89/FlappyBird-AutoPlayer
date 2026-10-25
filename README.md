# FlappyBird-AutoPlayer 🐤

**A Python + OpenCV + Playwright project that tries to play Flappy Bird using screenshots only.**

**Bot thử chơi Flappy Bird bằng hình ảnh, không đọc biến game hoặc can thiệp mã JavaScript của trò chơi.**

The repository contains a **working, locally testable MVP** and an original offline game arena. Playing on a changing public website is **experimental**, not guaranteed to work without tuning the color thresholds and screenshot region.

## Main features

- Opens a browser with Playwright (Chromium)
- Automatically selects the largest visible `<canvas>` on the page (even when nested inside an iframe)
- Reads Chromium's frame stream and processes the newest image with OpenCV, without queuing old frames
- Detects **yellow/orange bird** and **green pipe pair** with HSV masks
- Predicts bird position and triggers **Space** with cooldown control
- Supports debug overlay, screenshot capture, dry-run, manual start, and customizable region
- Includes a tiny **self-contained local game demo**, unit tests, GitHub Actions CI and MIT license

## Quick start on Windows 11 (PowerShell)

Install Python 3.11 or 3.12 and Git. Open VS Code terminal in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m playwright install chromium
python -m pytest -q
```

If `py -3.11` is unavailable, install a supported Python version or try `py -m venv .venv`. If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope Process Bypass`, then activate again; or run `.\.venv\Scripts\python.exe` without activation.

**Try the included offline game first** (requires no internet once packages and Chromium are installed):

```powershell
python -m flappy_auto --demo --seconds 40 --debug
```

Play **flappybird.io**:

```powershell
python -m flappy_auto --seconds 40 --debug
```

After setup you can also double-click `run_demo.cmd` or `run_site.cmd` in File Explorer.

The website version might need calibration because its current graphics, game overlay, canvas layout and physics may differ from the offline demo. Close with `Ctrl+C` in the console or `Q` in the OpenCV debug window.

### Capture performance

The default viewport is **480 × 720**, with a target of **40 FPS**. The console reports the measured average processing rate; `--fps` sets a ceiling and does not guarantee that rate. Chromium streams JPEG frames, and the bot keeps only the latest frame. Still pages and canvas regions outside the viewport use a screenshot fallback. Coordinates are normalized to CSS pixels, including on Windows displays with scaling enabled.

To reduce the requested capture rate or change the viewport:

```powershell
python -m flappy_auto --debug --seconds 60 --fps 30 --viewport 480 720
```

To run without browser or debug windows:

```powershell
python -m flappy_auto --headless --seconds 60
```

Known third-party ad loaders on `flappybird.io` are blocked by default to reduce rendering and network load. Use `--allow-ads` to disable this filter. The public site remains experimental: menus, ads, game-over screens, and different sprite colors can still confuse detection. A higher capture rate alone does not guarantee a higher score.

### Optional: use system Chrome instead of Playwright's downloaded Chromium

If the browser was not installed by Playwright, you can specify your local Chrome executable:

```powershell
python -m flappy_auto --demo --headless --browser-executable "C:\Program Files\Google\Chrome\Application\chrome.exe"
```

### Capture a still image to diagnose detection

```powershell
python -m flappy_auto --capture-only artifacts\site.png
```

If no canvas is found, capture a viewport and crop it with `--roi` (X Y W H) after inspecting the screenshot:

```powershell
python -m flappy_auto --capture-only artifacts\full.png
python -m flappy_auto --roi 425 100 360 600 --seconds 40 --debug
```

`425 100 360 600` is **illustrative only**, not a confirmed live-game location. When a canvas is detected, ROI is relative to the canvas screenshot; otherwise it is relative to the viewport screenshot.

To record debugging images approximately once a second at the requested frame rate (less often if actual FPS is lower):

```powershell
python -m flappy_auto --demo --save-debug artifacts\debug --seconds 15
```

To inspect detection without controlling the game (start manually):

```powershell
python -m flappy_auto --demo --dry-run --debug
```

A full list of options:

```powershell
python -m flappy_auto --help
```

## Project structure

```text
FlappyBird-AutoPlayer/
├── src/flappy_auto/
│   ├── __main__.py      # CLI and automation loop
│   ├── browser.py       # Playwright screenshots, input, canvas/iframe selection
│   ├── vision.py        # OpenCV bird and pipe detection, debug overlay
│   ├── strategy.py      # Velocity estimate and flap decision
│   └── demo_server.py   # Built-in HTTP server for local arena
├── demo/
│   ├── index.html       # Browser canvas demo page
│   └── game.js          # Original, simple Flappy Bird-like game
├── tests/               # Pytest unit tests
├── docs/architecture.md
├── .github/workflows/ci.yml
├── run_demo.cmd / run_site.cmd
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## How the bot works

```text
Chromium frame stream -> latest BGR image -> HSV color segmentation
                   -> locate yellow bird + green pipes
                   -> calculate next gap and bird velocity
                   -> choose flap / coast -> Playwright Space -> repeat
```

**Known MVP limitations:** object recognition relies on specific sprite colors; it can be confused by backgrounds and skins. Browser screenshot latency limits capture rate, and the heuristic controller does not guarantee high scores. No automatic game-over detection, score reading, AI learning or leaderboard interaction is implemented. No direct probing of protected/private game APIs. See `docs/architecture.md` for tuning instructions and future roadmap.

## Publish to GitHub

The GitHub repository already has a `main` branch. Clone it first so its existing history is preserved:

```powershell
cd E:\Project-OSS
git clone https://github.com/TieuBao-IT89/FlappyBird-AutoPlayer.git FlappyBird-AutoPlayer-GitHub
cd FlappyBird-AutoPlayer-GitHub
```

When importing an extracted source archive, copy `.github/`, `demo/`, `docs/`, `src/`, `tests/`, `.gitignore`, `LICENSE`, `pyproject.toml`, `README.md`, `requirements.txt`, `run_demo.cmd`, and `run_site.cmd` into the clone's root. Replace the initial README with this full README. The archive's enclosing folders are not part of the repository structure.

Keep `.venv/`, `venv/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info/`, `artifacts/`, `.env`, `dist/`, and `build/` out of Git. These are covered by `.gitignore`; do not force-add them.

Review the staged files before committing:

```powershell
git status --short
git add .
git diff --cached --stat
git status --short
git commit -m "feat: initialize FlappyBird AutoPlayer MVP"
git push -u origin main
```


Recommended initial GitHub Issues: improve site-specific detection; configure HSV masks from sample pixels; auto-restart; record scores in demo; add controller comparisons; implement UI; add browser integration tests to CI.

## Respectful use

This project is for learning and testing in a controlled environment. Prefer the bundled offline demo. Review website rules before automation, avoid excessive requests and **do not submit automated scores to public leaderboards**. This project is independent of flappybird.io and does not distribute website sprites.

## License

MIT — see [LICENSE](LICENSE). The local demo assets/code in this repo are original.
