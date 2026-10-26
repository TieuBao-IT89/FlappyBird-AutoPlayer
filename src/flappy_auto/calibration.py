"""Interactive calibration; browser calls and Tk callbacks share their owning thread."""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .profiles import GameProfile, HSVRange, VisionConfig, validate_roi
from .vision import annotate, detect, vision_masks


@dataclass(frozen=True)
class DisplayTransform:
    """Map letterboxed Tk coordinates to CSS image coordinates, independent of DPI."""
    image_size: tuple[int, int]
    display_size: tuple[int, int]

    @property
    def scale(self):
        return min(self.display_size[0] / self.image_size[0], self.display_size[1] / self.image_size[1])

    @property
    def offset(self):
        return tuple((d - i * self.scale) / 2 for d, i in zip(self.display_size, self.image_size))

    def image_point(self, x, y, clamp=False):
        px, py = ((v - offset) / self.scale for v, offset in zip((x, y), self.offset))
        width, height = self.image_size
        if not clamp and not (0 <= px < width and 0 <= py < height):
            return None
        return min(width - 1, max(0, int(px))), min(height - 1, max(0, int(py)))


def patch_sample(frame, x, y, radius=2):
    """Use a circular hue medoid and S/V medians to reject JPEG noise/outliers."""
    h, w = frame.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError('Sample lies outside the image')
    patch = frame[max(0, y-radius):min(h, y+radius+1), max(0, x-radius):min(w, x+radius+1)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    hues = hsv[:, 0]
    distances = np.abs(hues[:, None] - hues[None, :])
    hue = hues[np.argmin(np.minimum(distances, 180-distances).sum(axis=1))]
    return int(hue), int(np.median(hsv[:, 1])), int(np.median(hsv[:, 2]))


def sampled_range(samples, hue_margin=10, saturation_margin=55, value_margin=55):
    if not samples:
        raise ValueError('Click at least one color sample first')
    hues = sorted(set(sample[0] for sample in samples))
    gaps = [(hues[(i+1) % len(hues)] - hue) % 180 for i, hue in enumerate(hues)]
    # A single hue covers zero degrees, rather than the entire circle.
    index = int(np.argmax(gaps))
    start, end = hues[(index+1) % len(hues)], hues[index]
    span = (end - start) % 180
    lo_h, hi_h = ((0, 179) if span + 2*hue_margin >= 179
                  else ((start-hue_margin) % 180, (end+hue_margin) % 180))
    lower = (lo_h, max(0, min(s[1] for s in samples)-saturation_margin),
             max(0, min(s[2] for s in samples)-value_margin))
    upper = (hi_h, min(255, max(s[1] for s in samples)+saturation_margin),
             min(255, max(s[2] for s in samples)+value_margin))
    return HSVRange(lower, upper)


class CalibrationSession:
    """Pure editing model shared by the window and calibration tests."""

    def __init__(self, frame, profile: GameProfile, region: str):
        self.frame = frame.copy()
        self.profile = replace(profile, reference_size=(frame.shape[1], frame.shape[0]), region=region, roi=None)
        self.roi = profile.roi
        if self.roi is not None:
            validate_roi(self.roi, self.profile.reference_size)
        self.config = profile.vision
        self.samples = {'bird': [], 'pipe': []}

    def update(self, frame):
        self.profile.validate_capture((frame.shape[1], frame.shape[0]), self.profile.region)
        self.frame = frame.copy()

    def set_roi(self, roi):
        if roi is not None:
            validate_roi(roi, self.profile.reference_size)
        self.roi = tuple(roi) if roi is not None else None
        # Samples belong to the old crop; do not mix them into a new selection.
        self.samples = {'bird': [], 'pipe': []}

    def crop(self):
        if self.roi is None:
            return self.frame
        x, y, w, h = self.roi
        return self.frame[y:y+h, x:x+w]

    def sample(self, target, x, y):
        rx, ry, w, h = self.roi or (0, 0, self.frame.shape[1], self.frame.shape[0])
        if not (rx <= x < rx+w and ry <= y < ry+h):
            raise ValueError('Hãy lấy mẫu bên trong vùng game đã chọn')
        point = patch_sample(self.crop(), x-rx, y-ry)
        self.samples[target].append(point)
        self.config = replace(self.config, **{target: sampled_range(self.samples[target])})

    def preview(self):
        crop = self.crop()
        # Use the same default 320 px processing size as the live detector.
        obs = detect(crop, config=self.config)
        scale = min(1., 320/crop.shape[1])
        small = cv2.resize(crop, (round(crop.shape[1]*scale), round(crop.shape[0]*scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else crop
        bird, pipe, _, _ = vision_masks(small, self.config)
        original = self.frame.copy()
        overlay = annotate(crop, obs, 'bird: %s | pipes: %d' % (bool(obs.bird), len(obs.pipes)))
        if self.roi is None:
            original = overlay
        else:
            x, y, w, h = self.roi
            original[y:y+h, x:x+w] = overlay
            cv2.rectangle(original, (x, y), (x+w-1, y+h-1), (255, 80, 255), 2)
        return original, bird, pipe, obs

    def saved_profile(self):
        return replace(self.profile, roi=self.roi, vision=self.config)


class CalibrationWindow:
    def __init__(self, session: CalibrationSession, path: Path, capture, start, flap, restart):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox
        self.session, self.path, self.capture = session, Path(path), capture
        self.root = tk.Tk()
        self.root.title('Flappy AutoPlayer — Hiệu chỉnh nhận diện')
        width, height = min(1260, self.root.winfo_screenwidth()-50), min(860, self.root.winfo_screenheight()-80)
        self.root.geometry(f'{width}x{height}')
        self.root.minsize(900, 620)
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.live = tk.BooleanVar(value=False)
        self.mode = tk.StringVar(value='roi')
        self.status = tk.StringVar(value='Kéo chọn toàn bộ sân chơi, gồm chim và cả hai đoạn ống. Sau đó lấy mẫu màu.')
        self.info = tk.StringVar()
        self.beak = tk.BooleanVar(value=session.config.use_beak)
        self.eye = tk.BooleanVar(value=session.config.use_eye)
        self._anchor = self._drag = None
        self._images = []
        self._render_job = None
        self._closed = False
        self._dirty = False

        toolbar = ttk.Frame(self.root, padding=6)
        toolbar.pack(fill='x')
        for label, command in [('Chụp ảnh', self.refresh), ('Bắt đầu', lambda: self.action(start)),
                               ('Space', lambda: self.action(flap)), ('Chơi lại', lambda: self.action(restart)),
                               ('Mở profile', self.load), ('Lưu profile…', self.save)]:
            ttk.Button(toolbar, text=label, command=command).pack(side='left', padx=2)
        ttk.Checkbutton(toolbar, text='Ảnh trực tiếp', variable=self.live).pack(side='left', padx=6)
        ttk.Label(self.root, textvariable=self.info, padding=(8, 2)).pack(fill='x')
        modes = ttk.Frame(self.root, padding=(8, 3))
        modes.pack(fill='x')
        for label, value in [('Kéo chọn vùng game', 'roi'), ('Lấy mẫu chim', 'bird'), ('Lấy mẫu ống', 'pipe')]:
            ttk.Radiobutton(modes, text=label, value=value, variable=self.mode).pack(side='left', padx=5)
        ttk.Button(modes, text='Toàn bộ ảnh', command=self.full_frame).pack(side='left', padx=5)
        ttk.Label(modes, text='Nhấn vào ảnh để dừng ảnh trực tiếp.').pack(side='left', padx=6)

        views = ttk.Frame(self.root, padding=6)
        views.pack(fill='both', expand=True)
        self.canvases = []
        for i, title in enumerate(('Ảnh gốc + nhận diện (chọn vùng / màu tại đây)', 'Mặt nạ chim', 'Mặt nạ ống')):
            views.columnconfigure(i, weight=1, uniform='preview')
            panel = ttk.Frame(views)
            panel.grid(row=0, column=i, sticky='nsew', padx=3)
            ttk.Label(panel, text=title, anchor='center').pack(fill='x')
            canvas = tk.Canvas(panel, bg='#20242b', highlightthickness=0, width=280, height=350)
            canvas.pack(fill='both', expand=True)
            canvas.bind('<Configure>', self.schedule_render)
            self.canvases.append(canvas)
        views.rowconfigure(0, weight=1)
        self.canvases[0].bind('<ButtonPress-1>', self.press)
        self.canvases[0].bind('<B1-Motion>', self.drag)
        self.canvases[0].bind('<ButtonRelease-1>', self.release)

        editor = ttk.Frame(self.root, padding=(8, 3))
        editor.pack(fill='x')
        self.bounds = {}
        self.counts = {}
        for row, (target, label) in enumerate((('bird', 'Chim'), ('pipe', 'Ống'))):
            self.counts[target] = tk.StringVar()
            ttk.Label(editor, textvariable=self.counts[target], width=18).grid(row=row, column=0, sticky='w')
            variables = []
            for i, name in enumerate(('H thấp', 'S thấp', 'V thấp', 'H cao', 'S cao', 'V cao')):
                ttk.Label(editor, text=name).grid(row=row, column=1+i*2, padx=(6, 2))
                variable = tk.StringVar()
                ttk.Spinbox(editor, from_=0, to=179 if i % 3 == 0 else 255, width=4,
                            textvariable=variable).grid(row=row, column=2+i*2)
                variables.append(variable)
            self.bounds[target] = variables
            ttk.Button(editor, text='Xóa mẫu', command=lambda key=target: self.clear_samples(key)).grid(row=row, column=13, padx=6)
        options = ttk.Frame(self.root, padding=6)
        options.pack(fill='x')
        ttk.Checkbutton(options, text='Ưu tiên mỏ đỏ', variable=self.beak, command=self.apply).pack(side='left')
        ttk.Checkbutton(options, text='Ưu tiên mắt trắng', variable=self.eye, command=self.apply).pack(side='left')
        ttk.Button(options, text='Áp dụng ngưỡng / Thử nhận diện', command=self.apply).pack(side='left', padx=8)
        ttk.Button(options, text='Màu mặc định', command=self.defaults).pack(side='left')
        ttk.Label(options, text='H thấp > H cao: đi qua 179 → 0.').pack(side='left', padx=8)
        ttk.Label(self.root, textvariable=self.status, wraplength=width-25, padding=8).pack(fill='x')
        self.sync_controls()
        self.root.after(80, self.render)
        self.root.after(150, self.poll)

    def close(self):
        if self._dirty and not self.messagebox.askyesno('Chưa lưu', 'Thoát và bỏ các thay đổi chưa lưu?', parent=self.root):
            return
        self._closed = True
        self.root.destroy()

    def run(self):
        self.root.mainloop()

    def error(self, exc):
        self.live.set(False)
        self.status.set(str(exc))
        self.messagebox.showerror('Không thể hiệu chỉnh', str(exc), parent=self.root)

    def sync_controls(self):
        for target, variables in self.bounds.items():
            color = getattr(self.session.config, target)
            for variable, number in zip(variables, (*color.lower, *color.upper)):
                variable.set(str(number))
            self.counts[target].set(f'{"Chim" if target == "bird" else "Ống"}: {len(self.session.samples[target])} mẫu')
        self.beak.set(self.session.config.use_beak)
        self.eye.set(self.session.config.use_eye)

    def apply(self):
        try:
            ranges = {}
            for target, variables in self.bounds.items():
                values = tuple(int(variable.get()) for variable in variables)
                ranges[target] = HSVRange(values[:3], values[3:])
            self.session.config = replace(self.session.config, **ranges, use_beak=self.beak.get(), use_eye=self.eye.get())
            self._dirty = True
            self.render()
        except ValueError as exc:
            self.error(exc)
            return False
        return True

    def defaults(self):
        self.session.config = replace(VisionConfig(), tracking=self.session.config.tracking)
        self.session.samples = {'bird': [], 'pipe': []}
        self._dirty = True
        self.sync_controls()
        self.render()

    def clear_samples(self, target):
        self.session.samples[target].clear()
        self.sync_controls()
        self.status.set('Đã xóa mẫu; ngưỡng hiện tại được giữ. Nhấp lấy mẫu mới hoặc dùng Màu mặc định.')

    def full_frame(self):
        self.session.set_roi(None)
        self._dirty = True
        self.sync_controls()
        self.render()

    def action(self, callback):
        try:
            callback()
            self.refresh()
        except Exception as exc:
            self.error(exc)

    def refresh(self):
        try:
            self.session.update(self.capture())
            self.render()
        except Exception as exc:
            self.error(exc)

    def poll(self):
        if self._closed:
            return
        if self.live.get():
            self.refresh()
        self.root.after(150, self.poll)

    def transform(self):
        canvas = self.canvases[0]
        h, w = self.session.frame.shape[:2]
        return DisplayTransform((w, h), (max(1, canvas.winfo_width()), max(1, canvas.winfo_height())))

    def press(self, event):
        self.live.set(False)
        point = self.transform().image_point(event.x, event.y)
        if point is None:
            return
        if self.mode.get() == 'roi':
            self._anchor = point
            self._drag = (event.x, event.y)
        else:
            try:
                self.session.sample(self.mode.get(), *point)
                self._dirty = True
                self.sync_controls()
                self.render()
            except ValueError as exc:
                self.error(exc)

    def drag(self, event):
        if self._anchor is not None:
            canvas = self.canvases[0]
            canvas.delete('selection')
            canvas.create_rectangle(*self._drag, event.x, event.y, outline='#ff55ff', width=2, tags='selection')

    def release(self, event):
        if self._anchor is None:
            return
        end = self.transform().image_point(event.x, event.y, clamp=True)
        x0, x1 = sorted((self._anchor[0], end[0]))
        y0, y1 = sorted((self._anchor[1], end[1]))
        self._anchor = None
        if x1-x0 >= 20 and y1-y0 >= 20:
            self.session.set_roi((x0, y0, x1-x0+1, y1-y0+1))
            self._dirty = True
            self.sync_controls()
        else:
            self.status.set('Vùng game quá nhỏ. Hãy kéo chọn toàn bộ sân chơi (ít nhất 20 × 20).')
        self.render()

    def schedule_render(self, event=None):
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
        self._render_job = self.root.after(80, self.render)

    def render(self):
        if self._closed:
            return
        if self._render_job is not None:
            self.root.after_cancel(self._render_job)
            self._render_job = None
        original, bird, pipe, obs = self.session.preview()
        self._images = []
        for canvas, frame in zip(self.canvases, (original, bird, pipe)):
            h, w = frame.shape[:2]
            transform = DisplayTransform((w, h), (max(1, canvas.winfo_width()), max(1, canvas.winfo_height())))
            resized = cv2.resize(frame, (max(1, round(w*transform.scale)), max(1, round(h*transform.scale))),
                                 interpolation=cv2.INTER_NEAREST if frame.ndim == 2 else cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.png', resized)
            if not ok:
                raise ValueError('Không thể tạo ảnh xem trước')
            photo = self.tk.PhotoImage(data=base64.b64encode(encoded).decode('ascii'))
            canvas.delete('all')
            canvas.create_image(*transform.offset, image=photo, anchor='nw')
            self._images.append(photo)
        profile = self.session.profile
        relative = f'phần tử game ({profile.selector})' if profile.region == 'element' else 'toàn bộ viewport'
        self.info.set(f'Gốc tọa độ: {relative} · Ảnh CSS: {profile.reference_size} · ROI: {self.session.roi or "toàn bộ ảnh"}')
        self.status.set(f'Chim: {"đã tìm thấy" if obs.bird else "chưa tìm thấy"} · Cặp ống: {len(obs.pipes)} · '
                        'Mẫu màu dùng vùng 5 × 5; nhấp nhiều lần để mở rộng ngưỡng. Chỉ lấy mẫu phần màu chính.')

    def load(self):
        selected = self.filedialog.askopenfilename(parent=self.root, title='Mở profile', filetypes=[('Profile JSON', '*.json')])
        if not selected:
            return
        try:
            profile = GameProfile.load(Path(selected))
            current = self.session.profile
            if any(getattr(profile, key) != getattr(current, key) for key in ('source', 'url', 'selector', 'viewport')):
                raise ValueError('Profile thuộc website/selector/viewport khác. Hãy mở lại với --calibrate --profile <tệp>.')
            profile.validate_capture(current.reference_size, current.region)
            self.session = CalibrationSession(self.session.frame, profile, current.region)
            self.path = Path(selected)
            self._dirty = False
            self.sync_controls()
            self.render()
        except ValueError as exc:
            self.error(exc)

    def save(self):
        if not self.apply():
            return
        selected = self.filedialog.asksaveasfilename(parent=self.root, title='Lưu profile', defaultextension='.json',
                                                   initialdir=str(self.path.resolve().parent), initialfile=self.path.name,
                                                   filetypes=[('Profile JSON', '*.json')])
        if not selected:
            return
        try:
            self.path = Path(selected)
            self.session.saved_profile().save(self.path)
            self._dirty = False
            self.status.set(f'Đã lưu {self.path}. Chạy bot: python -m flappy_auto --profile "{self.path}" --debug')
        except OSError as exc:
            self.error(exc)
