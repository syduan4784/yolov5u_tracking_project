from collections import deque
import numpy as np
from scipy.signal import savgol_filter

class MotionAnalyzer:
    def __init__(self, fps=30, hist_len=15, use_savgol=True):
        self.fps = fps
        self.hist_len = hist_len
        self.use_savgol = use_savgol
        self.hist = {}  # id -> deque[(t, cx, cy)]

    def update(self, tid, timestamp, cx, cy):
        dq = self.hist.get(tid, deque(maxlen=self.hist_len))
        dq.append((timestamp, float(cx), float(cy)))
        self.hist[tid] = dq

    def _speed_accel(self, dq, meters=False, pix_to_m=1.0):
        if len(dq) < 3: 
            return 0.0, 0.0, 0.0  # speed, accel, heading_deg

        ts = np.array([p[0] for p in dq])
        xs = np.array([p[1] for p in dq])
        ys = np.array([p[2] for p in dq])

        if self.use_savgol and len(dq) >= 7:
            xs = savgol_filter(xs, 7, 2)
            ys = savgol_filter(ys, 7, 2)

        dt = ts[-1] - ts[0]
        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        dist = np.hypot(dx, dy)

        if dt <= 1e-3:
            return 0.0, 0.0, 0.0

        speed = (dist / dt)  # px/s
        if meters:
            speed *= pix_to_m  # m/s

        # Gia tốc xấp xỉ từ 3 điểm cuối
        if len(dq) >= 3:
            dt2 = ts[-1] - ts[-3]
            d2 = np.hypot(xs[-1]-xs[-3], ys[-1]-ys[-3])
            accel = (d2 / max(dt2,1e-3)) - speed
        else:
            accel = 0.0

        heading = np.degrees(np.arctan2(dy, dx))  # -180..180
        return float(speed), float(accel), float(heading)

    def classify(self, cls_name, speed_px_s, cfg):
        # Ngưỡng theo lớp (ví dụ người/xe)
        thr = cfg.get('thresholds', {})
        px_walk = thr.get('pxps_walk', 30)
        px_run  = thr.get('pxps_run', 90)

        # Nếu có map riêng theo lớp:
        class_map = thr.get('by_class', {}).get(cls_name, {})
        px_walk  = class_map.get('pxps_walk', px_walk)
        px_run   = class_map.get('pxps_run', px_run)

        if speed_px_s < px_walk: return "walk"
        if speed_px_s < px_run:  return "jog/run"
        return "fast/vehicle/ride"

    def get_features_and_label(self, tid, cls_name, cfg, pix_to_m=1.0, meters=False):
        dq = self.hist.get(tid, None)
        if not dq: 
            return None
        speed, accel, heading = self._speed_accel(dq, meters=meters, pix_to_m=pix_to_m)
        label = self.classify(cls_name, speed if not meters else speed* (1.0/pix_to_m), cfg)
        return {
            "speed": speed,          # px/s hoặc m/s
            "accel": accel,
            "heading": heading,
            "label": label
        }
