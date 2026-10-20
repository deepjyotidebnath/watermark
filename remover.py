"""
Video watermark remover (OpenCV inpainting).
Use only on videos you own or have permission to edit.
"""
import os
import subprocess
import tempfile

import cv2
import numpy as np
import imageio_ffmpeg

INPAINT_FLAGS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def video_info(path):
    cap = cv2.VideoCapture(path)
    info = {
        "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def read_frame(path, index=0):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError("Could not read frame")
    return frame


def sample_frames(path, n=30):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), min(n, max(total, 1))).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def rect_mask(shape_hw, x, y, w, h):
    mask = np.zeros(shape_hw, np.uint8)
    mask[y:y + h, x:x + w] = 255
    return mask


def auto_mask(path, n=30, std_thresh=10.0, dilate=9, min_area=40):
    """Detect a static overlay in a video."""
    return auto_mask_from_frames(sample_frames(path, n), std_thresh, dilate, min_area)


def auto_mask_from_frames(frames, std_thresh=10.0, dilate=9, min_area=40):
    """Static overlay = low variance across frames/images AND strong edges."""
    if len(frames) < 5:
        raise ValueError("Need at least 5 different frames/images for auto-detection")
    stack = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
    std = stack.std(axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)

    gx = cv2.Sobel(median, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(median, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    edges = mag > (mag.mean() + 1.5 * mag.std())

    mask = ((std < std_thresh) & edges).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, n_lbl):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return cv2.dilate(clean, np.ones((dilate, dilate), np.uint8))


def overlay_mask(frame, mask, alpha=0.5):
    red = frame.copy()
    red[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(red, alpha, frame, 1 - alpha, 0)


def process_video(src, dst, mask, method="telea", radius=4, pad=20, progress=None):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty")
    info = video_info(src)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, info["w"])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, info["h"])
    roi_mask = np.ascontiguousarray(mask[y0:y1, x0:x1])
    flag = INPAINT_FLAGS[method]

    tmp_dir = tempfile.mkdtemp()
    tmp_video = os.path.join(tmp_dir, "silent.mp4")
    cap = cv2.VideoCapture(src)
    writer = cv2.VideoWriter(
        tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["w"], info["h"])
    )

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
        frame[y0:y1, x0:x1] = cv2.inpaint(roi, roi_mask, radius, flag)
        writer.write(frame)
        i += 1
        if progress and info["frames"]:
            progress(min(i / info["frames"], 1.0))
    cap.release()
    writer.release()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y", "-i", tmp_video, "-i", src,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", dst,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        os.remove(tmp_video)
        os.rmdir(tmp_dir)
    except OSError:
        pass
    return dst


def pct_rect_mask(shape_hw, x_pct, y_pct, w_pct, h_pct):
    """Rectangle mask from percentage coordinates (works across image sizes)."""
    h, w = shape_hw
    x, y = int(w * x_pct / 100), int(h * y_pct / 100)
    bw, bh = max(int(w * w_pct / 100), 1), max(int(h * h_pct / 100), 1)
    return rect_mask(shape_hw, x, y, bw, bh)


def inpaint_image(img, mask, method="telea", radius=4, pad=20):
    """Remove masked area from a BGR image array. Returns a new image."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return img.copy()
    h, w = img.shape[:2]
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, w)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, h)
    out = img.copy()
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])
    roi_mask = np.ascontiguousarray(mask[y0:y1, x0:x1])
    out[y0:y1, x0:x1] = cv2.inpaint(roi, roi_mask, radius, INPAINT_FLAGS[method])
    return out


def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    return img


def encode_png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Encoding failed")
    return buf.tobytes()


# ======================================================================
# Moving / dynamic watermark (e.g. a text link that drifts or jumps)
# ======================================================================
def _grad(gray):
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def build_template(frame, box, text_mask=True, dilate=5):
    """Template (gradient map) + removal mask from a box around the watermark."""
    x, y, w, h = box
    roi = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tpl = _grad(gray)
    if text_mask:
        k = max(3, (min(w, h) // 3) | 1)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        d = cv2.max(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se),
                    cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se))
        _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
        if m.mean() < 255 * 0.02:
            m = np.full((h, w), 255, np.uint8)
    else:
        m = np.full((h, w), 255, np.uint8)
    tpl = (tpl * (m > 0)).astype(np.float32)
    return tpl, m


class MovingLocator:
    """Finds the watermark in every frame (local search first, full-frame fallback)."""

    def __init__(self, tpl, mask, scales=(1.0,)):
        self.items = []
        for s in scales:
            if s == 1.0:
                t, m = tpl, mask
            else:
                size = (max(int(tpl.shape[1] * s), 3), max(int(tpl.shape[0] * s), 3))
                t = cv2.resize(tpl, size)
                m = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
            self.items.append((t.astype(np.float32), m))
        self.last = None

    def search(self, grad, region):
        x0, y0, x1, y1 = region
        win = grad[y0:y1, x0:x1]
        best = None
        for t, m in self.items:
            th, tw = t.shape
            if win.shape[0] < th or win.shape[1] < tw:
                continue
            res = np.nan_to_num(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED))
            _, mx, _, loc = cv2.minMaxLoc(res)
            if best is None or mx > best[0]:
                best = (mx, x0 + loc[0], y0 + loc[1], m)
        return best

    def locate(self, grad, thr=0.35, margin=80):
        H, W = grad.shape
        best = None
        if self.last is not None and margin:
            lx, ly, lw, lh = self.last
            region = (max(lx - margin, 0), max(ly - margin, 0),
                      min(lx + lw + margin, W), min(ly + lh + margin, H))
            best = self.search(grad, region)
        if best is None or best[0] < thr:
            best = self.search(grad, (0, 0, W, H))
        if best is None or best[0] < thr:
            return None
        sc, x, y, m = best
        self.last = (x, y, m.shape[1], m.shape[0])
        return x, y, m, sc


def inpaint_at(frame, m, x, y, method="telea", radius=4, pad=12):
    H, W = frame.shape[:2]
    h, w = m.shape
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    local = np.zeros((y1 - y0, x1 - x0), np.uint8)
    local[y - y0:y - y0 + h, x - x0:x - x0 + w] = m
    roi = np.ascontiguousarray(frame[y0:y1, x0:x1])
    frame[y0:y1, x0:x1] = cv2.inpaint(roi, local, radius, INPAINT_FLAGS[method])


def detect_preview(src, ref_frame, box, thr=0.35, scales=(1.0,), text_mask=True, n=6):
    """Run detection on n evenly spaced frames so the threshold can be tuned."""
    tpl, m = build_template(ref_frame, box, text_mask)
    loc = MovingLocator(tpl, m, scales)
    out = []
    for f in sample_frames(src, n):
        H, W = f.shape[:2]
        grad = _grad(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
        best = loc.search(grad, (0, 0, W, H))
        vis, score = f.copy(), 0.0
        if best:
            score, x, y, mm = best
            color = (0, 255, 0) if score >= thr else (0, 0, 255)
            cv2.rectangle(vis, (x, y), (x + mm.shape[1], y + mm.shape[0]), color, 3)
        out.append((vis, float(score)))
    return out


def _mux(tmp_video, src, dst):
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", tmp_video, "-i", src,
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", dst,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_video_dynamic(src, dst, ref_frame, box, method="telea", radius=4, thr=0.35,
                          scales=(1.0,), margin=80, hold=5, text_mask=True, progress=None):
    info = video_info(src)
    tpl, m = build_template(ref_frame, box, text_mask)
    loc = MovingLocator(tpl, m, scales)

    tmp_dir = tempfile.mkdtemp()
    tmp_video = os.path.join(tmp_dir, "silent.mp4")
    cap = cv2.VideoCapture(src)
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             info["fps"], (info["w"], info["h"]))
    i = found = miss = 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        grad = _grad(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        r = loc.locate(grad, thr, margin)
        if r:
            x, y, mm, _ = r
            last, miss = (x, y, mm), 0
            found += 1
            inpaint_at(frame, mm, x, y, method, radius)
        elif last and miss < hold:
            miss += 1
            inpaint_at(frame, last[2], last[0], last[1], method, radius)
        writer.write(frame)
        i += 1
        if progress and info["frames"]:
            progress(min(i / info["frames"], 1.0))
    cap.release()
    writer.release()

    _mux(tmp_video, src, dst)
    try:
        os.remove(tmp_video)
        os.rmdir(tmp_dir)
    except OSError:
        pass
    return {"frames": i, "detected": found}