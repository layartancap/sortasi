"""
detect_and_sort.py — Real-time sortasi biji kopi sangrai (v2).

PERUBAHAN UTAMA v2:
  1. Bisa ganti mode deteksi on-the-fly tanpa restart:
       Tekan 'M' → siklus: auto → grayscale → hsv → lab → adaptive → auto
  2. Anti-inversi: mask tidak lagi terbalik saat background terang.
  3. Tampilan debug mask ada di pojok kanan bawah frame (tidak perlu window terpisah).
  4. Confidence threshold: prediksi rendah tidak langsung ditrigger solenoid.
  5. HUD lebih informatif: mode deteksi aktif, confidence, total frame.
  6. Kontrol keyboard:
       'q' = keluar
       'm' = ganti mode deteksi
       'd' = toggle debug mask overlay
       'r' = reset counter
       's' = simpan screenshot
"""

import os
import pickle
import threading
import time

import cv2
import numpy as np

# Coba import GPIO (hanya ada di Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False
    print("[WARN] RPi.GPIO tidak tersedia — solenoid dinonaktifkan (mode simulasi).")

# Coba import Picamera2 (hanya ada di Raspberry Pi)
try:
    from picamera2 import Picamera2
    _PICAM_AVAILABLE = True
except ImportError:
    _PICAM_AVAILABLE = False
    print("[WARN] Picamera2 tidak tersedia — menggunakan webcam USB (cv2.VideoCapture).")

from preprocess import find_bean_contours, extract_features, CONFIG as PREP_CONFIG

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────

# GPIO solenoid
SOLENOID_PIN     = 18
SOLENOID_ON      = 0   # active LOW
SOLENOID_OFF     = 1
SOLENOID_DELAY_S = 0.05    # TODO: ukur jarak kamera→solenoid (detik)
SOLENOID_PULSE_S = 0.05    # TODO: tune
SOLENOID_MIN_GAP = 0.08    # minimum gap antar pulse

# Centroid dedup
DEDUP_DIST_PX  = 80
DEDUP_TIME_S   = 1.2

# Confidence threshold — prediksi di bawah ini diabaikan / ditampilkan beda warna
CONFIDENCE_THRESHOLD = 0.60

# Kamera
FRAME_SIZE   = (640, 480)
CAMERA_INDEX = 0           # untuk webcam USB

# ROI — set setelah hardware final. Format: (x, y, w, h) atau None
ROI_RECT = None            # TODO: contoh (100, 50, 440, 380)

# Model
MODEL_FILE = "defect_model.pkl"

# Mode deteksi yang bisa disiklus dengan tombol 'M'
DETECTION_MODES = ["auto", "grayscale", "hsv", "lab", "adaptive"]


# ─────────────────────────────────────────────────────────────
# CENTROID TRACKER
# ─────────────────────────────────────────────────────────────

class CentroidTracker:
    def __init__(self, max_dist=DEDUP_DIST_PX, ttl=DEDUP_TIME_S):
        self.entries  = []
        self.max_dist = max_dist
        self.ttl      = ttl

    def is_new(self, cx, cy):
        now = time.time()
        self.entries = [(x, y, t) for (x, y, t) in self.entries if now - t < self.ttl]
        for (x, y, _) in self.entries:
            if (cx - x) ** 2 + (cy - y) ** 2 < self.max_dist ** 2:
                return False
        self.entries.append((cx, cy, now))
        return True


# ─────────────────────────────────────────────────────────────
# SOLENOID CONTROLLER
# ─────────────────────────────────────────────────────────────

class SolenoidController:
    def __init__(self):
        self._lock          = threading.Lock()
        self._last_fire_end = 0.0
        self._active        = _GPIO_AVAILABLE

        if self._active:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(SOLENOID_PIN, GPIO.OUT)
            GPIO.output(SOLENOID_PIN, SOLENOID_OFF)

    def schedule_fire(self):
        t = threading.Timer(SOLENOID_DELAY_S, self._fire)
        t.daemon = True
        t.start()

    def _fire(self):
        if not self._active:
            print(f"  [SIM] Solenoid FIRE (simulasi, GPIO tidak tersedia)")
            return
        with self._lock:
            wait = self._last_fire_end + SOLENOID_MIN_GAP - time.time()
            if wait > 0:
                time.sleep(wait)
            GPIO.output(SOLENOID_PIN, SOLENOID_ON)
            time.sleep(SOLENOID_PULSE_S)
            GPIO.output(SOLENOID_PIN, SOLENOID_OFF)
            self._last_fire_end = time.time()

    def cleanup(self):
        if self._active:
            GPIO.output(SOLENOID_PIN, SOLENOID_OFF)
            GPIO.cleanup()


# ─────────────────────────────────────────────────────────────
# KAMERA ABSTRAKSI (Picamera2 atau Webcam)
# ─────────────────────────────────────────────────────────────

class Camera:
    def __init__(self):
        if _PICAM_AVAILABLE:
            self._picam = Picamera2()
            cfg = self._picam.create_preview_configuration(
                main={"size": FRAME_SIZE, "format": "RGB888"}
            )
            self._picam.configure(cfg)
            self._picam.start()
            time.sleep(1.0)
            self._mode = "picam"
        else:
            self._cap  = cv2.VideoCapture(CAMERA_INDEX)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
            self._mode = "cv2"

    def read(self):
        """Return frame BGR atau None jika gagal."""
        if self._mode == "picam":
            rgb = self._picam.capture_array()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self._cap.read()
            return frame if ret else None

    def release(self):
        if self._mode == "picam":
            self._picam.stop()
        else:
            self._cap.release()


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def build_roi_mask(frame_hw, rect):
    if rect is None:
        return None
    h, w = frame_hw
    m = np.zeros((h, w), dtype=np.uint8)
    x, y, rw, rh = rect
    m[y:y + rh, x:x + rw] = 255
    return m


def overlay_mask(frame, mask, alpha=0.35, color=(0, 200, 255)):
    """Overlay mask tipis di atas frame (visualisasi debug)."""
    colored = np.zeros_like(frame)
    colored[mask > 0] = color
    cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0, frame)


def draw_hud(frame, counter, fps, mode, det_mode, show_debug):
    h, w = frame.shape[:2]
    bar = np.zeros((48, w, 3), dtype=np.uint8)
    info = (f"Accept:{counter['accepted']}  Reject:{counter['reject']}  "
            f"FPS:{fps:.1f}  Det:{det_mode}  "
            f"Mask:{'ON' if show_debug else 'off'}")
    cv2.putText(bar, info, (8, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)
    frame[:48, :] = bar

    # Panduan keyboard
    legend = "M=mode  D=mask  R=reset  S=screenshot  Q=quit"
    cv2.putText(frame, legend, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] {MODEL_FILE} tidak ditemukan. Jalankan train.py dulu.")
        return

    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)
    model        = bundle["model"]
    class_names  = bundle["class_names"]
    reject_label = class_names.index("reject")
    print(f"[INFO] Model dimuat. Kelas: {class_names}")

    cam      = Camera()
    solenoid = SolenoidController()
    tracker  = CentroidTracker()
    roi_mask = build_roi_mask((FRAME_SIZE[1], FRAME_SIZE[0]), ROI_RECT)

    counter     = {"accepted": 0, "reject": 0}
    mode_idx    = 0   # indeks di DETECTION_MODES
    show_debug  = True
    frame_count = 0
    screenshot  = 0
    t_start     = time.time()
    fps_buf     = []

    WINDOW = "Sortasi Real-Time v2"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    print("[INFO] Mulai. Kontrol: M=ganti mode  D=toggle mask  R=reset  S=screenshot  Q=quit")

    try:
        while True:
            t0 = time.time()
            frame = cam.read()
            if frame is None:
                print("[ERROR] Gagal baca frame.")
                break

            det_mode = DETECTION_MODES[mode_idx]
            PREP_CONFIG["detection_mode"] = det_mode

            contours, mask, _ = find_bean_contours(frame, roi_mask=roi_mask)
            preview = frame.copy()

            # Overlay mask debug
            if show_debug:
                overlay_mask(preview, mask)

            # ── Proses setiap kontur ───────────────────────────────
            for c in contours:
                M_cnt = cv2.moments(c)
                if M_cnt["m00"] == 0:
                    continue
                cx = int(M_cnt["m10"] / M_cnt["m00"])
                cy = int(M_cnt["m01"] / M_cnt["m00"])
                x, y, w, h = cv2.boundingRect(c)

                # Kontur sudah diproses sebelumnya → abu-abu
                if not tracker.is_new(cx, cy):
                    cv2.rectangle(preview, (x, y), (x+w, y+h), (100, 100, 100), 1)
                    continue

                feat = extract_features(frame, c)
                if feat is None:
                    continue

                feat_2d  = feat.reshape(1, -1)
                pred     = int(model.predict(feat_2d)[0])
                prob     = float(model.predict_proba(feat_2d)[0].max())
                label    = class_names[pred]
                low_conf = prob < CONFIDENCE_THRESHOLD

                if low_conf:
                    color      = (0, 165, 255)   # oranye = tidak yakin
                    box_thick  = 1
                elif pred == reject_label:
                    color     = (0, 0, 255)       # merah = reject
                    box_thick = 2
                else:
                    color     = (0, 255, 0)       # hijau = accept
                    box_thick = 2

                cv2.rectangle(preview, (x, y), (x+w, y+h), color, box_thick)
                cv2.drawContours(preview, [c], -1, color, 1)
                tag = f"{label} {prob:.0%}"
                cv2.putText(preview, tag, (x, max(y-6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                if not low_conf:
                    counter[label] += 1
                    if pred == reject_label:
                        solenoid.schedule_fire()
                        print(f"[REJECT] cx={cx} cy={cy} conf={prob:.0%} "
                              f"→ solenoid +{SOLENOID_DELAY_S*1000:.0f}ms")

            # ── ROI border ────────────────────────────────────────
            if ROI_RECT is not None:
                rx, ry, rw, rh = ROI_RECT
                cv2.rectangle(preview, (rx, ry), (rx+rw, ry+rh), (255, 200, 0), 1)

            # ── FPS hitung ────────────────────────────────────────
            frame_count += 1
            dt = time.time() - t0
            fps_buf.append(1.0 / max(dt, 1e-5))
            if len(fps_buf) > 20:
                fps_buf.pop(0)
            fps = float(np.mean(fps_buf))

            draw_hud(preview, counter, fps, mode_idx, det_mode, show_debug)
            cv2.imshow(WINDOW, preview)

            # ── Keyboard ──────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mode_idx = (mode_idx + 1) % len(DETECTION_MODES)
                print(f"[MODE] Deteksi → {DETECTION_MODES[mode_idx]}")
            elif key == ord('d'):
                show_debug = not show_debug
                print(f"[DEBUG] Mask overlay {'ON' if show_debug else 'OFF'}")
            elif key == ord('r'):
                counter = {"accepted": 0, "reject": 0}
                print("[RESET] Counter direset.")
            elif key == ord('s'):
                fname = f"screenshot_{screenshot:04d}.jpg"
                cv2.imwrite(fname, preview)
                screenshot += 1
                print(f"[SCREENSHOT] Disimpan: {fname}")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        solenoid.cleanup()
        elapsed = max(time.time() - t_start, 1e-6)
        print(f"\n[DONE] Accept={counter['accepted']}  Reject={counter['reject']}  "
              f"Frame={frame_count}  Avg FPS={frame_count/elapsed:.2f}")


if __name__ == "__main__":
    main()
