"""
detect_and_sort.py — Real-time sortasi biji kopi sangrai.

Pipeline per frame:
  1. Capture frame dari Picamera2.
  2. Deteksi kontur biji (find_bean_contours dengan ROI + filter shape).
  3. Untuk setiap kontur, ekstraksi fitur dan klasifikasi (Random Forest).
  4. Deduplikasi via centroid tracker (cegah satu biji dihitung berkali-kali).
  5. Kalau prediksi = 'reject', jadwalkan trigger solenoid setelah delay fisik.
  6. Tampilkan counter accepted/reject dan bounding box.
"""

import os
import pickle
import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

import RPi.GPIO as GPIO

from preprocess import find_bean_contours, extract_features

# --- Konfigurasi GPIO solenoid ---
SOLENOID_PIN = 18                  # BCM
SOLENOID_ON = GPIO.LOW             # active LOW
SOLENOID_OFF = GPIO.HIGH

# --- Timing solenoid (KALIBRASI WAJIB) ---
SOLENOID_DELAY_S = 0.05            # TODO: ukur jarak kamera→solenoid
SOLENOID_PULSE_S = 0.05            # TODO: tune empiris
SOLENOID_MIN_GAP_S = 0.08          # TODO: gap minimum antar pulse

# --- Konfigurasi tracker dedup ---
DEDUP_DISTANCE_PX = 80             # TODO: kalibrasi
DEDUP_TIME_S = 1.0

# --- Konfigurasi kamera ---
FRAME_SIZE = (640, 480)
WINDOW_NAME = "Sortasi Real-Time"

# ROI rectangle (x, y, w, h). HARUS sama dengan capture.py agar feature
# distribution train vs inference konsisten.
ROI_RECT = None                    # TODO: set sama dengan capture.py
SHOW_DEBUG_MASK = False

MODEL_FILE = "defect_model.pkl"


# =====================================================================
# Tracker centroid sederhana untuk deduplikasi biji antar frame
# =====================================================================
class CentroidTracker:
    def __init__(self, max_distance=DEDUP_DISTANCE_PX, ttl=DEDUP_TIME_S):
        self.entries = []
        self.max_distance = max_distance
        self.ttl = ttl

    def is_new(self, cx, cy):
        now = time.time()
        self.entries = [(x, y, t) for (x, y, t) in self.entries
                        if now - t < self.ttl]
        for (x, y, _) in self.entries:
            if (cx - x) ** 2 + (cy - y) ** 2 < self.max_distance ** 2:
                return False
        self.entries.append((cx, cy, now))
        return True


# =====================================================================
# Solenoid controller dengan trigger terjadwal
# =====================================================================
class SolenoidController:
    def __init__(self, pin, delay_s, pulse_s, min_gap_s):
        self.pin = pin
        self.delay_s = delay_s
        self.pulse_s = pulse_s
        self.min_gap_s = min_gap_s
        self._lock = threading.Lock()
        self._last_fire_end = 0.0

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, SOLENOID_OFF)

    def schedule_fire(self):
        timer = threading.Timer(self.delay_s, self._fire)
        timer.daemon = True
        timer.start()

    def _fire(self):
        with self._lock:
            now = time.time()
            wait = self._last_fire_end + self.min_gap_s - now
            if wait > 0:
                time.sleep(wait)
            GPIO.output(self.pin, SOLENOID_ON)
            time.sleep(self.pulse_s)
            GPIO.output(self.pin, SOLENOID_OFF)
            self._last_fire_end = time.time()

    def cleanup(self):
        GPIO.output(self.pin, SOLENOID_OFF)
        GPIO.cleanup()


def build_roi_mask(frame_hw, rect):
    if rect is None:
        return None
    h, w = frame_hw
    m = np.zeros((h, w), dtype=np.uint8)
    x, y, rw, rh = rect
    m[y:y + rh, x:x + rw] = 255
    return m


# =====================================================================
# Main loop
# =====================================================================
def main():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] {MODEL_FILE} tidak ditemukan. Jalankan train.py dulu.")
        return

    with open(MODEL_FILE, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    class_names = bundle["class_names"]
    reject_label = class_names.index("reject")
    print(f"[INFO] Model dimuat. Kelas: {class_names}")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": FRAME_SIZE, "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    roi_mask = build_roi_mask((FRAME_SIZE[1], FRAME_SIZE[0]), ROI_RECT)

    solenoid = SolenoidController(
        pin=SOLENOID_PIN,
        delay_s=SOLENOID_DELAY_S,
        pulse_s=SOLENOID_PULSE_S,
        min_gap_s=SOLENOID_MIN_GAP_S,
    )
    tracker = CentroidTracker()

    counter = {"accepted": 0, "reject": 0}
    frame_count = 0
    t_start = time.time()

    print("[INFO] Mulai. Tekan 'q' untuk keluar.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            contours, mask = find_bean_contours(frame_bgr, roi_mask=roi_mask)
            preview = frame_bgr.copy()

            for c in contours:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(c)

                # Skip kalau biji sudah pernah diproses
                if not tracker.is_new(cx, cy):
                    cv2.rectangle(preview, (x, y), (x + w, y + h),
                                  (128, 128, 128), 1)
                    continue

                # Klasifikasi
                feat = extract_features(frame_bgr, c).reshape(1, -1)
                pred = int(model.predict(feat)[0])
                label = class_names[pred]
                counter[label] += 1

                color = (0, 0, 255) if pred == reject_label else (0, 255, 0)
                cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
                cv2.putText(preview, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if pred == reject_label:
                    solenoid.schedule_fire()
                    print(f"[REJECT] cx={cx} cy={cy} → solenoid scheduled "
                          f"(+{SOLENOID_DELAY_S*1000:.0f}ms)")

            # Overlay HUD + ROI
            if ROI_RECT is not None:
                rx, ry, rw, rh = ROI_RECT
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh),
                              (255, 0, 0), 1)

            frame_count += 1
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            hud = (f"Accepted: {counter['accepted']}  "
                   f"Reject: {counter['reject']}  "
                   f"FPS: {fps:.1f}")
            cv2.putText(preview, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, preview)
            if SHOW_DEBUG_MASK:
                cv2.imshow("Mask (debug)", mask)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        solenoid.cleanup()
        print(f"\n[DONE] Accepted={counter['accepted']}, "
              f"Reject={counter['reject']}, "
              f"Frame={frame_count}, "
              f"Avg FPS={frame_count/max(elapsed, 1e-6):.2f}")


if __name__ == "__main__":
    main()
  
