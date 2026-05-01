"""
capture.py — Smart capture untuk pengumpulan dataset biji kopi sangrai.
Mode kerja:
  - Pilih kelas target di awal (accepted / reject).
  - Auto-detect kemunculan biji via contour detection.
  - Frame penuh disimpan ke folder dataset sesuai kelas (rising-edge trigger).
  - Cooldown waktu mencegah duplikasi penyimpanan untuk biji yang sama.
Kontrol keyboard saat preview:
  - 'q'    : keluar.
  - SPASI  : capture manual (override auto-detect).
"""
import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
from picamera2 import Picamera2
from preprocess import find_bean_contours

# --- Konfigurasi ---
DATASET_ROOT = "dataset"
FRAME_SIZE = (640, 480)
CAPTURE_COOLDOWN_S = 0.4
PREVIEW_WINDOW = "Capture Preview"
ROI_RECT = None
SHOW_DEBUG_MASK = True

def select_class():
    choice = input("Capture untuk kelas mana? [a]ccepted / [r]eject: ").strip().lower()
    if choice in ("a", "accepted"):
        return "accepted"
    if choice in ("r", "reject"):
        return "reject"
    print("Pilihan tidak valid. Gunakan 'a' atau 'r'.")
    sys.exit(1)

def build_roi_mask(frame_hw, rect):
    if rect is None:
        return None
    h, w = frame_hw
    m = np.zeros((h, w), dtype=np.uint8)
    x, y, rw, rh = rect
    m[y:y + rh, x:x + rw] = 255
    return m

def make_filename(class_name, idx):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{class_name}_{ts}_{idx:04d}.jpg"

def main():
    class_name = select_class()
    save_dir = os.path.join(DATASET_ROOT, class_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Menyimpan ke: {save_dir}")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": FRAME_SIZE, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    roi_mask = build_roi_mask((FRAME_SIZE[1], FRAME_SIZE[0]), ROI_RECT)
    existing = [f for f in os.listdir(save_dir) if f.endswith((".jpg", ".png"))]
    counter = len(existing)
    print(f"[INFO] {counter} sampel sudah ada. Index lanjut dari sini.")
    prev_has_bean = False
    last_capture_t = 0.0
    print("[INFO] Streaming. 'q' = keluar, SPASI = capture manual.")
    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if roi_mask is not None:
                frame_processed = cv2.bitwise_and(frame_bgr, frame_bgr, mask=roi_mask)
            else:
                frame_processed = frame_bgr
            contours, mask = find_bean_contours(frame_processed)
            has_bean = len(contours) > 0
            now = time.time()
            preview = frame_bgr.copy()
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if ROI_RECT is not None:
                rx, ry, rw, rh = ROI_RECT
                cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
            info = f"Kelas: {class_name} | Tersimpan: {counter} | Biji terdeteksi: {len(contours)}"
            cv2.putText(preview, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(PREVIEW_WINDOW, preview)
            if SHOW_DEBUG_MASK:
                cv2.imshow("Mask (debug)", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            manual_trigger = (key == ord(" "))
            auto_trigger = (has_bean and not prev_has_bean and (now - last_capture_t) > CAPTURE_COOLDOWN_S)
            if manual_trigger or auto_trigger:
                fname = make_filename(class_name, counter)
                fpath = os.path.join(save_dir, fname)
                cv2.imwrite(fpath, frame_bgr)
                counter += 1
                last_capture_t = now
                trigger_type = "MANUAL" if manual_trigger else "AUTO"
                print(f"[{trigger_type}] {fpath}")
            prev_has_bean = has_bean
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print(f"[DONE] Total {counter} sampel di {save_dir}")

if __name__ == "__main__":
    main()
