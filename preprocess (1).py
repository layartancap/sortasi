"""
preprocess.py — Deteksi kontur biji dan ekstraksi fitur (HSV histogram + GLCM).

Modul ini punya dua peran:
  1. Library: menyediakan fungsi `find_bean_contours()` dan `extract_features()`
     yang dipakai oleh `capture.py` dan `detect_and_sort.py`.
  2. Script: saat dijalankan langsung, memproses seluruh dataset di folder
     `dataset/accepted/` dan `dataset/reject/`, lalu menyimpan vektor fitur
     ke `data.npy` dan label ke `labels.npy`.

Filter kontur (anti false positive):
  - Area     : tolak kontur terlalu kecil (noise) atau besar (tumpukan/talang).
  - Aspect   : biji ≈ oval, tolak bentuk panjang ekstrem (kabel, edge PCB).
  - Solidity : biji = bentuk cembung; tolak bentuk irregular.
"""

import os
import sys

import cv2
import numpy as np

# Kompatibilitas nama API skimage (graycomatrix vs greycomatrix lama)
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import (
        greycomatrix as graycomatrix,
        greycoprops as graycoprops,
    )

# --- Konfigurasi deteksi kontur ---
MIN_CONTOUR_AREA = 500       # TODO: kalibrasi sesuai ukuran biji di frame
MAX_CONTOUR_AREA = 50000     # TODO: kalibrasi
GAUSSIAN_KERNEL = (5, 5)
MORPH_KERNEL = np.ones((3, 3), np.uint8)

# --- Filter bentuk (anti false positive) ---
MIN_ASPECT_RATIO = 0.4
MAX_ASPECT_RATIO = 2.5
MIN_SOLIDITY = 0.85

# --- Konfigurasi fitur ---
HSV_BINS = (8, 8, 8)
GLCM_LEVELS = 32
GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "homogeneity", "energy", "correlation",
              "dissimilarity", "ASM"]

DATASET_ROOT = "dataset"
CLASS_NAMES = ["accepted", "reject"]
DATA_OUT = "data.npy"
LABELS_OUT = "labels.npy"


def find_bean_contours(frame_bgr,
                       min_area=MIN_CONTOUR_AREA,
                       max_area=MAX_CONTOUR_AREA,
                       roi_mask=None):
    """
    Deteksi kontur biji dengan filter area + bentuk + ROI opsional.

    Parameter:
        roi_mask: numpy uint8 array (H, W), 255 di area aktif, 0 diabaikan.
                  None = pakai seluruh frame.

    Return:
        contours: list kontur valid.
        mask    : citra biner threshold (untuk debug).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, GAUSSIAN_KERNEL, 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        aspect = w / h
        if aspect < MIN_ASPECT_RATIO or aspect > MAX_ASPECT_RATIO:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < MIN_SOLIDITY:
            continue

        valid.append(c)
    return valid, mask


def extract_features(frame_bgr, contour):
    """Ekstraksi vektor fitur dari satu biji. Return array 1D shape=(72,)."""
    full_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(full_mask, [contour], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(contour)
    roi_bgr = frame_bgr[y:y + h, x:x + w]
    roi_mask = full_mask[y:y + h, x:x + w]

    # HSV histogram (dengan mask)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], roi_mask, [HSV_BINS[0]], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], roi_mask, [HSV_BINS[1]], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], roi_mask, [HSV_BINS[2]], [0, 256]).flatten()
    color_feat = np.concatenate([h_hist, s_hist, v_hist])
    color_feat = color_feat / (color_feat.sum() + 1e-7)

    # GLCM texture
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray_q = (gray_roi // (256 // GLCM_LEVELS)).astype(np.uint8)
    gray_q = np.clip(gray_q, 0, GLCM_LEVELS - 1)

    glcm = graycomatrix(gray_q,
                        distances=GLCM_DISTANCES,
                        angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS,
                        symmetric=True,
                        normed=True)

    glcm_feat = []
    for prop in GLCM_PROPS:
        vals = graycoprops(glcm, prop).flatten()
        glcm_feat.extend(vals)
    glcm_feat = np.array(glcm_feat, dtype=np.float64)

    return np.concatenate([color_feat, glcm_feat]).astype(np.float32)


def process_dataset():
    """Iterasi folder dataset, ekstrak fitur, simpan ke .npy."""
    features_list = []
    labels_list = []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Folder {class_dir} tidak ditemukan, dilewati.")
            continue

        files = sorted(f for f in os.listdir(class_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        print(f"[INFO] Memproses {len(files)} file di kelas '{class_name}'...")

        n_extracted = 0
        n_skipped = 0
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"  [SKIP] Gagal membaca {fpath}")
                n_skipped += 1
                continue

            contours, _ = find_bean_contours(img)
            if not contours:
                print(f"  [SKIP] Tidak ada kontur valid di {fname}")
                n_skipped += 1
                continue

            for c in contours:
                feat = extract_features(img, c)
                features_list.append(feat)
                labels_list.append(label_idx)
                n_extracted += 1

        print(f"  → {n_extracted} sampel diekstrak, {n_skipped} file dilewati.")

    if not features_list:
        print("[ERROR] Tidak ada fitur yang berhasil diekstrak. Cek dataset.")
        sys.exit(1)

    data = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    np.save(DATA_OUT, data)
    np.save(LABELS_OUT, labels)

    print(f"\n[DONE] Tersimpan: {DATA_OUT} shape={data.shape}, "
          f"{LABELS_OUT} shape={labels.shape}")
    print(f"       Distribusi label:")
    for idx, name in enumerate(CLASS_NAMES):
        n = int((labels == idx).sum())
        print(f"         {name} (label={idx}): {n} sampel")


if __name__ == "__main__":
    process_dataset()
