"""
preprocess.py — Deteksi kontur biji dan ekstraksi fitur (multi-mode + anti-inversi).

PERUBAHAN UTAMA v2:
  1. Multi-mode detection: Grayscale Otsu, HSV masking, LAB masking, Adaptive.
     Semua mode digabung (union) agar biji tidak terlewat.
  2. Anti-inversi otomatis: mask dianalisa, jika biji gelap di background terang
     (kasus umum biji kopi sangrai) mask di-invert secara otomatis.
  3. Dataset: 1 file gambar → 1 sampel fitur terbaik (kontur terbesar/terbaik),
     mencegah 1 foto menghasilkan 300+ sampel ganda.
  4. Fitur diperkaya: HSV hist + LAB stats + GLCM + shape features.
  5. ROI mudah dikonfigurasi.

MODE DETEKSI (pilih lewat DETECTION_MODE di CONFIG):
  "auto"      — coba semua, ambil gabungan terbaik (default)
  "grayscale" — hanya Otsu grayscale
  "hsv"       — hanya HSV range masking
  "lab"       — hanya LAB range masking
  "adaptive"  — hanya adaptive threshold
"""

import os
import sys

import cv2
import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import (
        greycomatrix as graycomatrix,
        greycoprops as graycoprops,
    )

# ─────────────────────────────────────────────────────────────
# KONFIGURASI UTAMA — sesuaikan dengan kondisi fisik kamu
# ─────────────────────────────────────────────────────────────
CONFIG = {
    # Mode deteksi: "auto" | "grayscale" | "hsv" | "lab" | "adaptive"
    "detection_mode": "auto",

    # Filter ukuran kontur (px²) — sesuaikan ukuran biji di frame
    "min_area": 500,
    "max_area": 50000,

    # Filter bentuk
    "min_aspect": 0.35,
    "max_aspect": 2.8,
    "min_solidity": 0.75,   # diturunkan dari 0.85 agar biji tidak terpotong

    # HSV range untuk biji kopi sangrai (cokelat gelap)
    # Gunakan kalibrasi_warna.py untuk mencari nilai yang tepat
    "hsv_lower": np.array([5,  30,  20]),
    "hsv_upper": np.array([40, 220, 180]),

    # LAB range — lebih stabil terhadap perubahan cahaya
    "lab_lower": np.array([10,  125, 130]),
    "lab_upper": np.array([180, 148, 155]),

    # Dataset: jika True, ambil hanya 1 kontur terbaik per gambar
    # Jika False, ambil semua kontur valid (mode lama)
    "one_sample_per_image": True,

    # Morfologi
    "morph_kernel_size": 5,
    "morph_open_iter":   1,
    "morph_close_iter":  2,

    # Dataset path
    "dataset_root": "dataset",
    "class_names":  ["accepted", "reject"],
    "data_out":     "data.npy",
    "labels_out":   "labels.npy",
}

GLCM_LEVELS     = 32
GLCM_DISTANCES  = [1, 2]
GLCM_ANGLES     = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS      = ["contrast", "homogeneity", "energy", "correlation",
                   "dissimilarity", "ASM"]
HSV_BINS        = (8, 8, 8)

# ─────────────────────────────────────────────────────────────
# WHITE_RATIO_THRESHOLD — ambang batas deteksi inversi mask
# ─────────────────────────────────────────────────────────────
# Definisi:
#   white_ratio = jumlah piksel putih (255) di mask
#                 ÷ total piksel frame
#
# Logika anti-inversi:
#   Biji kopi sangrai berwarna gelap, background (talang/konveyor) terang.
#   Setelah Otsu threshold dengan THRESH_BINARY_INV, idealnya:
#     - biji kopi  → putih (foreground)
#     - background → hitam
#   Artinya white_ratio seharusnya KECIL (hanya biji yang putih).
#
#   Jika white_ratio > WHITE_RATIO_THRESHOLD (misal 0.60 = 60% frame putih),
#   berarti yang ter-threshold justru background → mask TERBALIK.
#   Solusi: flip mask dengan bitwise_not.
#
# Cara kalibrasi nilai ini:
#   - Jalankan program, lihat window mask debug.
#   - Jika mask tampak terbalik (biji hitam, background putih):
#     turunkan WHITE_RATIO_THRESHOLD (misal 0.50).
#   - Jika biji selalu hilang dari mask:
#     naikkan WHITE_RATIO_THRESHOLD (misal 0.70).
#   - Rentang aman: 0.50 – 0.75.
WHITE_RATIO_THRESHOLD = 0.60


# ─────────────────────────────────────────────────────────────
# MASK BUILDERS
# ─────────────────────────────────────────────────────────────

def _mask_grayscale_otsu(frame_bgr):
    """
    Otsu threshold + anti-inversi otomatis.
    Biji kopi sangrai lebih gelap dari background → setelah threshold,
    biji harusnya putih. Jika mayoritas piksel putih, kita flip mask.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Anti-inversi otomatis ──────────────────────────────────────────
    # white_ratio = proporsi piksel putih di mask (0.0 – 1.0).
    # Contoh: white_ratio = 0.08  → 8% frame putih  → wajar (hanya biji)
    #         white_ratio = 0.75  → 75% frame putih → mask terbalik!
    #
    # np.count_nonzero(mask) : hitung piksel bernilai 255 (putih)
    # mask.size              : total piksel = tinggi × lebar
    total_pixels = mask.size                          # H × W
    white_pixels = np.count_nonzero(mask)             # piksel bernilai 255
    white_ratio  = white_pixels / total_pixels        # proporsi 0.0–1.0

    if white_ratio > WHITE_RATIO_THRESHOLD:
        # Lebih dari 60% frame putih → background yang ter-threshold,
        # bukan biji kopi. Balik mask supaya biji jadi putih.
        mask = cv2.bitwise_not(mask)

    return mask


def _mask_adaptive(frame_bgr):
    """Adaptive threshold untuk pencahayaan tidak merata."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=4
    )
    return mask


def _mask_hsv(frame_bgr):
    """HSV range masking — ubah CONFIG['hsv_lower/upper'] via kalibrasi."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CONFIG["hsv_lower"], CONFIG["hsv_upper"])
    return mask


def _mask_lab(frame_bgr):
    """LAB range masking — lebih stabil terhadap perubahan pencahayaan."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, CONFIG["lab_lower"], CONFIG["lab_upper"])
    return mask


def _apply_morph(mask):
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CONFIG["morph_kernel_size"], CONFIG["morph_kernel_size"])
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=CONFIG["morph_open_iter"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=CONFIG["morph_close_iter"])
    return mask


def _build_mask(frame_bgr, mode="auto"):
    """Buat mask sesuai mode. Mode 'auto' = gabungan semua."""
    if mode == "grayscale":
        mask = _mask_grayscale_otsu(frame_bgr)
    elif mode == "hsv":
        mask = _mask_hsv(frame_bgr)
    elif mode == "lab":
        mask = _mask_lab(frame_bgr)
    elif mode == "adaptive":
        mask = _mask_adaptive(frame_bgr)
    else:  # "auto" — gabungan
        m_gray = _mask_grayscale_otsu(frame_bgr)
        m_hsv  = _mask_hsv(frame_bgr)
        m_lab  = _mask_lab(frame_bgr)
        m_adap = _mask_adaptive(frame_bgr)
        # Union semua mode, lalu ambil yang paling "bersih"
        mask = cv2.bitwise_or(m_gray, m_hsv)
        mask = cv2.bitwise_or(mask, m_lab)
        # Adaptive lebih noisy, masukkan hanya jika hasil lain kosong
        if cv2.countNonZero(mask) < 50:
            mask = m_adap

    return _apply_morph(mask)


# ─────────────────────────────────────────────────────────────
# DETEKSI KONTUR UTAMA
# ─────────────────────────────────────────────────────────────

def find_bean_contours(frame_bgr, roi_mask=None, mode=None):
    """
    Deteksi kontur biji dengan multi-mode + filter bentuk + ROI.

    Parameter:
        roi_mask : np.uint8 (H, W) — 255 di area aktif, 0 di luar. None = full frame.
        mode     : override CONFIG['detection_mode'] jika diisi.

    Return:
        contours : list kontur valid (sudah difilter).
        mask     : mask biner gabungan (untuk debug).
        mode_used: nama mode yang dipakai.
    """
    det_mode = mode or CONFIG["detection_mode"]
    mask = _build_mask(frame_bgr, det_mode)

    if roi_mask is not None:
        mask = cv2.bitwise_and(mask, roi_mask)

    contours_raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for c in contours_raw:
        area = cv2.contourArea(c)
        if not (CONFIG["min_area"] < area < CONFIG["max_area"]):
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        aspect = w / h
        if aspect < CONFIG["min_aspect"] or aspect > CONFIG["max_aspect"]:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < CONFIG["min_solidity"]:
            continue

        valid.append(c)

    # Urutkan dari terbesar (berguna untuk one_sample_per_image)
    valid.sort(key=cv2.contourArea, reverse=True)
    return valid, mask, det_mode


# ─────────────────────────────────────────────────────────────
# EKSTRAKSI FITUR — 24 HSV hist + 6 LAB stats + 48 GLCM + 4 shape = 82 dim
# ─────────────────────────────────────────────────────────────

def extract_features(frame_bgr, contour):
    """
    Ekstrak vektor fitur dari satu kontur biji.
    Return: np.float32 array 1D.
    """
    full_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(full_mask, [contour], -1, 255, thickness=cv2.FILLED)

    x, y, w, h = cv2.boundingRect(contour)
    roi_bgr  = frame_bgr[y:y + h, x:x + w]
    roi_mask = full_mask[y:y + h, x:x + w]

    if roi_bgr.size == 0 or roi_mask.size == 0:
        return None

    # ── 1. HSV histogram (8 bin × 3 channel = 24 dim) ───────────────
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], roi_mask, [HSV_BINS[0]], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], roi_mask, [HSV_BINS[1]], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], roi_mask, [HSV_BINS[2]], [0, 256]).flatten()
    color_hsv = np.concatenate([h_hist, s_hist, v_hist])
    color_hsv /= (color_hsv.sum() + 1e-7)

    # ── 2. LAB stats (mean + std × 3 channel = 6 dim) ────────────────
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    lab_stats = []
    for ch_idx in range(3):
        ch = lab[:, :, ch_idx]
        px = ch[roi_mask > 0].astype(np.float32)
        if len(px) == 0:
            lab_stats += [0.0, 0.0]
        else:
            lab_stats += [float(np.mean(px)), float(np.std(px))]
    lab_feat = np.array(lab_stats, dtype=np.float32)

    # ── 3. Grayscale stats (4 dim) ───────────────────────────────────
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    px_gray = gray_roi[roi_mask > 0].astype(np.float32)
    if len(px_gray) == 0:
        gray_stats = np.zeros(4, dtype=np.float32)
    else:
        gray_stats = np.array([
            np.mean(px_gray),
            np.std(px_gray),
            float(np.percentile(px_gray, 25)),
            float(np.percentile(px_gray, 75)),
        ], dtype=np.float32)

    # ── 4. GLCM texture (48 dim) ─────────────────────────────────────
    gray_q = (gray_roi // (256 // GLCM_LEVELS)).astype(np.uint8)
    gray_q = np.clip(gray_q, 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(gray_q,
                        distances=GLCM_DISTANCES,
                        angles=GLCM_ANGLES,
                        levels=GLCM_LEVELS,
                        symmetric=True,
                        normed=True)
    glcm_vals = []
    for prop in GLCM_PROPS:
        glcm_vals.extend(graycoprops(glcm, prop).flatten())
    glcm_feat = np.array(glcm_vals, dtype=np.float32)

    # ── 5. Shape features (5 dim) ────────────────────────────────────
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-7)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / (hull_area + 1e-7)
    aspect = w / (h + 1e-7)
    extent = area / ((w * h) + 1e-7)
    shape_feat = np.array([circularity, solidity, aspect, extent,
                           area / 10000.0],  # normalized
                          dtype=np.float32)

    return np.concatenate([color_hsv, lab_feat, gray_stats, glcm_feat, shape_feat])


# ─────────────────────────────────────────────────────────────
# PROSES DATASET
# ─────────────────────────────────────────────────────────────

def process_dataset():
    """
    Iterasi folder dataset, ekstrak fitur, simpan ke .npy.

    Jika CONFIG['one_sample_per_image'] = True:
      → Ambil hanya 1 kontur terbesar per gambar (= 1 sampel per foto).
      Ini mencegah 1 foto menghasilkan ratusan sampel ganda.

    Jika False (mode lama):
      → Ambil semua kontur valid.
    """
    features_list = []
    labels_list = []
    one_per_img = CONFIG["one_sample_per_image"]

    for label_idx, class_name in enumerate(CONFIG["class_names"]):
        class_dir = os.path.join(CONFIG["dataset_root"], class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Folder {class_dir} tidak ditemukan, dilewati.")
            continue

        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"\n[INFO] Kelas '{class_name}': {len(files)} foto ditemukan.")

        n_ok = 0
        n_skip = 0
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                print(f"  [SKIP] Gagal baca: {fname}")
                n_skip += 1
                continue

            contours, mask, mode_used = find_bean_contours(img)

            if not contours:
                # Coba mode lain sebagai fallback
                for fallback in ["hsv", "lab", "adaptive", "grayscale"]:
                    if fallback == CONFIG["detection_mode"]:
                        continue
                    contours, mask, mode_used = find_bean_contours(img, mode=fallback)
                    if contours:
                        break

            if not contours:
                print(f"  [SKIP] Tidak ada kontur: {fname}")
                n_skip += 1
                continue

            # Pilih kontur yang akan dijadikan sampel
            if one_per_img:
                # Hanya kontur terbesar
                selected = [contours[0]]
            else:
                selected = contours

            for c in selected:
                feat = extract_features(img, c)
                if feat is None:
                    continue
                features_list.append(feat)
                labels_list.append(label_idx)
                n_ok += 1

        mode_str = "(1 sampel/foto)" if one_per_img else "(semua kontur)"
        print(f"  → {n_ok} sampel diekstrak {mode_str}, {n_skip} foto dilewati.")

    if not features_list:
        print("[ERROR] Tidak ada fitur berhasil diekstrak. Cek dataset dan konfigurasi.")
        sys.exit(1)

    data   = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list,   dtype=np.int64)

    np.save(CONFIG["data_out"],   data)
    np.save(CONFIG["labels_out"], labels)

    print(f"\n[DONE] Tersimpan:")
    print(f"  {CONFIG['data_out']}   shape={data.shape}")
    print(f"  {CONFIG['labels_out']} shape={labels.shape}")
    print(f"\nDistribusi:")
    for idx, name in enumerate(CONFIG["class_names"]):
        n = int((labels == idx).sum())
        print(f"  {name} (label={idx}): {n} sampel")


if __name__ == "__main__":
    process_dataset()
