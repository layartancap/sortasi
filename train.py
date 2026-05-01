"""
train.py — Training Random Forest dengan penanganan class imbalance.
=====================================================================
AUTHOR   : Solehin Rizal (original) — diperbaiki untuk akurasi lebih baik
WEBSITE  : www.cytron.io

ROOT CAUSE masalah "semua prediksi ikut kelas mayoritas":
─────────────────────────────────────────────────────────
  Ketika filter kontur diubah, jumlah sampel accepted vs reject berubah.
  Jika accepted > reject, RF belajar "tebak accepted selalu = aman".
  Jika reject > accepted, RF belajar "tebak reject selalu = aman".
  Ini terjadi karena:
    1. Tidak ada class_weight → semua sampel dianggap sama penting.
    2. Threshold selalu 0.5 meskipun distribusi probabilitas condong satu sisi.
    3. Evaluasi hanya pakai accuracy → tidak kelihatan kalau model bias.

SOLUSI:
─────────────────────────────────────────────────────────
  A. class_weight="balanced"  → RF beri bobot lebih ke kelas minoritas.
  B. Balancing data            → potong mayoritas agar jumlah sama persis.
  C. Threshold tuning          → cari threshold optimal via F1, bukan pakai 0.5.
  D. Evaluasi per-kelas        → tampilkan precision, recall, F1 tiap kelas.
"""

import pickle
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
DATA_FILE   = "data.npy"
LABELS_FILE = "labels.npy"
MODEL_FILE  = "defect_model.pkl"
CLASS_NAMES = ["accepted", "reject"]   # label 0 = accepted, label 1 = reject

# Random Forest
N_ESTIMATORS = 100   # sama seperti asli, bisa dinaikkan ke 200 untuk lebih akurat
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  TRAINING SORTASI BIJI KOPI — ANTI CLASS IMBALANCE")
print("=" * 55)

data   = np.load(DATA_FILE)
labels = np.load(LABELS_FILE)

print(f"\n[DATA] Total sampel: {len(labels)}")

# Tampilkan distribusi SEBELUM balancing
classes, counts = np.unique(labels, return_counts=True)
for cls, cnt in zip(classes, counts):
    pct = cnt / len(labels) * 100
    bar = "█" * int(pct / 2)
    print(f"  {CLASS_NAMES[cls]:10s} (label={cls}): {cnt:4d} sampel  {pct:5.1f}%  {bar}")

imbalance_ratio = counts.max() / counts.min()
print(f"\n  Rasio imbalance: {imbalance_ratio:.2f}x  ", end="")
if imbalance_ratio > 2.0:
    print("⚠️  TINGGI — balancing wajib dilakukan")
elif imbalance_ratio > 1.3:
    print("⚠️  SEDANG — balancing disarankan")
else:
    print("✓  Distribusi sudah cukup seimbang")

# ─────────────────────────────────────────────────────────────
# STEP 2 — BALANCING: undersample kelas mayoritas
# ─────────────────────────────────────────────────────────────
# Definisi:
#   Kelas mayoritas = kelas dengan jumlah sampel terbanyak.
#   Kelas minoritas = kelas dengan jumlah sampel paling sedikit.
#
# Strategi: potong (undersample) kelas mayoritas secara acak
#   agar jumlahnya sama persis dengan kelas minoritas.
#   Tidak menambah data baru — hanya membuang kelebihan mayoritas.
#
# Mengapa ini perlu?
#   Jika accepted=300, reject=50 → tanpa balancing, RF bisa asal
#   prediksi "accepted" terus dan dapat accuracy 85% padahal tidak berguna.
#   Setelah balancing accepted=50, reject=50 → RF harus sungguh belajar.

min_count = counts.min()   # jumlah sampel kelas paling sedikit
rng = np.random.RandomState(RANDOM_STATE)

idx_balanced = []
for cls in classes:
    idx_cls = np.where(labels == cls)[0]          # semua indeks kelas ini
    idx_chosen = rng.choice(                       # pilih acak sejumlah min_count
        idx_cls, size=min_count, replace=False
    )
    idx_balanced.append(idx_chosen)

idx_balanced = np.concatenate(idx_balanced)
rng.shuffle(idx_balanced)                          # acak urutan agar tidak bias urutan

X_bal = data[idx_balanced]
y_bal = labels[idx_balanced]

print(f"\n[BALANCE] Setelah undersample:")
for cls in classes:
    n = int((y_bal == cls).sum())
    print(f"  {CLASS_NAMES[cls]:10s} (label={cls}): {n} sampel")
print(f"  Total: {len(y_bal)} sampel")

# ─────────────────────────────────────────────────────────────
# STEP 3 — SPLIT TRAIN / TEST
# ─────────────────────────────────────────────────────────────
# Pakai stratify=y_bal agar proporsi kelas sama di train dan test set.
# Tanpa stratify, bisa saja test set kebetulan hanya berisi satu kelas.

trainX, testX, trainY, testY = train_test_split(
    X_bal, y_bal,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=y_bal          # ← penting: jaga proporsi di setiap split
)
print(f"\n[SPLIT] Train: {len(trainY)} sampel | Test: {len(testY)} sampel")

# ─────────────────────────────────────────────────────────────
# STEP 4 — TRAINING dengan class_weight="balanced"
# ─────────────────────────────────────────────────────────────
# class_weight="balanced":
#   RF menghitung bobot tiap kelas = total_sampel / (n_kelas × n_sampel_kelas).
#   Kelas yang lebih sedikit mendapat bobot lebih besar saat menghitung loss.
#   Ini berlaku sebagai lapisan perlindungan KEDUA setelah undersample.
#
# Pipeline dengan StandardScaler:
#   GLCM dan HSV histogram punya skala sangat berbeda.
#   StandardScaler normalisasi semua fitur ke mean=0, std=1
#   sehingga fitur berskala besar tidak mendominasi.

print("\n[TRAIN] Melatih Random Forest...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",    # ← kunci anti-imbalance
        max_features="sqrt",
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ))
])
model.fit(trainX, trainY)
print("  ✓ Training selesai.")

# ─────────────────────────────────────────────────────────────
# STEP 5 — EVALUASI PER KELAS (bukan hanya accuracy)
# ─────────────────────────────────────────────────────────────
# Accuracy saja menyesatkan saat imbalance.
# Kita perlu precision, recall, dan F1 PER KELAS:
#
#   Precision = dari semua yang diprediksi "reject",
#               berapa % yang benar-benar reject?
#               (tinggi = sedikit false alarm)
#
#   Recall    = dari semua biji reject yang ada,
#               berapa % yang berhasil terdeteksi?
#               (tinggi = sedikit biji jelek yang lolos)
#
#   F1        = rata-rata harmonis precision & recall.
#               Nilai 1.0 = sempurna.

y_pred  = model.predict(testX)
y_proba = model.predict_proba(testX)[:, 1]   # probabilitas kelas "reject"

accuracy = (y_pred == testY).mean()

print(f"\n[EVALUASI] Accuracy keseluruhan: {accuracy * 100:.2f}%")
print("\n  Laporan per kelas:")
print("  " + "-" * 52)
report = classification_report(
    testY, y_pred,
    target_names=CLASS_NAMES,
    digits=3
)
for line in report.strip().split("\n"):
    print("  " + line)
print("  " + "-" * 52)

cm = confusion_matrix(testY, y_pred)
print(f"\n  Confusion Matrix:")
print(f"                  Pred accepted  Pred reject")
print(f"  Actual accepted     {cm[0,0]:5d}          {cm[0,1]:5d}")
print(f"  Actual reject       {cm[1,0]:5d}          {cm[1,1]:5d}")
print(f"\n  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"  (FP = accepted salah diklasifikasi reject)")
print(f"  (FN = reject lolos tidak terdeteksi)")

# ─────────────────────────────────────────────────────────────
# STEP 6 — THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────
# Default threshold = 0.5:
#   predict kelas "reject" jika P(reject) > 0.5
#
# Masalah: saat data tidak seimbang, distribusi probabilitas condong
# ke kelas mayoritas. Threshold 0.5 jadi terlalu tinggi untuk
# kelas minoritas → banyak reject yang lolos.
#
# Solusi: cari threshold yang memaksimalkan F1 kelas reject.
# Kita scan semua kemungkinan threshold dari 0.1 sampai 0.9.

print(f"\n[THRESHOLD] Mencari threshold optimal untuk kelas reject...")
print(f"  (semakin kecil threshold, makin banyak biji dianggap reject)")

thresholds   = np.arange(0.10, 0.91, 0.01)
best_thresh  = 0.5
best_f1      = 0.0
results_scan = []

for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    f1 = f1_score(testY, y_pred_thr, pos_label=1, zero_division=0)
    results_scan.append((thr, f1))
    if f1 > best_f1:
        best_f1    = f1
        best_thresh = thr

print(f"\n  Threshold default (0.50) → F1 reject = "
      f"{f1_score(testY, (y_proba>=0.5).astype(int), pos_label=1, zero_division=0):.3f}")
print(f"  Threshold optimal ({best_thresh:.2f}) → F1 reject = {best_f1:.3f}")

# Tampilkan prediksi ulang dengan threshold optimal
y_pred_opt = (y_proba >= best_thresh).astype(int)
acc_opt    = (y_pred_opt == testY).mean()
print(f"\n  Dengan threshold {best_thresh:.2f}:")
print(f"  Accuracy: {acc_opt * 100:.2f}%")
cm_opt = confusion_matrix(testY, y_pred_opt)
print(f"  Confusion Matrix:")
print(f"                  Pred accepted  Pred reject")
print(f"  Actual accepted     {cm_opt[0,0]:5d}          {cm_opt[0,1]:5d}")
print(f"  Actual reject       {cm_opt[1,0]:5d}          {cm_opt[1,1]:5d}")

# ─────────────────────────────────────────────────────────────
# STEP 7 — CROSS VALIDATION (opsional, lebih terpercaya)
# ─────────────────────────────────────────────────────────────
print(f"\n[CV] Validasi 5-fold (lebih reliable dari 1 split)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
y_cv_pred = cross_val_predict(model, X_bal, y_bal, cv=cv)

cv_report = classification_report(
    y_bal, y_cv_pred,
    target_names=CLASS_NAMES,
    digits=3
)
print("  Laporan CV per kelas:")
print("  " + "-" * 52)
for line in cv_report.strip().split("\n"):
    print("  " + line)

# ─────────────────────────────────────────────────────────────
# STEP 8 — SIMPAN MODEL
# ─────────────────────────────────────────────────────────────
# Simpan dalam format bundle dict (kompatibel dengan detect_and_sort.py).
# bundle["model"]       → pipeline scaler + RF
# bundle["class_names"] → ["accepted", "reject"]
# bundle["threshold"]   → threshold optimal hasil tuning
#
# PENTING: format ini BERBEDA dari kode asli (pickle.dump(model, f)).
# detect_and_sort.py harus baca dengan:
#   bundle = pickle.load(f)
#   model  = bundle["model"]

bundle = {
    "model":       model,
    "class_names": CLASS_NAMES,
    "threshold":   float(best_thresh),   # dipakai di detect_and_sort.py
}

with open(MODEL_FILE, "wb") as f:
    pickle.dump(bundle, f)

print(f"\n[SIMPAN] Model tersimpan: {MODEL_FILE}")
print(f"  class_names : {CLASS_NAMES}")
print(f"  threshold   : {best_thresh:.2f}  (dipakai di detect_and_sort.py)")
print(f"\n✓ Selesai. Jalankan detect_and_sort.py untuk mulai sortasi.")
print("=" * 55)
