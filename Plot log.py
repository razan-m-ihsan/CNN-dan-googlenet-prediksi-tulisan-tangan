"""
Buat grafik kurva loss & akurasi dari log training (tanpa perlu training ulang).

Simpan output training ke file teks, lalu:
    python plot_log.py --log train_log.txt --out training_curves.png

Untuk menyimpan log otomatis saat training berikutnya (PowerShell):
    python model.py train --data .\\dataset_HW --epochs 20 | Tee-Object train_log.txt

Kebutuhan: matplotlib  (pip install matplotlib)
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

PATTERN = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*train loss\s+([\d.]+)\s+acc\s+([\d.]+)"
    r"\s*\|\s*val loss\s+([\d.]+)\s+acc\s+([\d.]+)"
)


def parse_log(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                rows.append(tuple(float(g) for g in m.groups()))
    if not rows:
        raise SystemExit("Tidak ada baris 'Epoch ...' yang cocok di log.")
    return list(zip(*rows))  # epoch, tr_loss, tr_acc, va_loss, va_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="train_log.txt")
    ap.add_argument("--out", default="training_curves.png")
    args = ap.parse_args()

    ep, tr_loss, tr_acc, va_loss, va_acc = parse_log(args.log)
    best = max(range(len(va_acc)), key=lambda i: va_acc[i])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ep, tr_loss, "o-", color="#9aa0a6", label="Train (main + aux)", ms=4)
    ax1.plot(ep, va_loss, "o-", color="#d1495b", label="Validasi", ms=4)
    ax1.set_title("Loss per epoch", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax1.legend()

    ax2.plot(ep, [a * 100 for a in tr_acc], "o-", color="#9aa0a6", label="Train", ms=4)
    ax2.plot(ep, [a * 100 for a in va_acc], "o-", color="#2e86ab", label="Validasi", ms=4)
    ax2.scatter([ep[best]], [va_acc[best] * 100], s=120, facecolors="none",
                edgecolors="#2e86ab", linewidths=2, zorder=5)
    ax2.annotate(f"terbaik {va_acc[best] * 100:.2f}% (epoch {int(ep[best])})",
                 (ep[best], va_acc[best] * 100), textcoords="offset points",
                 xytext=(-10, -28), ha="right", fontsize=9)
    ax2.set_title("Akurasi per epoch", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Akurasi (%)")
    ax2.set_ylim(min(min(va_acc), 0.97) * 100 - 0.5, 100.2)
    ax2.legend(loc="lower right")

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Tersimpan: {args.out}")


if __name__ == "__main__":
    main()