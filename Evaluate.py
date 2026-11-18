"""
Evaluasi model GoogLeNet: confusion matrix, metrik per kelas, dan gambar yang salah prediksi.

Letakkan di folder yang sama dengan model.py, lalu:
    python evaluate.py --data .\\dataset_HW --ckpt googlenet_digits.pth

Keluaran:
    confusion_matrix.png   - matriks kebingungan (jumlah gambar)
    misclassified.png      - galeri gambar yang salah diprediksi
    (+ tabel precision/recall/F1 per kelas di terminal)

Kebutuhan: matplotlib  (pip install matplotlib)

Catatan: split validasi memakai fungsi & seed yang sama dengan training
(stratified_split, seed=42, val_ratio=0.15), jadi datanya persis set validasi
yang dipakai saat training. Set ini juga dipakai memilih checkpoint terbaik,
jadi angkanya sedikit optimistis dibanding test set yang benar-benar terpisah.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from model import DEVICE, load_for_inference, stratified_split


@torch.no_grad()
def collect_predictions(model, loader):
    preds, labels, confs = [], [], []
    for imgs, y in loader:
        probs = F.softmax(model(imgs.to(DEVICE)), dim=1)
        conf, pred = probs.max(1)
        preds.append(pred.cpu())
        labels.append(y)
        confs.append(conf.cpu())
    return torch.cat(preds).numpy(), torch.cat(labels).numpy(), torch.cat(confs).numpy()


def confusion_matrix(labels, preds, n):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, names, out):
    n = len(names)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Label sebenarnya", fontsize=11)
    ax.set_title("Confusion matrix (data validasi)", fontsize=14, fontweight="bold")
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified(paths, labels, preds, confs, names, out, max_show=20):
    wrong = np.where(labels != preds)[0]
    if len(wrong) == 0:
        print("Tidak ada gambar yang salah prediksi.")
        return
    # urutkan dari yang paling "yakin tapi salah"
    wrong = wrong[np.argsort(-confs[wrong])][:max_show]
    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.0 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, k in zip(axes, wrong):
        ax.imshow(Image.open(paths[k]).convert("L"), cmap="gray")
        ax.set_title(f"asli {names[labels[k]]} -> pred {names[preds[k]]}\nconf {confs[k] * 100:.0f}%",
                     fontsize=9, color="#b00020")
    fig.suptitle(f"Gambar salah prediksi ({len(wrong)} dari {int((labels != preds).sum())} ditampilkan)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", default="googlenet_digits.pth")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    model, eval_tf, idx_to_class = load_for_inference(args.ckpt)
    ds = datasets.ImageFolder(args.data, transform=eval_tf)
    _, val_idx = stratified_split(ds.targets, args.val_ratio)
    loader = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.workers)

    preds, labels, confs = collect_predictions(model, loader)
    paths = [ds.samples[i][0] for i in val_idx]  # urutan sama dgn loader (shuffle=False)
    n = len(idx_to_class)
    names = [idx_to_class[i] for i in range(n)]

    cm = confusion_matrix(labels, preds, n)
    acc = (preds == labels).mean()
    print(f"\nData validasi: {len(labels)} gambar | akurasi: {acc * 100:.2f}% "
          f"| salah: {int((preds != labels).sum())}\n")

    print(f"{'Kelas':>5} {'Jumlah':>7} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    for c in range(n):
        tp = cm[c, c]
        prec = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        rec = tp / cm[c, :].sum() if cm[c, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"{names[c]:>5} {cm[c, :].sum():>7} {prec * 100:>9.2f}% {rec * 100:>7.2f}% {f1 * 100:>6.2f}%")

    print("\nPasangan paling sering tertukar (asli -> prediksi):")
    off = [(cm[i, j], names[i], names[j]) for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    for cnt, a, b in sorted(off, reverse=True)[:5]:
        print(f"  {a} -> {b}: {cnt} gambar")

    plot_confusion(cm, names, "confusion_matrix.png")
    plot_misclassified(paths, labels, preds, confs, names, "misclassified.png")
    print("\nTersimpan: confusion_matrix.png, misclassified.png")


if __name__ == "__main__":
    main()