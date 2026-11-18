
import argparse
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

# 1. PREPROCESSING

class OtsuBinarize:

    def __init__(self, size: int = 112, margin: float = 0.15, crop: bool = True,
                 clean_border: bool = True, band: float = 0.12, min_len: float = 0.45):
        self.size, self.margin, self.crop = size, margin, crop
        self.clean_border, self.band, self.min_len = clean_border, band, min_len

    @staticmethod
    def _otsu(gray):
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def _border_line_mask(self, bw):
        h, w = bw.shape
        n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        m = np.zeros_like(bw)
        for i in range(1, n):
            x, y, cw, ch, _area = st[i]
            cx = x + cw / 2
            if (cx < w * self.band or cx > w * (1 - self.band)) and cw <= 0.10 * w and ch >= self.min_len * h:
                m[lab == i] = 255
        return cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2)

    def _binarize(self, gray):
        bw = self._otsu(gray)
        if not self.clean_border:
            return bw
        h, w = bw.shape
        m = self._border_line_mask(bw)
        if m.any():
            g2 = gray.copy()
            g2[m > 0] = 255            # hapus garis dari citra abu-abu
            bw = self._otsu(g2)        # Otsu ulang -> digit pudar tidak lagi putus-putus
            bw[m > 0] = 0
        # buang serpihan garis putus-putus / bintik kecil di pita kiri/kanan
        n, lab, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        total = max(1, int((bw > 0).sum()))
        for i in range(1, n):
            x, y, cw, ch, area = st[i]
            cx = x + cw / 2
            in_band = cx < w * 0.10 or cx > w * 0.90
            if in_band and (area < 0.03 * total or cw <= 0.06 * w):
                bw[lab == i] = 0
        return bw

    def __call__(self, img: Image.Image) -> Image.Image:
        bw = self._binarize(np.array(img.convert("L")))

        if self.crop:
            ys, xs = np.where(bw > 0)
            if len(xs) > 0:
                bw = bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        h, w = bw.shape
        side = max(h, w)
        pad = int(side * self.margin)
        canvas = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
        y0, x0 = pad + (side - h) // 2, pad + (side - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = bw

        canvas = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return Image.fromarray(canvas, mode="L")


def build_transforms(img_size: int, pretrained: bool, clean_border: bool = True):

    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if pretrained else ((0.5,) * 3, (0.5,) * 3)

    train_tf = transforms.Compose([
        OtsuBinarize(img_size, clean_border=clean_border),
        transforms.RandomAffine(degrees=10,             # rotasi +-10 derajat
                                translate=(0.1, 0.1),   # geser maks 10%
                                scale=(0.9, 1.15),      # zoom in/out
                                shear=5, fill=0),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        OtsuBinarize(img_size, clean_border=clean_border),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# 2. ARSITEKTUR GOOGLENET (INCEPTION V1)

class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride, padding, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class Inception(nn.Module):
    """4 cabang paralel: 1x1 | 1x1->3x3 | 1x1->5x5 | maxpool->1x1, lalu di-concat."""

    def __init__(self, cin, c1, c3r, c3, c5r, c5, cp):
        super().__init__()
        self.b1 = ConvBNReLU(cin, c1, 1)
        self.b2 = nn.Sequential(ConvBNReLU(cin, c3r, 1), ConvBNReLU(c3r, c3, 3, padding=1))
        self.b3 = nn.Sequential(ConvBNReLU(cin, c5r, 1), ConvBNReLU(c5r, c5, 5, padding=2))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, 1, 1), ConvBNReLU(cin, cp, 1))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class AuxClassifier(nn.Module):
    
    def __init__(self, cin, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.conv = ConvBNReLU(cin, 128, 1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.drop = nn.Dropout(0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv(self.pool(x)).flatten(1)
        return self.fc2(self.drop(F.relu(self.fc1(x), inplace=True)))


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES, aux_logits=True, dropout=0.4):
        super().__init__()
        self.aux_logits = aux_logits
        self.stem = nn.Sequential(
            ConvBNReLU(3, 64, 7, stride=2, padding=3),
            nn.MaxPool2d(3, 2, ceil_mode=True),
            ConvBNReLU(64, 64, 1),
            ConvBNReLU(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2, ceil_mode=True),
        )
        self.i3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.i3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.i4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.i4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.i4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.i4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.i4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(3, 2, ceil_mode=True)

        self.i5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.i5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)  # <- 10 kelas

        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)  # setelah 4a
            self.aux2 = AuxClassifier(528, num_classes)  # setelah 4d

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool3(self.i3b(self.i3a(x)))
        x = self.i4a(x)
        a1 = self.aux1(x) if (self.aux_logits and self.training) else None
        x = self.i4d(self.i4c(self.i4b(x)))
        a2 = self.aux2(x) if (self.aux_logits and self.training) else None
        x = self.pool4(self.i4e(x))
        x = self.i5b(self.i5a(x))
        x = self.fc(self.dropout(self.avgpool(x).flatten(1)))
        return (x, a1, a2) if a1 is not None else x


def build_model(pretrained: bool = False, num_classes: int = NUM_CLASSES) -> nn.Module:
    if pretrained:
        # GoogLeNet bobot ImageNet; aux head tidak dipakai saat fine-tuning
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # 1000 -> 10 kelas
        return model
    return GoogLeNet(num_classes=num_classes, aux_logits=True)


# 3. DATASET & TRAINING LOOP
def stratified_split(targets, val_ratio=0.15, seed=42):
    """Split train/val per kelas agar distribusi kelas terjaga."""
    rng = random.Random(seed)
    per_class = {}
    for i, t in enumerate(targets):
        per_class.setdefault(t, []).append(i)
    train_idx, val_idx = [], []
    for idxs in per_class.values():
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_ratio)
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


def build_loaders(data_dir, img_size, pretrained, batch_size, workers, val_ratio, clean_border=True):
    train_tf, eval_tf = build_transforms(img_size, pretrained, clean_border)
    # Dua instance ImageFolder (transform berbeda), indeks split yang sama
    train_full = datasets.ImageFolder(data_dir, transform=train_tf)
    val_full = datasets.ImageFolder(data_dir, transform=eval_tf)
    train_idx, val_idx = stratified_split(train_full.targets, val_ratio)

    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size,
                              shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(Subset(val_full, val_idx), batch_size=batch_size * 2,
                            shuffle=False, **kw)
    return train_loader, val_loader, train_full.class_to_idx


def compute_loss(outputs, labels, criterion, aux_weight=0.3):
    """Loss total = main + 0.3 * (aux1 + aux2), sesuai paper GoogLeNet."""
    if isinstance(outputs, (tuple, list)) and outputs[1] is not None:
        main, a1, a2 = outputs
        return criterion(main, labels) + aux_weight * (criterion(a1, labels) + criterion(a2, labels)), main
    main = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    return criterion(main, labels), main


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)
    return total_loss / n, correct / n


def train(args):
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    img_size = 224 if args.pretrained and args.img_size is None else (args.img_size or 112)
    print(f"Device: {DEVICE} | img_size: {img_size} | pretrained: {args.pretrained} "
          f"| clean_border: {not args.no_clean_border}")

    train_loader, val_loader, class_to_idx = build_loaders(
        args.data, img_size, args.pretrained, args.batch_size, args.workers, args.val_ratio,
        clean_border=not args.no_clean_border)
    print(f"Kelas: {class_to_idx}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = build_model(args.pretrained).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run_loss, correct, n = time.time(), 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=DEVICE.type == "cuda"):
                outputs = model(imgs)                                   # forward
                loss, main = compute_loss(outputs, labels, criterion)   # loss (+aux)
            scaler.scale(loss).backward()                               # backward
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * labels.size(0)
            correct += (main.argmax(1) == labels).sum().item()
            n += labels.size(0)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {run_loss / n:.4f} acc {correct / n:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "img_size": img_size,
                "pretrained": args.pretrained,
                "clean_border": not args.no_clean_border,
            }, args.ckpt)
            print(f"  -> checkpoint disimpan ({args.ckpt}), val acc terbaik {best_acc:.4f}")
    print(f"Selesai. Val acc terbaik: {best_acc:.4f}")


# 4. INFERENCE (SATU GAMBAR)

def load_for_inference(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(ckpt["pretrained"])
    if not ckpt["pretrained"]:
        model.aux_logits = False  # aux head tidak diperlukan saat inferensi
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE).eval()
    # ckpt lama (tanpa kunci ini) dilatih tanpa pembersihan garis tepi
    _, eval_tf = build_transforms(ckpt["img_size"], ckpt["pretrained"], ckpt.get("clean_border", False))
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    return model, eval_tf, idx_to_class


@torch.no_grad()
def predict_image(image_path: str, model, eval_tf, idx_to_class, topk: int = 3):
    """Return: (digit_prediksi, confidence, [(digit, prob), ...] top-k)."""
    img = Image.open(image_path).convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(DEVICE)          # preprocessing identik dgn validasi
    probs = F.softmax(model(x), dim=1)[0].cpu()
    conf, idx = probs.max(0)
    top = [(idx_to_class[i.item()], p.item()) for p, i in zip(*probs.topk(topk))]
    return idx_to_class[idx.item()], conf.item(), top


def predict(args):
    model, eval_tf, idx_to_class = load_for_inference(args.ckpt)
    digit, conf, top = predict_image(args.image, model, eval_tf, idx_to_class)
    print(f"Prediksi : {digit}  (confidence {conf * 100:.2f}%)")
    print("Top-3    : " + ", ".join(f"{d}: {p * 100:.1f}%" for d, p in top))


# =============================================================================
if __name__ == "__main__":  # guard wajib di Windows karena DataLoader multiprocessing
    parser = argparse.ArgumentParser(description="GoogLeNet handwritten digit recognition")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--data", required=True, help="folder ImageFolder (berisi folder 0..9)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img_size", type=int, default=None, help="default 112 (scratch) / 224 (pretrained)")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--no_clean_border", action="store_true",
                   help="matikan pembuangan garis tepi (untuk pembanding A/B)")
    p.add_argument("--ckpt", default="googlenet_digits.pth")
    p.set_defaults(func=train)

    p = sub.add_parser("predict")
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", default="googlenet_digits.pth")
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)