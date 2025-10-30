# =========================
# Lab 2 — ECE 491
# Basic Image Processing (pure NumPy; optional imageio/Pillow for I/O)
# =========================
from __future__ import annotations
import argparse, os
from typing import Tuple
import numpy as np

# Try readers (imageio or Pillow). If missing, we’ll synthesize an image.
def _try_import_image_readers():
    imageio = None; pil_Image = None
    try:
        import imageio.v2 as imageio
    except Exception:
        pass
    try:
        from PIL import Image as pil_Image
    except Exception:
        pass
    return imageio, pil_Image

IMAGEIO, PIL_IMAGE = _try_import_image_readers()

def load_image(path: str) -> np.ndarray:
    if os.path.isfile(path):
        try:
            if IMAGEIO is not None:
                img = IMAGEIO.imread(path)
            elif PIL_IMAGE is not None:
                img = np.array(PIL_IMAGE.open(path))
            else:
                raise RuntimeError("No image reader available (install imageio or pillow)")
        except Exception:
            print(f"[WARN] Failed to read {path}; using synthetic pattern.")
            img = synthetic_image()
    else:
        print(f"[WARN] File not found: {path}; using synthetic pattern.")
        img = synthetic_image()
    img = img.astype(np.float32)
    if img.max() > 0: img /= img.max()
    return img

def synthetic_image(size: Tuple[int, int]=(256,256)) -> np.ndarray:
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    grad = (x/(w-1) + y/(h-1)) / 2.0
    circle = (((x-w/2)**2 + (y-h/2)**2) < (min(h,w)/4)**2).astype(float)
    return np.clip(0.6*grad + 0.4*circle, 0, 1)

def conv2d(image: np.ndarray, kernel: np.ndarray, mode: str="same") -> np.ndarray:
    if image.ndim != 2: raise ValueError("conv2d expects a 2D array")
    k = np.flipud(np.fliplr(kernel))
    H, W = image.shape; kH, kW = k.shape
    py, px = kH//2, kW//2
    if mode == "same":
        padded = np.pad(image, ((py,py),(px,px)), mode="reflect")
        out = np.zeros_like(image, dtype=np.float32)
        for i in range(H):
            for j in range(W):
                patch = padded[i:i+kH, j:j+kW]
                out[i,j] = float(np.sum(patch * k))
        return out
    else:
        padded = np.pad(image, ((kH-1,kH-1),(kW-1,kW-1)), mode="reflect")
        outH, outW = H + kH - 1, W + kW - 1
        out = np.zeros((outH,outW), dtype=np.float32)
        for i in range(outH):
            for j in range(outW):
                patch = padded[i:i+kH, j:j+kW]
                out[i,j] = float(np.sum(patch * k))
        return out

# Kernels
BOX_3   = (1/9.0)*np.array([[1,1,1],[1,1,1],[1,1,1]], dtype=np.float32)
SOBEL_L = np.array([[ 1,0,-1],[ 2,0,-2],[ 1,0,-1]], dtype=np.float32)  # vertical-edge (x-gradient)
EDGE_H  = np.array([[-1, 2, -1]], dtype=np.float32)                    # 1x3 row  -> horizontal edges
EDGE_V  = np.array([[-1],[ 2],[-1]], dtype=np.float32)                 # 3x1 col  -> vertical edges

def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return img
    if img.ndim == 3:
        if img.shape[2] == 4: img = img[..., :3]
        if img.shape[2] == 3:
            r,g,b = img[...,0], img[...,1], img[...,2]
            return 0.299*r + 0.587*g + 0.114*b
    raise ValueError(f"Unsupported image shape: {img.shape}")

def normalize(x: np.ndarray) -> np.ndarray:
    m, M = x.min(), x.max()
    return np.zeros_like(x) if M-m < 1e-12 else (x-m)/(M-m)

def save_image(path: str, arr: np.ndarray):
    arr8 = (np.clip(arr,0,1)*255).astype(np.uint8)
    if IMAGEIO is not None: IMAGEIO.imwrite(path, arr8)
    elif PIL_IMAGE is not None: PIL_IMAGE.fromarray(arr8).save(path)
    else: np.save(path+'.npy', arr8)

def process_single(name: str, img: np.ndarray, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    save_image(os.path.join(outdir, f"{name}_orig.png"), np.clip(img,0,1))
    if img.ndim == 3 and img.shape[2] == 4: img = img[...,:3]
    is_color = (img.ndim == 3 and img.shape[2] == 3)
    if is_color:
        # Box per channel
        channels = [conv2d(img[...,c], BOX_3) for c in range(3)]
        save_image(os.path.join(outdir, f"{name}_box_color.png"), np.stack(channels, axis=2))
        gray = to_grayscale(img)
    else:
        gray = img
    H,W = gray.shape
    save_image(os.path.join(outdir, f"{name}_quarter.png"), gray[:H//2, :W//2])
    box  = conv2d(gray, BOX_3)
    sob  = conv2d(gray, SOBEL_L)              # vertical-edge detector (x-gradient)
    eh   = conv2d(gray, EDGE_H)               # horizontal edges
    ev   = conv2d(gray, EDGE_V)               # vertical edges
    emag = normalize(np.sqrt(eh**2 + ev**2))
    for tag,arr in [("gray",gray),("box",box),("sobel_like",sob),
                    ("edge_h",eh),("edge_v",ev),("edge_mag",emag)]:
        save_image(os.path.join(outdir, f"{name}_{tag}.png"), arr)
    return {
        "is_color": is_color,
        "shape": img.shape if is_color else gray.shape,
        "box_mean": float(box.mean()),
        "sobel_mean_abs": float(np.mean(np.abs(sob))),
        "edge_mag_mean": float(emag.mean()),
    }

def write_analysis(outdir: str, stats: dict):
    lines = ["Lab 2 Analysis", "================", ""]
    for name,s in stats.items():
        lines += [
            f"Image: {name}",
            f"  Original shape: {s['shape']}, color={s['is_color']}",
            f"  Box filter mean intensity: {s['box_mean']:.4f}",
            f"  Mean abs(Sobel-like): {s['sobel_mean_abs']:.4f}",
            f"  Edge magnitude mean: {s['edge_mag_mean']:.4f}",
            ""
        ]
    lines += [
        "Observations:",
        "- The 3x3 box filter smooths fine detail (low-pass).",
        "- Sobel-like emphasizes vertical transitions (x-gradient).",
        "- Row kernel [-1 2 -1] highlights horizontal edges; column version highlights vertical edges.",
        "- Edge magnitude combines both orientations.",
    ]
    with open(os.path.join(outdir, "analysis_lab2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--boat", default="boat.png")        # or boat.jpg
    p.add_argument("--barbara", default="Barbara.jpeg") # match your actual file
    p.add_argument("--outdir", default="results_lab2")
    args = p.parse_args()

    imgs = {
        "boat":    load_image(args.boat),
        "barbara": load_image(args.barbara),
    }
    stats = {}
    for name, img in imgs.items():
        print(f"[INFO] Processing {name} (shape={img.shape})")
        stats[name] = process_single(name, img, args.outdir)
    write_analysis(args.outdir, stats)
    print(f"[DONE] Results written to {args.outdir}/")

if __name__ == "__main__":
    main()
