# milestone1_pipeline.py
import os, zipfile, json, math, shutil
from glob import glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

# CONFIG
# Set your data directory here/ data paths for input images and outputs
DATA_DIR = Path("D:/asu/Fall 2025/CSE 483 Computer vision/Project/Gravity Falls")
OUT_DIR = Path("puzzle_output")            # output folder
MAX_DM = 1200  # resize large images to this max dim for processing; contours mappedback
MIN_AREA_RATIO = 0.002  # min conour area relative to image area (tune per dataset)

# UTIL
# helper functions
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def normalize_contour(cnt):
    cnt = np.squeeze(cnt).astype(int)
    if cnt.ndim == 1:
        cnt = cnt.reshape(1,2)
    return cnt.tolist()

def discrete_curvature(contour, k=5):
    n = len(contour)
    curv = np.zeros(n)
    for i in range(n):
        prev = contour[(i-k)%n]
        curr = contour[i]
        nxt = contour[(i+k)%n]
        v1 = np.array(curr) - np.array(prev)
        v2 = np.array(nxt) - np.array(curr)
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            curv[i]=0; continue
        cosang = np.dot(v1,v2)/(n1*n2)
        cosang = np.clip(cosang, -1, 1)
        curv[i] = math.acos(cosang)
    return curv

def split_contour_by_curvature(cnt, k=6, factor=1.7, min_seg_len=25):
    curv = discrete_curvature(cnt, k=k)
    thresh = max(np.median(curv)*factor, np.percentile(curv,90)*0.5)
    peaks = np.where(curv > thresh)[0]
    if len(peaks) == 0:
        return [cnt.tolist()]
    peaks_sorted = np.sort(peaks)
    clusters=[]; current=[peaks_sorted[0]]
    for p in peaks_sorted[1:]:
        if p - current[-1] <= k*2:
            current.append(p)
        else:
            clusters.append(current); current=[p]
    clusters.append(current)
    cut_indices = [int(np.mean(c)) for c in clusters]
    cut_indices = sorted(list(set(cut_indices)))
    segs=[]
    n=len(cnt)
    for i in range(len(cut_indices)):
        a=cut_indices[i]; b=cut_indices[(i+1)%len(cut_indices)]
        if b>a:
            seg = cnt[a:b+1]
        else:
            seg = np.vstack([cnt[a:], cnt[:b+1]])
        if len(seg) >= min_seg_len:
            segs.append(seg.tolist())
    if len(segs)==0:
        return [cnt.tolist()]
    return segs


def smooth_contour(contour, window=5):
    """Circular moving-average smoothing of a contour (Nx2 array).
    Returns a smoothed Nx2 numpy array (float32).
    """
    if window <= 1:
        return np.asarray(contour, dtype=np.float32)
    arr = np.asarray(contour, dtype=np.float32)
    n = len(arr)
    pad = window // 2
    # pad circularly so smoothing wraps around the contour
    xs = np.pad(arr[:, 0], pad, mode='wrap')
    ys = np.pad(arr[:, 1], pad, mode='wrap')
    kernel = np.ones(window, dtype=np.float32) / float(window)
    xs_s = np.convolve(xs, kernel, mode='valid')
    ys_s = np.convolve(ys, kernel, mode='valid')
    sm = np.vstack([xs_s, ys_s]).T.astype(np.float32)
    if len(sm) != n:
        sm = sm[:n]
    return sm
#لحد هنا كل اللي فوق ده عبارة عن تجهيزات للصوره عشان اقدر اعمل عليها العمليه المطلوبه 

# main processing function
def process_all_images(data_dir, out_dir, max_dim=MAX_DM, min_area_ratio=MIN_AREA_RATIO):
    ensure_dir(out_dir)
    imgs = sorted(glob(str(Path(data_dir) / "**" / "*.png"), recursive=True) +
                  glob(str(Path(data_dir) / "**" / "*.jpg"), recursive=True) +
                  glob(str(Path(data_dir) / "**" / "*.jpeg"), recursive=True))
    metadata = []
    summary = []
    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            print("Couldn't read", img_path); continue
        h,w = img.shape[:2]
        scale = 1.0
        if max(h,w) > max_dim:
            scale = max_dim / max(h,w)
            img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            img_small = img.copy()
        # preprocessing
        den = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); gray = clahe.apply(gray)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        min_area = img_small.shape[0]*img_small.shape[1]*min_area_ratio
        pieces = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area: continue
            pieces.append((area, cnt))
        pieces = sorted(pieces, key=lambda x:-x[0])
        base = Path(out_dir) / Path(img_path).stem
        ensure_dir(base / "masks"); ensure_dir(base / "crops"); ensure_dir(base / "vis")
        # save visualization
        vis = img_small.copy()
        if pieces:
            cv2.drawContours(vis, [p[1] for p in pieces], -1, (0,255,0), 2)
        cv2.imwrite(str(base / "vis" / f"{Path(img_path).stem}_contours.png"), vis)
        # each piece
        for i,(area,cnt) in enumerate(pieces):
            uid = f"{Path(img_path).stem}_piece{i}"
            cnt_orig = (cnt.astype(np.float32) / scale).astype(int) if scale!=1.0 else cnt
            cnt_norm = normalize_contour(cnt_orig)
            # smooth contour for more stable curvature/edge detection
            cnt_np = np.array(cnt_norm, dtype=np.float32)
            cnt_np_smooth = smooth_contour(cnt_np, window=5)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt_orig], -1, 255, -1)
            x,y,wid,ht = cv2.boundingRect(cnt_orig)
            crop = cv2.bitwise_and(img[y:y+ht, x:x+wid], img[y:y+ht, x:x+wid], mask=mask[y:y+ht, x:x+wid])
            # save cropped mask (more convenient for downstream processing)
            mask_crop = mask[y:y+ht, x:x+wid]
            mask_path = str(base / "masks" / f"{uid}_mask.png")
            crop_path = str(base / "crops" / f"{uid}_crop.png")
            cv2.imwrite(mask_path, mask_crop)
            cv2.imwrite(crop_path, crop)
            # edges computed from smoothed contour
            edges = split_contour_by_curvature(cnt_np_smooth, k=6, factor=1.7, min_seg_len=25)
            meta = {
                "id": uid,
                "source_image": img_path,
                "area_px": int(area/(scale*scale)),
                "bbox": [int(x),int(y),int(wid),int(ht)],
                "mask_path": mask_path,
                "crop_path": crop_path,
                "contour_n_points": int(len(cnt_norm)),
                "contour": cnt_norm,
                "contour_smoothed": cnt_np_smooth.tolist(),
                "edges": edges,
                "image_shape": list(img.shape),
                "scale": float(scale)
            }
            metadata.append(meta)
            summary.append({"id":uid, "source":Path(img_path).stem, "area_px":meta["area_px"], "bbox":meta["bbox"], "n_contour_pts":meta["contour_n_points"], "n_edges": len(edges)})
    # save metadata
    with open(Path(out_dir)/"pieces_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    pd.DataFrame(summary).to_csv(Path(out_dir)/"pieces_summary.csv", index=False)
    print("Done. outputs in:", out_dir)

if __name__ == "__main__":
    # set DATA_DIR before running or modify the DATA_DIR value above
    process_all_images(DATA_DIR, OUT_DIR)
