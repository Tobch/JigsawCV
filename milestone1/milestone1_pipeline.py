# milestone1_pipeline.py
import os, zipfile, json, math, shutil
from glob import glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt  # Added for debug visualization

# CONFIG
# Set your data directory here/ data paths for input images and outputs
DATA_DIR = r"C:\Users\belal\Desktop\Fall 2026\computer vision\project\JigsawCV\Gravity Falls\puzzle_8x8"
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

# =============================================================================
# NEW DEBUG FUNCTIONS ADDED FROM ENHANCED CODE
# =============================================================================

def create_debug_visualization(original, denoised, enhanced, binary, contours, output_path):
    """
    Create comprehensive visualization of processing pipeline
    Shows all steps from original image to final contours
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Denoised image
    axes[0, 1].imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Denoised (Bilateral Filter)')
    axes[0, 1].axis('off')
    
    # Enhanced image (grayscale)
    axes[0, 2].imshow(enhanced, cmap='gray')
    axes[0, 2].set_title('Enhanced (CLAHE + Blur)')
    axes[0, 2].axis('off')
    
    # Binary mask
    axes[1, 0].imshow(binary, cmap='gray')
    axes[1, 0].set_title('Binary Mask')
    axes[1, 0].axis('off')
    
    # Contours on original
    contour_vis = original.copy()
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
    axes[1, 1].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(f'Detected Contours: {len(contours)}')
    axes[1, 1].axis('off')
    
    # Individual pieces (color coded)
    piece_vis = np.zeros_like(original)
    for i, contour in enumerate(contours):
        color = plt.cm.tab10(i % 10)
        color_bgr = [int(c * 255) for c in color[:3]][::-1]
        cv2.drawContours(piece_vis, [contour], -1, color_bgr, -1)
    axes[1, 2].imshow(cv2.cvtColor(piece_vis, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Segmented Pieces')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_edges(contour, edges, output_path):
    """
    Visualize detected edges on a contour with different colors
    """
    plt.figure(figsize=(10, 8))
    
    # Plot full contour
    contour_array = np.array(contour)
    plt.plot(contour_array[:, 0], -contour_array[:, 1], 'k-', alpha=0.3, label='Full Contour')
    
    # Plot each edge with different color
    colors = plt.cm.Set3(np.linspace(0, 1, len(edges)))
    for i, edge in enumerate(edges):
        edge_array = np.array(edge)
        plt.plot(edge_array[:, 0], -edge_array[:, 1], 'o-', 
                color=colors[i], markersize=4, label=f'Edge {i+1}')
    
    plt.legend()
    plt.title(f'Detected Edges: {len(edges)}')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# =============================================================================
# MODIFIED MAIN PROCESSING FUNCTION
# =============================================================================

# main processing function
def process_all_images(data_dir, out_dir, max_dim=MAX_DM, min_area_ratio=MIN_AREA_RATIO):
    ensure_dir(out_dir)
    imgs = sorted(glob(str(Path(data_dir) / "**" / "*.png"), recursive=True) +
                  glob(str(Path(data_dir) / "**" / "*.jpg"), recursive=True) +
                  glob(str(Path(data_dir) / "**" / "*.jpeg"), recursive=True))
    metadata = []
    summary = []
    
    print(f"📁 Found {len(imgs)} images in {data_dir}")
    
    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            print("Couldn't read", img_path); continue
        
        print(f"🔍 Processing: {Path(img_path).name}")
        
        h,w = img.shape[:2]
        scale = 1.0
        if max(h,w) > max_dim:
            scale = max_dim / max(h,w)
            img_small = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            img_small = img.copy()
        
        # preprocessing - STORE INTERMEDIATE RESULTS FOR DEBUGGING
        den = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(den, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); 
        gray_clahe = clahe.apply(gray)  # Store CLAHE result
        blur = cv2.GaussianBlur(gray_clahe, (5,5), 0)
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
        ensure_dir(base / "masks"); ensure_dir(base / "crops"); 
        ensure_dir(base / "vis"); ensure_dir(base / "debug")  # ADDED DEBUG FOLDER
        
        # =============================================================================
        # NEW: CREATE DEBUG VISUALIZATION FOR ENTIRE PIPELINE
        # =============================================================================
        debug_path = str(base / "debug" / f"{Path(img_path).stem}_pipeline.png")
        create_debug_visualization(
            original=img_small,
            denoised=den,
            enhanced=blur,  # Using blurred image as enhanced representation
            binary=clean,
            contours=[p[1] for p in pieces],
            output_path=debug_path
        )
        print(f"   💾 Saved debug visualization: {debug_path}")
        
        # save original visualization (existing functionality)
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
            
            # =============================================================================
            # NEW: CREATE EDGE VISUALIZATION FOR EACH PIECE
            # =============================================================================
            edge_vis_path = str(base / "vis" / f"{uid}_edges.png")
            visualize_edges(cnt_np_smooth, edges, edge_vis_path)
            
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
            
            print(f"   ✅ Piece {i}: area={meta['area_px']}px, edges={len(edges)}")
        
        print(f"   📊 Found {len(pieces)} valid pieces")
    
    # save metadata
    with open(Path(out_dir)/"pieces_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    pd.DataFrame(summary).to_csv(Path(out_dir)/"pieces_summary.csv", index=False)
    print("Done. outputs in:", out_dir)
    print(f"🎉 Total pieces processed: {len(metadata)}")

if __name__ == "__main__":
    # set DATA_DIR before running or modify the DATA_DIR value above
    process_all_images(DATA_DIR, OUT_DIR)