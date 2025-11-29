# milestone1_pipeline.py
import os, zipfile, json, math, shutil
from glob import glob
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG
DATA_DIR = r"C:\Users\belal\Desktop\Fall 2026\computer vision\project\JigsawCV\Gravity Falls\correct"
OUT_DIR = Path("puzzle_output")
MAX_DM = 1200
MIN_AREA_RATIO = 0.002

# UTIL FUNCTIONS (same as before)
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
    if window <= 1:
        return np.asarray(contour, dtype=np.float32)
    arr = np.asarray(contour, dtype=np.float32)
    n = len(arr)
    pad = window // 2
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
# UPDATED DEBUG VISUALIZATION FOR NEW PIPELINE ORDER
# =============================================================================

def create_debug_visualization(original, clahe_enhanced, denoised, gamma_corrected, gray, binary, contours, output_path):
    """
    Visualization showing the NEW pipeline order
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original color image
    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Color Image')
    axes[0, 0].axis('off')
    
    # CLAHE enhanced color image (FIRST STEP)
    axes[0, 1].imshow(cv2.cvtColor(clahe_enhanced, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('CLAHE Enhanced (Step 1)')
    axes[0, 1].axis('off')
    
    # Denoised color image (SECOND STEP)
    axes[0, 2].imshow(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Denoised (Step 2)')
    axes[0, 2].axis('off')
    
    # Gamma corrected color image (THIRD STEP)
    axes[0, 3].imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
    axes[0, 3].set_title('Gamma Corrected (Step 3)')
    axes[0, 3].axis('off')
    
    # Grayscale image (FOURTH STEP)
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title('Grayscale (Step 4)')
    axes[1, 0].axis('off')
    
    # Binary mask (FIFTH STEP)
    axes[1, 1].imshow(binary, cmap='gray')
    axes[1, 1].set_title('Binary Mask (Step 5)')
    axes[1, 1].axis('off')
    
    # Contours on CLAHE enhanced image (as requested)
    contour_vis = clahe_enhanced.copy()
    cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
    axes[1, 2].imshow(cv2.cvtColor(contour_vis, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title(f'Contours on CLAHE Image\n{len(contours)} contours')
    axes[1, 2].axis('off')
    
    # Individual pieces (color coded)
    piece_vis = np.zeros_like(original)
    for i, contour in enumerate(contours):
        color = plt.cm.tab10(i % 10)
        color_bgr = [int(c * 255) for c in color[:3]][::-1]
        cv2.drawContours(piece_vis, [contour], -1, color_bgr, -1)
    axes[1, 3].imshow(cv2.cvtColor(piece_vis, cv2.COLOR_BGR2RGB))
    axes[1, 3].set_title('Segmented Pieces')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_edges(contour, edges, output_path):
    plt.figure(figsize=(10, 8))
    contour_array = np.array(contour)
    plt.plot(contour_array[:, 0], -contour_array[:, 1], 'k-', alpha=0.3, label='Full Contour')
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
# IMAGE ENHANCEMENT FUNCTIONS - SAME AS BEFORE
# =============================================================================

def gamma_correction_color(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(image, table)
    return gamma_corrected

def enhance_contrast_color(image, clip_limit=3.0):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def remove_noise_color(image, method='median'):
    if method == 'median':
        denoised = cv2.medianBlur(image, 3)
    elif method == 'nlm':
        denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, 
                                                 templateWindowSize=7, searchWindowSize=21)
    else:
        denoised = image
    return denoised

# =============================================================================
# UPDATED MAIN PROCESSING FUNCTION WITH YOUR PIPELINE ORDER
# =============================================================================

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
        
        # =============================================================================
        # YOUR PIPELINE ORDER: Color → CLAHE → Denoising → Gamma → Grayscale → Thresholding
        # =============================================================================
        
        # Step 1: CLAHE Enhancement (FIRST as requested)
        clahe_enhanced = enhance_contrast_color(img_small, clip_limit=3.0)
        
        # Step 2: Denoising (SECOND)
        denoised = remove_noise_color(clahe_enhanced, method='median')
        
        # Step 3: Gamma Correction (THIRD)
        gamma_corrected = gamma_correction_color(denoised, gamma=1.8)
        
        # Step 4: Convert to Grayscale (FOURTH)
        gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
        
        # Step 5: Thresholding (FIFTH)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Step 6: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # =============================================================================
        # CONTOUR DETECTION ON FINAL BINARY IMAGE (from the full pipeline)
        # =============================================================================
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
        ensure_dir(base / "vis"); ensure_dir(base / "debug")
        
        # =============================================================================
        # UPDATED DEBUG VISUALIZATION SHOWING CONTOURS ON CLAHE IMAGE
        # =============================================================================
        debug_path = str(base / "debug" / f"{Path(img_path).stem}_pipeline.png")
        create_debug_visualization(
            original=img_small,
            clahe_enhanced=clahe_enhanced,
            denoised=denoised,
            gamma_corrected=gamma_corrected,
            gray=gray,
            binary=clean,
            contours=[p[1] for p in pieces],
            output_path=debug_path
        )
        print(f"   💾 Saved debug visualization: {debug_path}")
        
        # Save contour visualization ON CLAHE IMAGE (as requested)
        contour_vis = clahe_enhanced.copy()
        if pieces:
            cv2.drawContours(contour_vis, [p[1] for p in pieces], -1, (0,255,0), 2)
        cv2.imwrite(str(base / "vis" / f"{Path(img_path).stem}_contours.png"), contour_vis)
        
        # Process each piece
        for i,(area,cnt) in enumerate(pieces):
            uid = f"{Path(img_path).stem}_piece{i}"
            cnt_orig = (cnt.astype(np.float32) / scale).astype(int) if scale!=1.0 else cnt
            cnt_norm = normalize_contour(cnt_orig)
            cnt_np = np.array(cnt_norm, dtype=np.float32)
            cnt_np_smooth = smooth_contour(cnt_np, window=5)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt_orig], -1, 255, -1)
            x,y,wid,ht = cv2.boundingRect(cnt_orig)
            crop = cv2.bitwise_and(img[y:y+ht, x:x+wid], img[y:y+ht, x:x+wid], mask=mask[y:y+ht, x:x+wid])
            mask_crop = mask[y:y+ht, x:x+wid]
            mask_path = str(base / "masks" / f"{uid}_mask.png")
            crop_path = str(base / "crops" / f"{uid}_crop.png")
            cv2.imwrite(mask_path, mask_crop)
            cv2.imwrite(crop_path, crop)
            
            edges = split_contour_by_curvature(cnt_np_smooth, k=6, factor=1.7, min_seg_len=25)
            
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
    
    # Save metadata
    with open(Path(out_dir)/"pieces_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    pd.DataFrame(summary).to_csv(Path(out_dir)/"pieces_summary.csv", index=False)
    print("Done. outputs in:", out_dir)
    print(f"🎉 Total pieces processed: {len(metadata)}")

if __name__ == "__main__":
    process_all_images(DATA_DIR, OUT_DIR)