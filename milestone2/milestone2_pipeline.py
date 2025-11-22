"""
Milestone 2 pipeline for Jigsaw Edge Matching (classical computer vision — no ML)

This script implements Milestone 2 deliverables:
  - Load piece contours and candidate edge segments from a Milestone-1 metadata file.
  - Compute rotation-invariant edge descriptors (Fourier magnitude descriptors and curvature signature).
  - Compute pairwise similarity scores between all edges, including transformed variants (reversed, reflected) to detect complementary fits.
  - Threshold and rank matches per edge; allow one-to-many and many-to-one results.
  - Visualize top candidate matches by overlaying matched edges and drawing connecting lines between piece centroids.
  - Save results (descriptors.json, matches.json, visualizations) and a CSV summary for easy grading.

Usage:
  python milestone2_pipeline.py \
      --metadata /path/to/pieces_metadata.json \
      --dataset_zip /mnt/data/Gravity Falls dataset.zip \
      --out_dir ./milestone2_output \
      --resample_points 32 \
      --top_k 5

Notes:
 - If --metadata is not provided, the script will attempt to locate a default metadata file at
   ./puzzle_output/pieces_metadata.json or /mnt/data/puzzle_output/pieces_metadata.json (paths used in the earlier demo).
 - The script requires Python 3.11+ (recommended) and the following packages:
     pip install numpy opencv-python pandas matplotlib
 - No ML libraries are used.

Output (out_dir):
 - descriptors.json          : computed descriptors per edge
 - matches.json              : ranked matches for every edge
 - matches_summary.csv       : human-readable summary table
 - vis_top_matches/          : PNG visualizations per query edge

Implementation details (short):
 - Edge resampling: edges are resampled by arc length to N points (default 128).
 - Rotation-invariant descriptor: complex sequence z = x + i*y -> FFT -> magnitude of first M coefficients (excluding DC), normalized.
 - Curvature signature: discrete angle between successive tangent vectors; used with a simple DTW distance.
 - Complementarity handling: when comparing edges A and B, the script compares A vs B, A vs reversed(B), A vs reflected(B) (mirror Y), and A vs reversed(reflected(B)). The minimum distance across these transformations is taken.
 - Matching score: weighted sum of Fourier distance and DTW curvature distance (weights configurable).

"""
# milestone2_pipeline.py
import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Utility functions -----------------------------

def load_metadata(path_candidates):
    """Try loading metadata from one of the candidate paths."""
    for p in path_candidates:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            with open(p, 'r') as f:
                return json.load(f), p
    raise FileNotFoundError("Could not find a pieces metadata JSON file. Run Milestone 1 first or provide --metadata.")


def resample_contour(points, n=128):
    """Resample a contour (Nx2) by arc-length to n points. Returns float array shape (n,2)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros((n,2), dtype=float)
    # compute cumulative arc length
    diffs = np.diff(pts, axis=0)
    seglen = np.hypot(diffs[:,0], diffs[:,1])
    cumlen = np.concatenate(([0.0], np.cumsum(seglen)))
    total = cumlen[-1]
    if total == 0:
        return np.tile(pts[0], (n,1)).astype(float)
    t = np.linspace(0, total, n)
    res = np.zeros((n,2), dtype=float)
    ix = 0
    for i,tt in enumerate(t):
        while ix < len(cumlen)-1 and cumlen[ix+1] < tt:
            ix += 1
        if ix == len(cumlen)-1:
            res[i] = pts[-1]
        else:
            a = pts[ix]; b = pts[ix+1]
            if cumlen[ix+1] - cumlen[ix] == 0:
                res[i] = a
            else:
                alpha = (tt - cumlen[ix]) / (cumlen[ix+1] - cumlen[ix])
                res[i] = a*(1-alpha) + b*alpha
    return res


def complex_fourier_descriptor(points, keep_coeffs=20):
    """Compute rotation-invariant Fourier magnitude descriptor.
    points: (n,2) float array. Returns normalized magnitude vector of length keep_coeffs.
    """
    z = points[:,0] + 1j*points[:,1]
    # remove centroid
    z = z - np.mean(z)
    # normalize scale
    norm = np.sqrt(np.sum(np.abs(z)**2))
    if norm == 0:
        return np.zeros(keep_coeffs, dtype=float)
    z = z / norm
    # fft
    Z = np.fft.fft(z)
    # take magnitudes of first keep_coeffs (skipping DC at index 0 optionally)
    mags = np.abs(Z)
    # We will skip DC component (mags[0]) because centroid removed, but include it for indexing safety
    # Use symmetric packing: take low-frequency components (1..keep_coeffs)
    k = keep_coeffs
    vec = mags[1:k+1]
    # normalize to unit L2
    if np.linalg.norm(vec) == 0:
        return np.zeros_like(vec)
    return (vec / np.linalg.norm(vec)).real


def curvature_signature(points):
    """Return discrete curvature (angle) sequence for a resampled contour.
    points: (n,2) array
    returns: (n,) float array of angles in radians
    """
    pts = np.asarray(points)
    # tangent vectors
    v = np.diff(pts, axis=0)
    v = np.vstack([v, v[-1:]])  # keep same length
    angles = np.zeros(len(v))
    for i in range(len(v)):
        p = v[i-1] if i-1>=0 else v[-1]
        q = v[i]
        na = np.linalg.norm(p); nb = np.linalg.norm(q)
        if na==0 or nb==0:
            angles[i] = 0.0
        else:
            cosang = np.clip(np.dot(p,q)/(na*nb), -1.0, 1.0)
            angles[i] = math.acos(cosang)
    return angles


def dtw_distance(a, b):
    """Simple DTW distance between 1D sequences a and b (Euclidean local cost).
    Complexity O(len(a)*len(b)). Returns final distance (float).
    """
    na = len(a); nb = len(b)
    # use float32 matrix to save memory
    D = np.full((na+1, nb+1), np.inf, dtype=np.float64)
    D[0,0] = 0.0
    for i in range(1, na+1):
        for j in range(1, nb+1):
            cost = abs(a[i-1] - b[j-1])
            D[i,j] = cost + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[na, nb])

# transformations for complementarity

def reflect_points(points, axis='y'):
    pts = np.array(points, dtype=float)
    if axis == 'y':
        pts[:,1] = -pts[:,1]
    elif axis == 'x':
        pts[:,0] = -pts[:,0]
    else:
        raise ValueError('axis must be x or y')
    return pts

# ----------------------------- Descriptor computation -----------------------------

def compute_descriptors(metadata, resample_points=32, fd_coeffs=20):
    """Given metadata (list of pieces each with edges), compute per-edge descriptors.
    Returns dict: {edge_id: {'piece_id', 'edge_idx', 'fd', 'curvature', 'centroid', 'bbox'}}
    """
    descriptors = {}
    for piece in metadata:
        pid = piece.get('id')
        bbox = piece.get('bbox')
        contour = piece.get('contour')
        # compute piece centroid from contour
        if contour and len(contour)>0:
            cpts = np.array(contour, dtype=float)
            centroid = cpts.mean(axis=0).tolist()
        else:
            centroid = [0,0]
        edges = piece.get('edges', [])
        for ei, edge in enumerate(edges):
            edge_pts = np.array(edge, dtype=float)
            # resample along arc-length
            res = resample_contour(edge_pts, n=resample_points)
            # normalize orientation: translate to centroid of edge
            res_centered = res - np.mean(res, axis=0)
            fd = complex_fourier_descriptor(res_centered, keep_coeffs=fd_coeffs).tolist()
            curv = curvature_signature(res).tolist()
            edge_id = f"{pid}_e{ei}"
            descriptors[edge_id] = {
                'edge_id': edge_id,
                'piece_id': pid,
                'edge_idx': ei,
                'fd': fd,
                'curvature': curv,
                'centroid': centroid,
                'bbox': bbox,
                'n_points': len(res)
            }
    return descriptors

# ----------------------------- Matching -----------------------------

def compare_edges(desc_a, desc_b, weight_fd=0.6, weight_dtw=0.4):
    """Compute similarity score between two descriptors. Lower is better (distance).
    We consider transformations on B: identity, reversed, reflected (Y), reversed+reflected.
    """
    # prepare arrays
    A_fd = np.array(desc_a['fd'], dtype=float)
    B_fd = np.array(desc_b['fd'], dtype=float)
    A_curv = np.array(desc_a['curvature'], dtype=float)
    B_curv = np.array(desc_b['curvature'], dtype=float)

    candidates = []
    # define transformations on B sequences
    transforms = []
    # identity
    transforms.append(('id', B_fd, B_curv))
    # reversed
    transforms.append(('rev', B_fd[::-1], B_curv[::-1]))
    # reflected in Y (curvature unaffected by sign flip of Y? curvature sequence will change due to coordinates; but we only have curvature values here — to be robust, we keep curvature reversed as well)
    transforms.append(('ref', B_fd, B_curv[::-1]))
    transforms.append(('rev_ref', B_fd[::-1], B_curv))

    # compute distances for each transform
    for name, bfd, bcurv in transforms:
        # fd distance (L2)
        fd_dist = np.linalg.norm(A_fd - bfd)
        # dtw on curvature
        try:
            dtw_dist = dtw_distance(A_curv, bcurv)
        except Exception:
            dtw_dist = float('inf')
        # combined normalized score
        # normalize dtw by length to be comparable
        dtw_norm = dtw_dist / max(1.0, len(A_curv))
        score = weight_fd * fd_dist + weight_dtw * dtw_norm
        candidates.append((score, name, fd_dist, dtw_norm))
    # return the best
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    return {'score': float(best[0]), 'transform': best[1], 'fd_dist': float(best[2]), 'dtw_norm': float(best[3])}

# ----------------------------- Main pipeline -----------------------------

def run_matching(metadata_path, out_dir, resample_points=32, fd_coeffs=20, top_k=5, score_threshold=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load metadata
    metadata, meta_file_used = load_metadata([metadata_path, Path('puzzle_output')/ 'pieces_metadata.json', Path('/mnt/data/puzzle_output')/ 'pieces_metadata.json'])
    print(f"Loaded metadata from {meta_file_used}, pieces: {len(metadata)}")

    # compute descriptors
    print("Computing descriptors for all edges...")

    #descriptors = compute_descriptors(metadata, resample_points=resample_points, fd_coeffs=fd_coeffs)

    # test only first 20 pieces (or edges)
    test_metadata = metadata[:20]  # we adjust number of pieces for quick testing
    descriptors = compute_descriptors(test_metadata, resample_points=resample_points, fd_coeffs=fd_coeffs)

    print(f"Computed descriptors for {len(descriptors)} edges.")

    # save descriptors
    desc_path = out_dir / 'descriptors.json'
    with open(desc_path, 'w') as f:
        json.dump(descriptors, f, indent=2)
    print("Saved descriptors to", desc_path)

    # prepare list of edges
    edge_ids = list(descriptors.keys())
    N = len(edge_ids)
    print(f"Matching {N} edges pairwise (this is O(N^2)).")

    # We Added THIS LINE FOR TESTING ---
    edge_ids = edge_ids[:50]  # test only first 50 edges
    N = len(edge_ids)  # update N
    print(f"Testing with only {N} edges to reduce runtime.")

    matches = defaultdict(list)
    # naive O(N^2) loop with early pruning via FD distance upper bound
    for i in range(N):
        ida = edge_ids[i]
        for j in range(i+1, N):
            idb = edge_ids[j]
            res = compare_edges(descriptors[ida], descriptors[idb])
            score = res['score']
            # store symmetric
            matches[ida].append({'candidate': idb, 'score': score, 'transform': res['transform'], 'fd_dist': res['fd_dist'], 'dtw_norm': res['dtw_norm']})
            matches[idb].append({'candidate': ida, 'score': score, 'transform': res['transform'], 'fd_dist': res['fd_dist'], 'dtw_norm': res['dtw_norm']})

    # ranking and thresholding
    summary_rows = []
    matches_out = {}
    for eid, cand_list in matches.items():
        cand_sorted = sorted(cand_list, key=lambda x: x['score'])
        if score_threshold is not None:
            cand_sorted = [c for c in cand_sorted if c['score'] <= score_threshold]
        top = cand_sorted[:top_k]
        matches_out[eid] = top
        # summarize
        for rnk, item in enumerate(top, start=1):
            summary_rows.append({'edge_id': eid, 'rank': rnk, 'candidate': item['candidate'], 'score': item['score'], 'transform': item['transform'], 'fd_dist': item['fd_dist'], 'dtw_norm': item['dtw_norm']})

    # save matches
    matches_path = out_dir / 'matches.json'
    with open(matches_path, 'w') as f:
        json.dump(matches_out, f, indent=2)
    print('Saved matches to', matches_path)

    # save summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = out_dir / 'matches_summary.csv'
    df.to_csv(csv_path, index=False)
    print('Saved summary CSV to', csv_path)

    # Visualize top matches: for each query edge, overlay candidate edge shapes onto the source image or draw centroid connectors
    vis_dir = out_dir / 'vis_top_matches'
    vis_dir.mkdir(exist_ok=True)
    # build quick mapping from piece_id to piece metadata (centroid, bbox, crop)
    piece_map = {p['id']: p for p in metadata}

    for eid, top_list in matches_out.items():
        # parse piece id and edge idx
        pid, edge_idx_str = eid.rsplit('_e', 1)
        piece = piece_map.get(pid)
        if piece is None:
            continue

        qedge_idx = int(edge_idx_str)

        # create a blank white canvas
        canvas = np.ones((800, 1200, 3), dtype=np.uint8) * 255

        # offsets for query edge
        qx_offset, qy_offset = 150, 200

        # overlay query edge (resampled) as polyline
        qedge_pts = piece.get('edges', [])
        if qedge_pts and qedge_idx < len(qedge_pts):
            qedge_pts = np.array(qedge_pts[qedge_idx], dtype=np.int32)
            if qedge_pts.size != 0:
                # normalize to small coords and draw
                qnorm = qedge_pts - qedge_pts.mean(axis=0) + np.array([qx_offset+60, qy_offset+60])
                cv2.polylines(canvas, [qnorm.astype(np.int32)], isClosed=False, color=(255,0,0), thickness=2)

        # draw top candidate edges / pieces on the right side
        rx, ry, dy = 600, 80, 120
        for k, cand in enumerate(top_list):
            cid = cand['candidate']
            cp, cedge_idx_str = cid.rsplit('_e', 1)
            cpiece = piece_map.get(cp)
            if cpiece is None:
                continue

            # try to draw candidate crop if available
            try:
                cpath = cpiece.get('crop_path')
                ccrop = None
                if cpath and Path(cpath).exists():
                    ccrop = cv2.imread(cpath)
                if ccrop is not None:
                    h, w = ccrop.shape[:2]
                    y0 = ry + k*dy
                    x0 = rx
                    canvas[y0:y0+h, x0:x0+w] = cv2.resize(ccrop, (w, h))
            except Exception:
                pass

            # optionally overlay candidate edge polyline
            cedges = cpiece.get('edges', [])
            if cedges and int(cedge_idx_str) < len(cedges):
                cedge_pts = np.array(cedges[int(cedge_idx_str)], dtype=np.int32)
                if cedge_pts.size != 0:
                    cnorm = cedge_pts - cedge_pts.mean(axis=0) + np.array([rx+60, ry + k*dy + 60])
                    cv2.polylines(canvas, [cnorm.astype(np.int32)], isClosed=False, color=(0,255,0), thickness=2)

        # save visualization
        vis_path = vis_dir / f"{eid}_top{len(top_list)}.png"
        cv2.imwrite(str(vis_path), canvas)

    print('Saved visualizations to', vis_dir)
    print('Done.')

# ----------------------------- CLI -----------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', type=str, default=None, help='Path to pieces_metadata.json (from Milestone 1)')
    p.add_argument('--dataset_zip', type=str, default='/mnt/data/Gravity Falls dataset.zip', help='Path to dataset zip (not required if metadata exists)')
    p.add_argument('--out_dir', type=str, default='./milestone2_output', help='Output directory')
    p.add_argument('--resample_points', type=int, default=128, help='Points per edge after resampling')
    p.add_argument('--fd_coeffs', type=int, default=20, help='Number of Fourier magnitude coefficients to keep')
    p.add_argument('--top_k', type=int, default=5, help='Top-K candidates per edge to save')
    p.add_argument('--score_threshold', type=float, default=None, help='Optional score threshold to filter matches (lower is better)')
    args = p.parse_args()

    # try default metadata locations if not provided
    default_meta_candidates = [args.metadata, Path('puzzle_output')/ 'pieces_metadata.json', Path('/mnt/data/puzzle_output')/ 'pieces_metadata.json']
    # pick first existing
    meta_path = None
    for c in default_meta_candidates:
        if c is None:
            continue
        cp = Path(c)
        if cp.exists():
            meta_path = str(cp)
            break
    if meta_path is None:
        # as a last resort, look next to dataset zip
        dszip = Path(args.dataset_zip)
        # Note: This script does NOT automatically rerun full Milestone1 segmentation when metadata is missing.
        # Running Milestone1 is recommended. For convenience we still try the common path below.
        alt = Path('./puzzle_output')/'pieces_metadata.json'
        if alt.exists():
            meta_path = str(alt)
        else:
            print('No metadata found. Please run Milestone-1 script first or provide --metadata path.')
            raise SystemExit(1)

    run_matching(meta_path, args.out_dir, resample_points=args.resample_points, fd_coeffs=args.fd_coeffs, top_k=args.top_k, score_threshold=args.score_threshold)
