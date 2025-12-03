import json
import os
import cv2
import numpy as np
# --- FIX: Set backend to Agg before importing pyplot to prevent GUI crashes ---
import matplotlib
matplotlib.use('Agg') 
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time

"""
MILESTONE 2: PROJECT REQUIREMENTS COMPLIANCE CHECKLIST
------------------------------------------------------
1. Similarity Computation: 
   - Implemented in `_procrustes_distance()`.
   - Uses Procrustes Analysis (Classical CV) to compute Mean Squared Error 
     after optimal rotation alignment.
   
2. Thresholding and Ranking:
   - Implemented in `compute_all_matches()`.
   - Filters matches > MATCH_THRESHOLD (0.15).
   - Ranks candidates by score (ascending).
   - Handles one-to-many ambiguity by storing top 5 candidates per edge.

3. Visualization of Matches:
   - Implemented in `visualize_matches()`.
   - Generates overlay plots of the Query Edge vs Matched Edge (aligned).

4. Documentation & Demo:
   - This script runs as a standalone demo via `if __name__ == "__main__":`.
   - Outputs ranked CSVs and visual PNGs.

5. Constraints:
   - No Machine Learning used (Geometric Algebra only).
   - No Scipy (removed in favor of Numpy).
"""

# CONFIGURATION
INPUT_METADATA = Path("puzzle_output/pieces_metadata.json")
OUTPUT_DIR = Path("puzzle_output/milestone2")
VIS_DIR = OUTPUT_DIR / "matches_vis"
NUM_RESAMPLE_POINTS = 50  # Number of points to normalize every edge to
MATCH_THRESHOLD = 0.15    # Lower is better (0.0 is perfect match). 
TOP_K = 5                 # Number of candidates to save

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

class PuzzleMatcher:
    def __init__(self, metadata_path):
        with open(metadata_path, 'r') as f:
            self.data = json.load(f)
        
        self.edges_db = []
        self._preprocess_all_edges()

    def _resample_curve(self, points, num_points=NUM_RESAMPLE_POINTS):
        """
        Resamples a curve (list of points) to a fixed number of equidistant points
        using Linear Interpolation (Numpy only). 
        Essential for comparing edges of different pixel lengths.
        """
        points = np.array(points)
        if len(points) < 2:
            return np.zeros((num_points, 2))

        # Calculate cumulative distance along the curve
        # np.diff computes difference between consecutive points
        dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
        cum_dist = np.insert(np.cumsum(dists), 0, 0)
        total_len = cum_dist[-1]
        
        if total_len == 0:
            return np.resize(points, (num_points, 2))

        # Create equidistant target distances
        target_dists = np.linspace(0, total_len, num_points)
        
        # Interpolate x and y separately using numpy (removes scipy dependency)
        new_x = np.interp(target_dists, cum_dist, points[:, 0])
        new_y = np.interp(target_dists, cum_dist, points[:, 1])
        
        new_points = np.column_stack((new_x, new_y))
        return new_points

    def _normalize_for_matching(self, points):
        """
        Centers the edge at the origin and scales it to unit length.
        This makes the descriptor translation and scale invariant.
        """
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # Scale by RMS distance from centroid
        scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        if scale < 1e-6: scale = 1.0
        
        return centered / scale

    def _preprocess_all_edges(self):
        """Extracts, resamples, and indexes all edges from the metadata."""
        print("Preprocessing edges...")
        if not self.data:
            print("Warning: Metadata is empty.")
            return

        for piece in self.data:
            piece_id = piece['id']
            # We assume 'edges' is a list of lists of points [ [x,y]... ]
            # M1 output format: piece['edges'] -> list of 4 contours
            for edge_idx, edge_points in enumerate(piece['edges']):
                if len(edge_points) < 5: continue # Skip noise
                
                # 1. Resample to fixed N
                resampled = self._resample_curve(edge_points)
                
                # 2. Identify if it's a straight border (flat edge)
                # Simple metric: distance from start to end vs cumulative length
                d_euclidean = np.linalg.norm(resampled[0] - resampled[-1])
                d_path = np.sum(np.sqrt(np.sum(np.diff(resampled, axis=0)**2, axis=1)))
                tortuosity = d_path / (d_euclidean + 1e-6)
                
                is_border = tortuosity < 1.05 # Threshold for flat edges

                # OPTIMIZATION: Pre-calculate normalized form here
                # This avoids recalculating it N^2 times in the loop
                normalized = self._normalize_for_matching(resampled)
                
                self.edges_db.append({
                    "unique_id": f"{piece_id}_e{edge_idx}",
                    "piece_id": piece_id,
                    "edge_idx": edge_idx,
                    "raw_points": np.array(edge_points),
                    "resampled": resampled,
                    "normalized": normalized, # Cached descriptor
                    "is_border": is_border
                })
        print(f"Loaded {len(self.edges_db)} edges. (Border edges flagged)")

    def _procrustes_distance(self, shape1, shape2):
        """
        Computes the Procrustes distance (similarity) between two shapes.
        It finds the optimal rotation to align shape2 to shape1 and calculates MSE.
        
        shape1, shape2: (N, 2) arrays, already centered and scaled.
        """
        # Optimal rotation matrix using SVD (Singular Value Decomposition)
        # H = U * S * Vt
        H = np.dot(shape1.T, shape2)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(U, Vt)
        
        # Rotate shape2 to match shape1
        shape2_rotated = np.dot(shape2, R)
        
        # Compute squared error (MSE)
        diff = shape1 - shape2_rotated
        mse = np.mean(np.sum(diff**2, axis=1))
        return mse

    def compute_all_matches(self):
        """
        Compares every edge against every other edge.
        Returns a list of match results.
        """
        print(f"Computing similarity matrix for {len(self.edges_db)} edges...")
        print("This may take a while depending on dataset size. Progress will be updated below.")
        
        results = []
        n = len(self.edges_db)
        start_time = time.time()
        
        for i in range(n):
            # Progress Logging
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (n - i) / rate if rate > 0 else 0
                print(f"Processing edge {i}/{n} ({i/n*100:.1f}%) - Est. remaining: {remaining/60:.1f} min")

            edge_A = self.edges_db[i]
            if edge_A['is_border']: continue # Skip border matching for now
            
            # Use pre-calculated descriptor
            desc_A = edge_A['normalized']
            
            matches_for_A = []
            
            for j in range(n):
                if i == j: continue # Don't match self
                
                edge_B = self.edges_db[j]
                if edge_B['piece_id'] == edge_A['piece_id']: continue # Don't match same piece
                if edge_B['is_border']: continue
                
                # OPTIMIZATION: Use pre-calculated descriptor reversed
                # Normalizing the reversed points is equivalent to reversing the normalized points
                # because centroid and scale are order-independent.
                # This slicing [::-1] is much faster than calling _normalize_for_matching again.
                desc_B_reversed = edge_B['normalized'][::-1]
                
                score = self._procrustes_distance(desc_A, desc_B_reversed)
                
                if score < MATCH_THRESHOLD:
                    matches_for_A.append({
                        "match_id": edge_B['unique_id'],
                        "score": float(score),
                        "piece_A": edge_A['piece_id'],
                        "piece_B": edge_B['piece_id'],
                        "raw_B": edge_B['resampled']
                    })
            
            # Sort best matches for A
            matches_for_A.sort(key=lambda x: x['score'])
            
            # Store Top K (Handles One-to-Many ambiguity)
            top_matches = matches_for_A[:TOP_K]
            if top_matches:
                results.append({
                    "query_edge": edge_A['unique_id'],
                    "candidates": top_matches,
                    "raw_A": edge_A['resampled']
                })
                
        total_time = time.time() - start_time
        print(f"Similarity computation finished in {total_time:.1f} seconds.")
        return results

    def visualize_matches(self, results):
        """Generates visual overlays for top matches."""
        print(f"Generating visualizations in {VIS_DIR}...")
        ensure_dir(VIS_DIR)
        
        for res in results:
            query_id = res['query_edge']
            pts_A = res['raw_A']
            
            # Plot top 1 match for clarity in demo
            if not res['candidates']: continue
            
            best = res['candidates'][0]
            pts_B = best['raw_B']
            score = best['score']
            
            # Alignment for visualization
            # We align B to A using the Procrustes logic just for plotting
            norm_A = self._normalize_for_matching(pts_A)
            norm_B_rev = self._normalize_for_matching(pts_B[::-1])
            
            # Calculate rotation again to display them overlaid
            H = np.dot(norm_A.T, norm_B_rev)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(U, Vt)
            norm_B_aligned = np.dot(norm_B_rev, R)
            
            # Ensure a new figure is created for each plot and closed properly
            fig = plt.figure(figsize=(6, 6))
            plt.plot(norm_A[:, 0], norm_A[:, 1], 'b-', linewidth=2, label=f'Edge A ({query_id})')
            plt.plot(norm_B_aligned[:, 0], norm_B_aligned[:, 1], 'r--', linewidth=2, label=f'Edge B ({best["match_id"]})')
            plt.title(f"Match Score: {score:.4f} (Lower is better)")
            plt.legend()
            plt.axis('equal')
            plt.grid(True, alpha=0.3)
            
            out_name = f"{query_id}_vs_{best['match_id']}.png"
            plt.savefig(VIS_DIR / out_name)
            plt.close(fig) # Explicitly close the specific figure

    def run(self):
        results = self.compute_all_matches()
        
        # Save structured results
        out_csv_data = []
        for r in results:
            for c in r['candidates']:
                out_csv_data.append({
                    "Query_Edge": r['query_edge'],
                    "Match_Edge": c['match_id'],
                    "Score": c['score'],
                    "Piece_A": c['piece_A'],
                    "Piece_B": c['piece_B']
                })
        
        df = pd.DataFrame(out_csv_data)
        df.to_csv(OUTPUT_DIR / "matches_ranked.csv", index=False)
        print(f"Saved ranked matches to {OUTPUT_DIR / 'matches_ranked.csv'}")
        
        self.visualize_matches(results)
        print("Milestone 2 Complete.")

if __name__ == "__main__":
    if not INPUT_METADATA.exists():
        print(f"Error: {INPUT_METADATA} not found. Please run Milestone 1 first.")
    else:
        ensure_dir(OUTPUT_DIR)
        matcher = PuzzleMatcher(INPUT_METADATA)
        matcher.run()
