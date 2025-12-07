"""
MILESTONE 2: PUZZLE EDGE MATCHING
Takes output from Simple Milestone 1 and finds matching edges.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import time
import argparse
import sys

class PuzzleEdgeMatcher:
    def __init__(self, config):
        self.config = config
        self.edges_db = []
        
        # Create output directories
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['vis_dir']).mkdir(parents=True, exist_ok=True)
    
    def load_metadata(self, metadata_path):
        print(f"📂 Loading metadata from: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        if 'pieces' not in data:
            raise ValueError("Invalid metadata: 'pieces' key not found")
        
        self.metadata = data
        self.pieces = data['pieces']
        self.puzzle_type = data.get('puzzle_type', 'unknown')
        
        print(f"✅ Loaded {len(self.pieces)} puzzle pieces ({self.puzzle_type})")
        return True
    
    def preprocess_edges(self):
        print("\n🔄 Preprocessing edges...")
        
        total_edges = 0
        for piece in self.pieces:
            piece_id = piece['id']
            edges = piece.get('edges', [])
            
            # We expect 4 edges per piece for rectangular puzzles
            for edge_idx, edge_points in enumerate(edges):
                # Resample to fixed number of points
                resampled = self._resample_edge(edge_points, self.config['num_points'])
                
                # Detect if it's a border edge
                is_border = self._is_border_edge(resampled)
                
                # Normalize
                normalized = self._normalize_edge(resampled)
                
                self.edges_db.append({
                    'id': f"{piece_id}_e{edge_idx}",
                    'piece_id': piece_id,
                    'edge_idx': edge_idx,
                    'points': np.array(edge_points, dtype=np.float32),
                    'resampled': resampled,
                    'normalized': normalized,
                    'is_border': is_border
                })
                total_edges += 1
        
        border_count = sum(1 for e in self.edges_db if e['is_border'])
        print(f"✅ Processed {total_edges} edges ({border_count} border edges)")
        return True
    
    def _resample_edge(self, points, num_points):
        if len(points) < 2:
            return np.zeros((num_points, 2))
        
        points = np.array(points, dtype=np.float32)
        diffs = np.diff(points, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cum_length = np.insert(np.cumsum(seg_lengths), 0, 0)
        total_length = cum_length[-1]
        
        if total_length == 0:
            return np.zeros((num_points, 2))
        
        target_dists = np.linspace(0, total_length, num_points)
        resampled_x = np.interp(target_dists, cum_length, points[:, 0])
        resampled_y = np.interp(target_dists, cum_length, points[:, 1])
        
        return np.column_stack((resampled_x, resampled_y))
    
    def _is_border_edge(self, points, threshold=1.05):
        if len(points) < 2:
            return True
        
        path_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        straight_dist = np.linalg.norm(points[0] - points[-1])
        
        if straight_dist == 0:
            return True
        
        tortuosity = path_length / straight_dist
        return tortuosity < threshold
    
    def _normalize_edge(self, points):
        points = np.array(points, dtype=np.float32)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        scale = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        if scale < 1e-6:
            return centered
        
        return centered / scale
    
    def compute_procrustes_distance(self, edge1, edge2):
        min_len = min(len(edge1), len(edge2))
        if min_len < 10:
            return float('inf')
        
        A = edge1[:min_len]
        B = edge2[:min_len]
        
        try:
            H = A.T @ B
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt
            
            B_aligned = B @ R.T
            mse = np.mean(np.sum((A - B_aligned)**2, axis=1))
            return mse
            
        except np.linalg.LinAlgError:
            return float('inf')
    
    def find_all_matches(self):
        print(f"\n🔍 Finding matches among {len(self.edges_db)} edges...")
        
        # Only match non-border edges
        candidate_edges = [e for e in self.edges_db if not e['is_border']]
        print(f"   Matching {len(candidate_edges)} non-border edges")
        
        all_matches = []
        start_time = time.time()
        
        for i, query_edge in enumerate(candidate_edges):
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (i + 1)) * (len(candidate_edges) - i - 1) if i > 0 else 0
                print(f"   Progress: {i+1}/{len(candidate_edges)} edges "
                      f"({(i+1)/len(candidate_edges)*100:.1f}%)")
            
            edge_matches = []
            
            for target_edge in candidate_edges:
                if query_edge['id'] == target_edge['id']:
                    continue
                
                if query_edge['piece_id'] == target_edge['piece_id']:
                    continue
                
                distance = self.compute_procrustes_distance(
                    query_edge['normalized'],
                    target_edge['normalized'][::-1]
                )
                
                if distance < self.config['match_threshold']:
                    edge_matches.append({
                        'match_id': target_edge['id'],
                        'piece_id': target_edge['piece_id'],
                        'edge_idx': target_edge['edge_idx'],
                        'distance': float(distance)
                    })
            
            edge_matches.sort(key=lambda x: x['distance'])
            top_matches = edge_matches[:self.config['top_k']]
            
            if top_matches:
                all_matches.append({
                    'query_edge_id': query_edge['id'],
                    'query_piece_id': query_edge['piece_id'],
                    'query_edge_idx': query_edge['edge_idx'],
                    'candidates': top_matches
                })
        
        elapsed = time.time() - start_time
        print(f"\n✅ Match computation completed in {elapsed:.1f} seconds")
        print(f"   Found {len(all_matches)} query edges with matches")
        
        total_candidates = sum(len(m['candidates']) for m in all_matches)
        print(f"   Total candidate matches: {total_candidates}")
        
        return all_matches
    
    def save_results(self, matches):
        print("\n💾 Saving results...")
        
        # Save CSV
        csv_data = []
        for match in matches:
            for rank, candidate in enumerate(match['candidates'], 1):
                csv_data.append({
                    'Query_Edge_ID': match['query_edge_id'],
                    'Query_Piece_ID': match['query_piece_id'],
                    'Query_Edge_Index': match['query_edge_idx'],
                    'Match_Edge_ID': candidate['match_id'],
                    'Match_Piece_ID': candidate['piece_id'],
                    'Match_Edge_Index': candidate['edge_idx'],
                    'Procrustes_Distance': candidate['distance'],
                    'Rank': rank
                })
        
        df = pd.DataFrame(csv_data)
        csv_path = Path(self.config['output_dir']) / "matches_ranked.csv"
        df.to_csv(csv_path, index=False)
        print(f"   ✅ CSV saved: {csv_path} ({len(df)} matches)")
        
        # Save JSON
        json_data = {
            'puzzle_type': self.puzzle_type,
            'match_threshold': self.config['match_threshold'],
            'total_edges': len(self.edges_db),
            'non_border_edges': len([e for e in self.edges_db if not e['is_border']]),
            'matches_found': len(matches),
            'matches': matches
        }
        
        json_path = Path(self.config['output_dir']) / "matches_detailed.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"   ✅ JSON saved: {json_path}")
        
        # Save summary
        if len(df) > 0:
            summary = {
                'best_match_distance': float(df['Procrustes_Distance'].min()),
                'worst_match_distance': float(df['Procrustes_Distance'].max()),
                'average_distance': float(df['Procrustes_Distance'].mean()),
                'median_distance': float(df['Procrustes_Distance'].median()),
                'unique_query_edges': df['Query_Edge_ID'].nunique(),
                'unique_match_edges': df['Match_Edge_ID'].nunique()
            }
            
            summary_path = Path(self.config['output_dir']) / "match_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"   ✅ Summary saved: {summary_path}")
            
            print("\n" + "=" * 50)
            print("MATCHING SUMMARY")
            print("=" * 50)
            for key, value in summary.items():
                print(f"  {key}: {value}")
        
        return df
    
    def visualize_matches(self, matches, max_plots=10):
        print(f"\n🎨 Creating visualizations...")
        
        vis_count = min(max_plots, len(matches))
        
        for i, match in enumerate(matches[:vis_count]):
            if not match['candidates']:
                continue
            
            query_edge = next(e for e in self.edges_db if e['id'] == match['query_edge_id'])
            best_match = match['candidates'][0]
            match_edge = next(e for e in self.edges_db if e['id'] == best_match['match_id'])
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot 1: Original edges
            axes[0].plot(query_edge['resampled'][:, 0], query_edge['resampled'][:, 1],
                        'b-', linewidth=3, label='Query Edge')
            axes[0].plot(match_edge['resampled'][:, 0], match_edge['resampled'][:, 1],
                        'r--', linewidth=2, label='Best Match')
            axes[0].set_title(f'{match["query_edge_id"]} vs {best_match["match_id"]}')
            axes[0].legend()
            axes[0].axis('equal')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Normalized and aligned
            query_norm = query_edge['normalized']
            match_norm_rev = match_edge['normalized'][::-1]
            
            H = query_norm.T @ match_norm_rev
            U, S, Vt = np.linalg.svd(H)
            R = U @ Vt
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt
            match_aligned = match_norm_rev @ R.T
            
            axes[1].plot(query_norm[:, 0], query_norm[:, 1],
                        'b-', linewidth=3, label='Query (normalized)')
            axes[1].plot(match_aligned[:, 0], match_aligned[:, 1],
                        'r--', linewidth=2, label='Match (aligned)')
            axes[1].set_title(f'Aligned Comparison\nDistance: {best_match["distance"]:.4f}')
            axes[1].legend()
            axes[1].axis('equal')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f"match_{i:03d}_{match['query_edge_id']}.png"
            plt.savefig(Path(self.config['vis_dir']) / filename, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Saved {vis_count} visualizations to {self.config['vis_dir']}")
    
    def run(self, metadata_path):
        print("=" * 60)
        print("MILESTONE 2: PUZZLE EDGE MATCHING")
        print("=" * 60)
        
        # Step 1: Load data
        if not self.load_metadata(metadata_path):
            return False
        
        # Step 2: Preprocess edges
        self.preprocess_edges()
        
        # Step 3: Find matches
        matches = self.find_all_matches()
        
        if not matches:
            print("\n⚠️  No matches found!")
            print(f"   Try increasing threshold (current: {self.config['match_threshold']})")
            print("   Or check if edges were detected correctly in Milestone 1")
            return False
        
        # Step 4: Save results
        matches_df = self.save_results(matches)
        
        # Step 5: Create visualizations
        self.visualize_matches(matches, max_plots=self.config['max_visualizations'])
        
        print("\n" + "=" * 60)
        print("🎉 MILESTONE 2 COMPLETED!")
        print("=" * 60)
        print(f"\nOutputs in: {self.config['output_dir']}")
        
        return True

def main():
    # YOUR EXACT PATHS
    BASE_PROJECT = Path(r"C:\Users\belal\Desktop\Fall 2026\computer vision\project\JigsawCV")
    SIMPLE_M1_OUTPUT = BASE_PROJECT / "simple_milestone1_output"
    MILESTONE2_OUTPUT = BASE_PROJECT / "milestone2_output"
    
    parser = argparse.ArgumentParser(description="Milestone 2: Puzzle Edge Matching")
    
    # List available JSON files from simple_milestone1_output
    available_json = []
    if SIMPLE_M1_OUTPUT.exists():
        available_json = list(SIMPLE_M1_OUTPUT.glob("*.json"))
    
    if available_json:
        default_input = str(available_json[0])
        print("Available JSON files from Milestone 1:")
        for i, json_file in enumerate(available_json[:5]):
            print(f"  [{i+1}] {json_file.name}")
        print()
    else:
        default_input = str(SIMPLE_M1_OUTPUT / "pieces_2x2.json")
    
    parser.add_argument("--input", type=str, default=default_input,
                       help=f"Path to Milestone 1 JSON output (default: {default_input})")
    parser.add_argument("--output_dir", type=str, default=str(MILESTONE2_OUTPUT),
                       help=f"Directory for output files (default: {MILESTONE2_OUTPUT})")
    parser.add_argument("--threshold", type=float, default=0.15,
                       help="Match threshold (lower = stricter, default: 0.15)")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Top K matches per edge (default: 5)")
    parser.add_argument("--max_vis", type=int, default=10,
                       help="Max visualizations to create (default: 10)")
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'input_json': args.input,
        'output_dir': args.output_dir,
        'vis_dir': Path(args.output_dir) / "visualizations",
        'match_threshold': args.threshold,
        'top_k': args.top_k,
        'num_points': 50,
        'max_visualizations': args.max_vis
    }
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        print("\nPlease run Simple Milestone 1 first:")
        print("  python simple_milestone1.py --puzzle_type 2x2")
        return 1
    
    # Create and run matcher
    matcher = PuzzleEdgeMatcher(config)
    success = matcher.run(args.input)
    
    if not success:
        print("\n❌ Milestone 2 failed. Check errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())