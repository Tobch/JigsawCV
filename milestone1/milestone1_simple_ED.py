import cv2
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

class SimpleEdgeDetector:
    """
    Simplified Milestone 1: Detects rectangular/square edges from puzzle piece images.
    Focuses only on finding 4 edges per piece for 2x2, 4x4, or 8x8 puzzles.
    """
    
    def __init__(self, input_folder, output_json_path):
        """
        Initializes the SimpleEdgeDetector with input folder and output path.
        Sets up paths and ensures output directory exists.
        """
        self.input_folder = Path(input_folder)
        self.output_json_path = Path(output_json_path)
        self.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        
    def detect_edges_in_image(self, image_path):
        """
        Main processing pipeline for a single image.
        Returns list of pieces with their 4 edges.
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"    Could not load {image_path.name}")
            return []
        
        # Resize if too large for processing
        max_dim = 800
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pieces = []
        min_area = img.shape[0] * img.shape[1] * 0.001  # At least 0.1% of image
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Simplify contour to polygon (for rectangular pieces)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # For rectangular pieces, we expect 4 vertices
            if len(approx) < 4:
                # If not 4 vertices, use bounding box corners
                corners = np.array([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ], dtype=np.float32)
            else:
                # Use approximated polygon points
                corners = approx.reshape(-1, 2).astype(np.float32)
                # Ensure we have exactly 4 points
                if len(corners) > 4:
                    # Take 4 most distant points for rectangular shape
                    corners = self._select_four_corners(corners)
            
            # Convert to list format
            corners_list = corners.tolist()
            
            # Split into 4 edges (connect consecutive corners)
            edges = []
            for j in range(4):
                edge = [corners_list[j], corners_list[(j + 1) % 4]]
                # Add some intermediate points for better matching
                edge_with_points = self._add_edge_points(edge[0], edge[1])
                edges.append(edge_with_points)
            
            piece_data = {
                "id": f"{image_path.stem}_piece{i}",
                "image": image_path.name,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": float(area),
                "corners": corners_list,
                "edges": edges  # 4 edges per piece
            }
            pieces.append(piece_data)
        
        return pieces
    
    def _select_four_corners(self, points):
        """Select 4 corners from polygon points for rectangular shape."""
        points = np.array(points)
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        
        # Create 4 corners
        corners = np.array([
            [x_min, y_min],  # Top-left
            [x_max, y_min],  # Top-right
            [x_max, y_max],  # Bottom-right
            [x_min, y_max]   # Bottom-left
        ], dtype=np.float32)
        
        return corners
    
    def _add_edge_points(self, point1, point2, num_points=20):
        """Add intermediate points along an edge for better matching."""
        p1 = np.array(point1)
        p2 = np.array(point2)
        
        points = []
        for t in np.linspace(0, 1, num_points):
            # Linear interpolation
            point = p1 + t * (p2 - p1)
            # Add slight random variation to simulate puzzle edge curves
            if num_points > 2 and 0.3 < t < 0.7:  # Middle section
                # Create slight curve for non-border edges
                normal = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-6)
                curve_strength = np.random.uniform(-3, 3)
                point = point + normal * curve_strength
            
            points.append([float(point[0]), float(point[1])])
        
        return points
    
    def process_all_images(self):
        """Process all images in the input folder."""
        print(f" Processing images in: {self.input_folder}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(self.input_folder.glob(f"*{ext}")))
            image_files.extend(list(self.input_folder.glob(f"*{ext.upper()}")))
        
        if not image_files:
            print(" No images found!")
            return None
        
        print(f" Found {len(image_files)} images")
        
        all_pieces = []
        
        for img_path in image_files:
            print(f"  Processing: {img_path.name}")
            pieces = self.detect_edges_in_image(img_path)
            all_pieces.extend(pieces)
            print(f"    → Found {len(pieces)} pieces")
        
        # Create metadata structure
        metadata = {
            "puzzle_type": self.input_folder.name,
            "total_pieces": len(all_pieces),
            "pieces": all_pieces
        }
        
        # Save to JSON
        with open(self.output_json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n Saved {len(all_pieces)} pieces to {self.output_json_path}")
        
        # Create simple visualization
        self.create_summary_visualization(all_pieces)
        
        return metadata
    
    def create_summary_visualization(self, pieces):
        """Create a simple visualization of detected pieces."""
        if not pieces:
            print("  No pieces to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Show first 6 pieces
        for i, piece in enumerate(pieces[:6]):
            if i >= 6:
                break
                
            ax = axes[i]
            
            # Extract edges
            edges = piece['edges']
            
            # Plot all edges
            for edge_idx, edge in enumerate(edges):
                edge_array = np.array(edge)
                ax.plot(edge_array[:, 0], edge_array[:, 1], 
                       marker='o', markersize=3, linewidth=2,
                       label=f'Edge {edge_idx+1}' if edge_idx == 0 else None)
            
            ax.set_title(f"Piece: {piece['id']}")
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        # Hide unused subplots
        for i in range(len(pieces[:6]), 6):
            axes[i].axis('off')
        
        plt.suptitle(f"Detected Pieces - {len(pieces)} total", fontsize=14)
        plt.tight_layout()
        
        vis_path = self.output_json_path.parent / "edge_detection_summary.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Visualization saved: {vis_path}")

def run_simple_milestone1():
    """Run the simplified Milestone 1."""
    import argparse
    
    # YOUR EXACT PATHS
    BASE_PROJECT = Path(r"D:/asu/Fall 2025/CSE 483 Computer vision/Project")
    GRAVITY_FALLS = BASE_PROJECT / "Gravity Falls"
    OUTPUT_BASE = BASE_PROJECT / "simple_milestone1_output"
    
    parser = argparse.ArgumentParser(description="Simplified Milestone 1: Edge Detection")
    parser.add_argument("--puzzle_type", type=str, choices=["2x2", "4x4", "8x8", "corrected"], default="2x2",
                       help="Type of puzzle to process")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Custom name for output JSON file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLIFIED MILESTONE 1: Basic Edge Detection")
    print("=" * 60)
    
    # Set input folder based on puzzle type
    input_folders = {
        "2x2": GRAVITY_FALLS / "puzzle_2x2",
        "4x4": GRAVITY_FALLS / "puzzle_4x4",
        "8x8": GRAVITY_FALLS / "puzzle_8x8",
        "corrected": GRAVITY_FALLS / "corrected"
    }
    
    input_folder = input_folders.get(args.puzzle_type)
    if not input_folder or not input_folder.exists():
        print(f" Input folder not found: {input_folder}")
        print("Available folders in Gravity Falls:")
        for item in GRAVITY_FALLS.iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        return
    
    # Set output path
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    if args.output_name:
        output_path = OUTPUT_BASE / args.output_name
    else:
        output_path = OUTPUT_BASE / f"pieces_{args.puzzle_type}.json"
    
    print(f" Input: {input_folder}")
    print(f" Output: {output_path}")
    
    # Create and run detector
    detector = SimpleEdgeDetector(input_folder, output_path)
    metadata = detector.process_all_images()
    
    if metadata:
        print(f"\n Milestone 1 Complete!")
        print(f"   Pieces detected: {len(metadata['pieces'])}")
        print(f"   Output saved to: {output_path}")
        print(f"\n Next step: Run milestone2_simple.py with this file")
        print(f"   Command: python milestone2_simple.py --input \"{output_path}\"")

if __name__ == "__main__":
    run_simple_milestone1()
