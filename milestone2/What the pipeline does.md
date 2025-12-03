# Puzzle Reconstruction Project - Milestone 2

## Overview
Milestone 2 focuses on the geometric analysis of the puzzle pieces segmented in Milestone 1. The objective is to identify matching edge pairs between different pieces. This is achieved by creating a mathematical descriptor for every edge and comparing them using a rotation-invariant shape metric.

## Pipeline Description

The `milestone2_matcher.py` script performs the following steps:

### 1. Data Ingestion
* Loads `pieces_metadata.json` generated in Milestone 1.
* Extracts the 4 contour segments (edges) for each puzzle piece.

### 2. Preprocessing & Normalization
Before edges can be compared, they must be mathematically comparable.
* **Resampling:** Raw edges vary in pixel length (e.g., one edge is 120 pixels, another is 130). We use linear interpolation (`numpy.interp`) to resample every edge to exactly **50 points**.
* **Border Detection:** We calculate the "tortuosity" (path length / endpoint distance). Edges with low tortuosity are classified as straight borders and excluded from the matching pool to reduce false positives.

### 3. Edge Representation
We represent each edge as a **Normalized Coordinate Vector**.
* **Centering:** The centroid of the 50 points is subtracted, moving the shape to $(0,0)$.
* **Scaling:** The shape is scaled by its Root Mean Square (RMS) distance from the center, making the descriptor **Scale Invariant**.

### 4. Similarity Computation (The Matcher)
We use **Procrustes Analysis** for shape matching.
* **The Logic:** A puzzle "tab" on Piece A matches a "hole" on Piece B. Geometrically, this means the shape of Edge A should match the *reverse* of Edge B (since the contours wind in opposite directions relative to the connection).
* **The Algorithm:**
    1.  Take Edge A (normalized) and Edge B (normalized & reversed).
    2.  Compute the optimal **Rotation Matrix** that aligns B to A using Singular Value Decomposition (SVD).
    3.  Apply the rotation.
    4.  Compute the **Mean Squared Error (MSE)** between the points.
    5.  This MSE is the **Similarity Score**. (0.0 = Identical, > 0.5 = Poor match).

### 5. Ranking & Visualization
* The system compares every edge against every other edge ($N^2$ complexity).
* Matches with a score > 0.15 (threshold) are discarded.
* The remaining matches are ranked, and the top 5 are saved to `matches_ranked.csv`.
* The top matches are visualized using Matplotlib, showing the two edges overlaid to demonstrate the fit.

## Design Justifications

1.  **Why Resampling?**
    Euclidean distance metrics require feature vectors of identical dimensions. Pixel-based comparisons fail if lengths differ. Resampling allows point-to-point correspondence.

2.  **Why Procrustes Analysis?**
    It is a classical statistical shape analysis method. Unlike simple correlation, it explicitly solves for the optimal rotation between two shapes. This ensures that if piece A is rotated 90 degrees relative to piece B, the system still detects the match.

3.  **Why Reverse the Edge?**
    Puzzle contours are extracted (usually) counter-clockwise. A convex tab is traversed Left->Right. The matching concave hole on the other piece is also traversed Left->Right, but to mesh, they must overlap in opposite directions. Reversing the point order allows us to compare the shapes directly.

## Requirements Met
* **Similarity Computation:** Implemented via `_procrustes_distance`.
* **Thresholding/Ranking:** Filtered by `MATCH_THRESHOLD` and sorted by score.
* **Visualization:** Generated in `matches_vis/` folder showing overlaid curves.
* **No ML:** Pure geometric algebra (SVD, Interpolation) used.

## References
* *Gower, J. C. (1975). Generalized procrustes analysis.*
* *OpenCV Contours Tutorial.*
