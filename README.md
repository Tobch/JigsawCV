# Jigsaw Puzzle Reconstruction System (Computer Vision)

**Course:** CSE 483 Computer Vision (Fall 2025)  
**Date:** December 2025

## 👥 Team Members

  * **Ahmed Mohamed Elsayed Tabbash** (20P1076)
  * **Belal Waleed Mohammed Abdul-Mumen Eldeghidy** (2101775)
  * **Ali Mohamed Ali Hassan** (22P0306)
  * **Zeyad Essam Elsayed Khalaf Hendawy** (20P5728)


# 📄 **PROJECT DESCRIPTION**

**PuzzleVision** is a classical computer-vision pipeline designed to automatically analyze, segment, and match jigsaw puzzle pieces using purely classical image processing techniques—without any machine learning or deep learning.

Developed as part of **Ain Shams University – CSE483 / CESS5004 (Computer Vision)** course project, the system emulates the visual reasoning humans use when solving physical puzzles. It extracts puzzle piece contours, represents edge shapes in a rotation-invariant form, and compares edges to suggest likely matches.

---

## 🔍 **Key Features**

* **Robust Preprocessing:**
  Noise reduction, edge enhancement, adaptive thresholding, and background removal.

* **Puzzle Piece Segmentation:**
  Extraction of clean binary masks for every puzzle piece using contour-based segmentation.

* **Contour & Edge Representation:**
  Computation of rotation-invariant descriptors and organized storage of edges with unique IDs.

* **Edge Similarity Matching:**
  Comparison of every pair of edges using classical distance metrics to identify the most likely complementary pairs.

* **Visualization:**
  Plotting matched edges, ranking similarity scores, and visual demo of candidate connections.

---

## 🧩 **Project Workflow**

### **Milestone 1 — Preprocessing & Segmentation**

1. Noise reduction & cleanup
2. Edge enhancement
3. Thresholding & background removal
4. Piece contour extraction & cropping
5. Descriptor storage (contours + edge points)

### **Milestone 2 — Edge Matching & Demo**

1. Rotation-invariant similarity computation
2. Thresholding + ranking matches
3. Visualization of matching candidates
4. Full system demo (clean case + challenging case)
5. Documentation, analysis & reflections

---

## 🎯 **Goal**

To design a complete classical computer vision system capable of understanding puzzle piece geometry and proposing valid puzzle edge matches through contour comparison — demonstrating strong understanding of preprocessing, segmentation, shape descriptors, and classical CV algorithms.

---

Milestone 1 builds the *foundation* (clean masks, contours, and descriptors).
Milestone 2 builds the *matching engine* on top of that.

### How it works in practice

1. **Run Milestone 1**

   * It preprocesses images
   * Segments puzzle pieces
   * Extracts contours
   * Generates descriptors
   * Stores everything in `pieces_metadata.json`

2. **Then run Milestone 2**

   * It loads the metadata from Milestone 1
   * Computes similarities between edges
   * Ranks matches
   * Generates visualizations, CSVs, and the final demo outputs

-----

## 🚀 Pipeline Architecture

The project is divided into two distinct milestones, implemented in `milestone1_pipeline.py` and `milestone2_pipeline.py`.

### **Milestone 1: Preprocessing & Segmentation**

**Goal:** Transform raw, potentially noisy images into structured metadata (contours, masks, edge segments).

#### [cite_start]1. Adaptive Contrast Enhancement (CLAHE) [cite: 38, 40, 41]

  * **Technique:** Contrast Limited Adaptive Histogram Equalization.
  * [cite_start]**Implementation:** `enhance_contrast_color` uses `cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))` on the Luminance channel (LAB color space). [cite: 42]
  * **Justification:** Unlike global equalization, CLAHE operates locally, allowing the system to detect puzzle edges even in shadowed areas or uneven lighting. [cite_start]It prevents noise amplification in flat regions. [cite: 43, 44]

#### [cite_start]2. Noise Attenuation [cite: 45]

  * [cite_start]**Technique:** Median Filtering (`cv2.medianBlur`, kernel size 3x3). [cite: 46, 47]
  * **Justification:** Effectively removes "salt-and-pepper" noise while preserving edge sharpness. [cite_start]Gaussian blur was avoided to prevent rounding the sharp corners of the puzzle pieces, which are critical for corner detection later. [cite: 48, 49]

#### [cite_start]3. Gamma Correction [cite: 50]

  * [cite_start]**Technique:** Power-law transformation (`gamma=1.8`). [cite: 51, 52]
  * [cite_start]**Justification:** Normalizes brightness distribution by lifting dark regions, aiding in separating the piece foreground from the background before thresholding. [cite: 53, 54]

#### [cite_start]4. Background Removal & Segmentation [cite: 55]

  * [cite_start]**Technique:** Otsu’s Binarization & Morphological Operations. [cite: 56, 59]
  * [cite_start]**Implementation:** `cv2.threshold` with `THRESH_OTSU` followed by `cv2.morphologyEx` (Opening to remove noise, Closing to fill holes). [cite: 57, 61, 62]
  * **Justification:** Otsu's method automatically calculates the optimal threshold to handle different background colors. [cite_start]Morphological operations ensure the resulting binary mask is solid and free of artifacts. [cite: 58]

#### [cite_start]5. Contour Extraction & Splitting [cite: 63]

  * [cite_start]**Technique:** Discrete Curvature Analysis. [cite: 68]
  * **Implementation:**
    1.  [cite_start]**Extraction:** `cv2.findContours` retrieves external contours. [cite: 65]
    2.  [cite_start]**Smoothing:** A windowed average filter (window=5) reduces pixel aliasing/jaggedness. [cite: 65, 66]
    3.  **Corner Detection:** `discrete_curvature` calculates the cosine angle between vectors at each point. [cite_start]Peaks in this curvature signal indicate corners. [cite: 70, 71]
    4.  [cite_start]**Splitting:** The contour is split at these peaks into 4 discrete edge segments (Top, Bottom, Left, Right). [cite: 72]

-----

### **Milestone 2: Edge Matching Strategy**

**Goal:** Mathematically compare the extracted edges to find complementary pairs using `PuzzleEdgeMatcher`.

#### [cite_start]1. Edge Normalization [cite: 102]

  * **Resampling:** Each edge is resampled to exactly **50 points** using linear interpolation (`np.interp`). [cite_start]This allows pairwise comparison between edges of different physical lengths (e.g., due to camera perspective). [cite: 102]
  * **Centering:** The centroid is subtracted from all points to center the edge at `(0,0)`.
  * **Scaling:** The edge is scaled so its RMS distance from the origin is 1.0.
  * **Justification:** These steps make the matching **Scale Invariant** and **Translation Invariant**.

#### 2\. Feature Categorization (Border Detection)

  * **Technique:** Tortuosity Analysis.
  * **Logic:** Calculates the ratio between the path length and the straight-line distance between endpoints. If the ratio is $< 1.05$, the edge is classified as a "Border" (straight edge) and excluded from the matching pool.
  * **Justification:** Reduces computational load and false positives by filtering out flat edges that don't carry unique shape information.

#### [cite_start]3. Shape Matching: Procrustes Analysis [cite: 106, 107]

  * **Technique:** Orthogonal Procrustes Analysis via SVD (Singular Value Decomposition).
  * **Implementation:** `compute_procrustes_distance`.
    1.  **Reversal:** The target edge is reversed (`target[::-1]`) because locking pieces are mirror images (convex matches concave).
    2.  **Alignment:** SVD calculates the optimal rotation matrix $R$ to align the two shapes.
    3.  **Scoring:** The score is the Mean Squared Error (MSE) between the points after optimal alignment.
  * **Justification:** This provides **Rotation Invariance**. [cite_start]The system doesn't need to know the orientation of the pieces; it mathematically "rotates" them to see if they fit. [cite: 109]

#### 4\. Ranking & Thresholding

  * **Logic:** The system computes the distance to all other edges and keeps the Top-K (5) matches. Matches with a distance score $> 0.15$ are discarded.
  * **Justification:** Handles the "one-to-many" ambiguity inherent in puzzles (pieces often look similar) by providing a ranked list of candidates rather than a single guess.

-----

## 🛠️ Usage Instructions

### Prerequisites

  * Python 3.8+
  * OpenCV (`opencv-python`)
  * NumPy, Pandas, Matplotlib

### Running Milestone 1 (Extraction)

```bash
python milestone1_pipeline.py
```

  * **Input:** Reads images from the configured `DATA_DIR`.
  * [cite_start]**Output:** Generates `pieces_metadata.json` containing contours and edge splits. [cite: 74, 78]
  * [cite_start]**Debug:** Saves visualization images in `puzzle_output/vis/` showing the segmented pieces and detected corners. [cite: 77]

### Running Milestone 2 (Matching)

```bash
python milestone2_pipeline.py --input "puzzle_output/pieces_metadata.json"
```

  * **Input:** Consumes the JSON file from Milestone 1.
  * **Output:** Generates `matches_ranked.csv` (list of matches) and `visualizations/` (plots showing the alignment).
  * **Arguments:**
      * `--threshold`: (Default 0.15) stricter matching \< 0.15, looser \> 0.15.
      * `--top_k`: Number of matches to save per edge.

-----

## 📚 References

1.  **Procrustes Analysis:** *Gower, J. C. (1975). Generalized procrustes analysis. Psychometrika, 40(1), 33-51.*
2.  **CLAHE:** *Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. Graphics gems IV, 474-485.*
3.  **Otsu's Method:** *Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and cybernetics, 9(1), 62-66.*

-----
