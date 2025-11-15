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
