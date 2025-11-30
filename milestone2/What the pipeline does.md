# What the pipeline does (technical summary)

Milestone 2 takes the segmented pieces + metadata from Milestone 1 and transforms them into **matchable puzzle-edge descriptors**. The pipeline does five major things:

1. **Load all metadata**
   Reads contours, masks, cropped images, and candidate edges from `pieces_metadata.json`.

2. **Normalize edges for comparison**
   Each edge segment is resampled to a fixed number of points and normalized to be scale-invariant.

3. **Compute rotation-invariant descriptors**
   For every edge, the pipeline computes:
   • **Curvature signature** (curvature vector across the edge)
   • **Fourier Descriptor** (frequency-domain shape descriptor)
   • **Length & shape ratio** (global geometry features)

4. **Edge-to-edge similarity scoring**
   Every edge from every piece is compared with every other edge:
   • L2 distance between curvature signatures
   • L2 distance between normalized Fourier descriptors
   • Length difference penalty
   The final score is a weighted combination of these.

5. **Generate ranked matches + visualizations**
   For each edge:
   • Save a CSV list of the top-k matching edges
   • Overlay visualizations to visually validate good matches
   • Output a summary file describing all matches

---

# Where the outputs are (quick demo)

Inside the Milestone-2 output folder:

* `edge_descriptors.csv` — a table listing every edge and the descriptor vectors.
* `matches/edge_X_top_matches.csv` — ranked match list for each edge.
* `vis/match_X_Y.png` — visual overlay showing how two edges align after normalization.
* `pairwise_scores.npy` — a full matrix of similarity scores between all edges.
* `milestone2_summary.json` — high-level info about number of pieces, edges, and strongest matches.

---

# Tuning tips & improvements (for stronger puzzle-solving)

* Adjust the number of resampled points (e.g., 128 → 256) for more precise descriptor matching.
* Change the Fourier descriptor length for smoother vs. sharper edge representation.
* Add orientation estimation so the system predicts how the pieces rotate relative to each other.
* Add tab/blank classification (convex vs. concave curvature sign) to block impossible matches.

---

# Explanation

Milestone 2 fully implements the project’s *matching* stage:
It transforms raw contours into standardized descriptors, computes edge signatures, and identifies which edges from different pieces likely connect. This includes rotation-invariant descriptors, curvature-based matching, global shape features, ranked output, and a complete match visualization system.

This satisfies the essential requirement:
**“Given segmented puzzle pieces, compute descriptors and identify candidate matching edges between pairs of pieces.”**

---

# One-to-one mapping (Milestone 2 → code)

Below every required milestone item, I show the exact functions/variables/files in the code that fulfill it.

---

### 1) Load metadata from Milestone 1

**What the code does:**
Reads all piece contours, edges, masks, and crops from the metadata JSON.

**Where:**

```python
with open(metadata_path, "r") as f:
    data = json.load(f)
```

**Why it's required:**
Matching cannot begin without the contour + edge segments extracted in Milestone 1.

---

### 2) Normalize edge segments (resampling + scaling)

**What the code does:**
Resamples each edge to a fixed count of points (e.g., 128) and normalizes them to zero-mean, unit-length.

**Where:**

```python
norm = normalize_edge(edge_np)
```

**Why:**
Without resampling, edges have wildly different lengths & point densities.
Normalization makes descriptors comparable.

---

### 3) Compute rotation-invariant descriptors

**What the code does:**
For each edge, computes:

• **Curvature signature**

```python
k = curvature_signature(norm)
```

• **Fourier Descriptor**

```python
f = fourier_descriptor(norm, FD_SIZE)
```

• **Global length ratios**

```python
L = compute_length(norm)
```

All stored in:

```python
descriptors[edge_id] = {...}
```

**Why:**
These are the fundamental building blocks of puzzle edge matching.

---

### 4) Edge-to-edge similarity scoring

**What the code does:**
Computes a weighted score of descriptor differences:

```python
score = (
    W_CURV * np.linalg.norm(k1 - k2) +
    W_FD   * np.linalg.norm(fd1 - fd2) +
    W_LEN  * abs(len1 - len2)
)
```

**Why:**
Two edges are a likely match if curvature, Fourier coefficients, and length all align.

---

### 5) Produce ranked matches + save results

**What the code outputs:**
For each edge, top-K most similar candidates are written to a CSV:

```python
save_match_csv(edge_id, ranked_list)
```

Visual overlays are created by plotting normalized edges:

```python
save_match_visualization(edge_id, match_id, norm1, norm2)
```

Full system summary is saved to:

```python
milestone2_summary.json
```

**Why:**
This satisfies the deliverable:
**“Produce top match candidates and visual proofs for each edge.”**

---

# Important implementation details & heuristics

* **Resampling** ensures edges are the same temporal length — required for Fourier comparison.
* **Normalization** makes shape comparisons rotation/scale independent.
* **Curvature signature smoothing** reduces noise-induced spikes.
* **Fourier descriptor truncation** discards high-frequency noise and keeps major shape trends.
* **Match filtering** removes comparisons of edges from the same piece.

These heuristics stabilize the matching stage significantly.

---

# Where to tune for better results

* Number of resampled points (64/128/256)
* Fourier descriptor size (10 → 20)
* Curvature smoothing window
* Increasing match scoring weights for curvature or Fourier descriptors
* Adding convex/concave filtering to block impossible fits

---

# What Milestone 2 deliverables are now satisfied

* ✔️ Use contours produced in Milestone 1
* ✔️ Normalize edges for descriptor extraction
* ✔️ Compute rotation-invariant shape descriptors
* ✔️ Compare all edges pairwise across all pieces
* ✔️ Produce similarity scores
* ✔️ Rank best matches
* ✔️ Save the results in CSV, visual PNGs, and JSON summary
* ✔️ Provide a systematic method for puzzle-piece matching

---