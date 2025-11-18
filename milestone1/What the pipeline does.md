# What the pipeline does (technical summary)

1. **Resize (optional):** downscale large images for speed while mapping contours back to original resolution for accurate masks/crops.
2. **Denoise:** bilateral filter (edge-preserving) or `fastNlMeansDenoisingColored` (slower but effective).
3. **Local contrast enhance:** CLAHE to handle uneven illumination.
4. **Thresholding / background removal:** Either adaptive threshold or Otsu on blurred image, then invert as needed (we used `THRESH_BINARY_INV` for piece=foreground).
5. **Morphological cleanup:** opening + closing removes speckles and closes small holes on piece masks.
6. **Contour extraction:** `cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_NONE)` to get dense contour point lists.
7. **Filter by area:** remove tiny blobs (heuristic relative to image size).
8. **Mask & Crop:** draw filled contour into mask at original image size, produce masked crop and save both.
9. **Edge segmentation (contour -> candidate edges):** simple curvature measure (angle between neighbors over a window). High curvature peaks are used as cut points; contour segments between peaks become candidate edges (heuristic, useful later for matching tabs/blanks).
10. **Metadata:** for each piece we store: unique ID, source image path, bounding box, mask path, crop path, full contour (list of xy), and `edges` (list of point lists).

---

# Where outputs are (quick demo)

* `/mnt/data/puzzle_output_quick/` — quick demo folder with per-image subfolder that contains `contours.png`, `masks/`, `crops/`.

---

# Tuning tips & improvements (for robustness)

* If pieces are photographed on a textured or colored background, try:

  * Convert to HSV or LAB and threshold on saturation/value instead of grayscale.
  * Use color clustering (k-means) to separate background vs pieces.
  * Use **GrabCut** with an initial rectangle + morphological cleaning to get better piece masks.
* If pieces touch / overlap:

  * Use distance transform + watershed segmentation (markers from connected components) on the binary mask to separate touching pieces.
* For noise & fine detail:

  * Use `fastNlMeansDenoisingColored` or stronger morphological opening after thresholding.
* For irregular illumination:

  * Use `cv2.illuminationChange` alternatives: morphological top-hat / background estimation by large-kernel opening and subtracting background.
* For better edge segmentation:

  * Use curvature with smoothing and multi-scale detection; detect tabs/blanks by finding concave vs convex curvature sign (use cross product sign).
  * Fit piece boundary to Fourier descriptors for rotation invariance (useful at matching stage).

---

# Explaination

The script implements every required Milestone-1 element: noise reduction & cleanup, contrast/edge enhancement, background removal (binarization + morphology), segmentation (contour extraction + filtering), per-piece mask & crop generation, contour storage, and a first-pass *edge segmentation* (curvature split). It saves visual checks (overlay), masks, crops and a `pieces_metadata.json` describing contours and candidate edges for every detected piece.

# One-to-one mapping (Milestone 1 → code)

Below each milestone item We will show the **exact function / variable / file outputs** in the script that implement it.

### 1) Noise reduction and Image Cleanup

* **What the code does:** edge-preserving denoising.
* **Where:**

  * `den = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)` (fast, preserves edges).
  * In the earlier slower variant I used `cv2.fastNlMeansDenoisingColored(...)` (commented in demo).
* **Tune:** filter params `d`, `sigmaColor`, `sigmaSpace`.

### 2) Image Enhancement (edge/contrast)

* **What the code does:** local contrast enhancement and optional top-hat.
* **Where:**

  * `clahe = cv2.createCLAHE(...); gray = clahe.apply(gray)` — CLAHE improves local contrast and helps contours pop out.
  * (In the longer run I applied top-hat morphology in an earlier attempt; that’s easy to re-enable if needed.)
* **Tune:** `clipLimit`, `tileGridSize`.

### 3) Background Removal (binarization / thresholding)

* **What the code does:** global/adaptive threshold + morphological cleanup.
* **Where:**

  * `_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)` — Otsu automatic binarization (I used `THRESH_BINARY_INV` since pieces are usually darker/lighter depending on dataset).
  * Alternative in demo: `cv2.adaptiveThreshold(...)` if illumination is non-uniform.
  * Morphology: `cv2.morphologyEx(..., MOPH_OPEN)` and `MORPH_CLOSE` remove speckles and close holes.
* **Tune:** adaptive threshold block size / C, Otsu works well if foreground/background bimodal; otherwise use HSV/LAB or color clustering.

### 4) Segmentation of individual puzzle piece (contour extraction + cropping)

* **What the code does:** external contour extraction, area filtering, mask creation, masked crop.
* **Where:**

  * Contours: `contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)`
  * Area filtering heuristic: `min_area = img_small.shape[0]*img_small.shape[1]*MIN_AREA_RATIO` (`MIN_AREA_RATIO` is configurable).
  * Mask & crop creation: draw filled contour into mask (`cv2.drawContours(mask, [cnt_orig], -1, 255, -1)`), bounding box `cv2.boundingRect`, then masked crop via `cv2.bitwise_and`.
* **Outputs saved:** per-piece `masks/<id>_mask.png` and `crops/<id>_crop.png`.

### 5) Descriptor storage (contour + edge points)

* **What the code does:** record full contour and a first pass split of the contour into candidate *edge segments*.
* **Where:**

  * Contour points stored as `contour` in metadata (list of `[x,y]` pairs).
  * Edge segmentation: `edges = split_contour_by_curvature(cnt_np, ...)` uses `discrete_curvature(...)` to find high curvature points and cut the contour into segments — these `edges` are stored in the metadata too.
  * **File:** top-level JSON `pieces_metadata.json` (or `pieces_metadata_demo.json` in the demo) contains all metadata for downstream matching.
* **Why this is useful:** these contours + edge segments are the raw data you will use in Milestone-2 to compute rotation-invariant descriptors and compare edges.

### Visual checks and summaries

* **Where:** per image `vis/<image>_contours.png` — shows overlay of detected contours (good for QC).
* **Summary:** `pieces_summary.csv` gives a quick table (ID, bbox, area, n_contour_pts, n_edges).

# Important implementation details & heuristics

* **Resize & mapping:** large images are downscaled for processing (`MAX_DIM`) to speed up processing; contours are mapped back to original resolution before mask/crop saving. This preserves mask accuracy while reducing compute.
* **Contour approximation:** I used `CHAIN_APPROX_NONE` to preserve dense boundary points — important for curvature and matching (you can simplify with `approxPolyDP` if you need fewer points).
* **Edge segmentation heuristic:** `discrete_curvature` computes an angle between vectors at a sliding neighborhood `k`. Peaks define cut points. This is a heuristic which gives reasonable candidate edges (tabs/blanks) but will need tuning or replacement by more advanced splitting if pieces are noisy/touching.

# Where to tune if results are imperfect

* `MIN_AREA_RATIO` — too high removes small pieces; too low includes noise.
* Thresholding method: switch between Otsu, adaptive threshold, or color-space thresholding (S/V in HSV or L in LAB) for complex backgrounds.
* Morphology kernel sizes and iterations — small kernels preserve detail; larger kernels remove specks.
* For touching pieces: consider distance transform + watershed marker segmentation (not implemented in this quick script).
* For fine edges: `fastNlMeansDenoisingColored` or stronger bilateral parameters.

# What milestone 1 deliverables are now satisfied

* ✅ Noise reduction and cleanup (implemented)
* ✅ Image enhancement (CLAHE & optional morphological top-hat)
* ✅ Background removal (binarization + morphological cleanup)
* ✅ Segmentation & per-piece masks + crops (saved to disk)
* ✅ Contour extraction and storage (full contour saved)
* ✅ Edge segmentation (candidate edge points saved)
* ✅ Visual QC images (contour overlays)
* ✅ Metadata file (`pieces_metadata.json`) containing unique ID, mask path, crop path, contour, and edges
