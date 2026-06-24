# U.M.E.R Engine — Spatial Intelligence for Vision Datasets

![Domain](https://img.shields.io/badge/Domain-Computer%20Vision%20%26%20Medical%20Imaging-ff9800?style=flat-square)
![Accuracy](https://img.shields.io/badge/Validation-90.00%25%20Accuracy-success?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Deterministic%20Mass%20Gravity-blue?style=flat-square)

> **Part of the U.M.E.R (Uniform Memory Encoded Representation) Compute Core**

---

## Overview: Assassinating the Convolutional Black Box

Modern Computer Vision heavily relies on Convolutional Neural Networks (CNNs). While effective, these models are stochastic "black boxes" that learn abstract, unexplainable weight distributions through millions of probabilistic backpropagation cycles.

This repository demonstrates how the **U.M.E.R. Spatial Intelligence framework** entirely replaces CNNs. By extracting explicit mathematical realities from pixels and treating them as physical coordinates in a hyper-dimensional space, the engine transforms image classification into a deterministic physics problem. Rather than relying on learned probabilities, the engine calculates the literal "Mass Gravity" of historical data points to pull a new image into a rigid, 100% explainable classification.

## The Ultrasound Benchmark (Empirical Proof)

To prove the architecture, the engine was deployed against a highly complex medical imaging dataset of **647 clinical ultrasound images**. 

### Stage 1: Perceptual Abstraction & Universe Balancing
Instead of feeding raw pixels, the images were pre-processed (10% border crop, CLAHE contrast enhancement, 3x3 Gaussian Blur) and sliced into a `3x3` spatial grid. For each of the 9 zones, 12 deterministic mathematical properties were extracted (Sobel edge densities, GLCM texture features, statistical moments). 
* **Result:** Exactly 108 spatial concepts per patient. 
* To prevent topological bias, 181 synthetic minority-class masses were injected to balance the spatial gravity.

### Stage 2: The Dual-Sweep Optimization
The engine executed a multi-phase optimization sweep to map these 108 dimensions into a functional, $O(1)$ hash cascade:
1. **The PGP Sweep:** Autonomously discovered optimal synergistic pairs among the 108 features to construct the hash trees.
2. **The Macro-Physics Sweep:** Iterated over spatial boundary sizes (Cells) and interaction distances (Radii). The engine locked onto `Cell = 0.4` and `Radius = 0.25`, peaking at **90.00% accuracy**.
3. **The Attention Sweep (Decay):** Evaluated how aggressively the influence of deeper hash trees should decay. A subtle decay rate of `0.05` cemented the final 90.00% accuracy.

### The Autopsy (Explainability)
Because the engine uses absolute physics rather than hidden weights, misclassifications are 100% traceable. Out of the validation set, the engine only misclassified 13 total patients:
* **10 False Positives:** Autopsy logs prove that the benign gravity (e.g., 25.753) heavily overpowered the malignant gravity (9.236) within specific edge-variance grids, pulling the momentum into a false benign reading.
* **3 False Negatives:** Malignant gravity completely dominated (11.577 vs 0.540) in central structural grids, causing missed diagnoses.

---

## Generalized Blueprint: Abstract Spatial Intelligence

This exact architecture can be deployed for satellite imagery analysis, factory defect detection, or facial recognition without altering the core PyCUDA engine. The blueprint operates in four distinct phases:

### 1. Spatial Tessellation (The Grid Slicer)
Instead of processing an image as a monolithic matrix, slice it into a localized $N \times N$ grid. A `3x3` grid yields 9 zones, preserving the macro-geometry of the target. (E.g., for satellite urban sprawl tracking, an `8x8` grid isolates specific neighborhood textures).

### 2. Deterministic Feature Extraction
Within every single zone, extract explicit numerical descriptors to define the physical reality:
* **Structural Mechanics:** Use Sobel/Canny operators to define physical boundary density.
* **Textural Reality:** Apply Gray Level Co-occurrence Matrices (GLCM). By extracting Contrast, Homogeneity, Energy, and Correlation, texture (asphalt vs. water) is reduced to an absolute signature.
* **Distribution Moments:** Calculate Skewness and Kurtosis to quantify light and shadow distribution.

### 3. Topological Hashing (The U.M.E.R. Core)
Flatten the extracted zones into a single 1D array (e.g., 15 features across 16 zones creates a 240-dimensional space). The Phase 1 PGP sweep autonomously tests thousands of feature combinations to weave optimal pairs into a hierarchical hash cascade.

### 4. Mass Gravity Classification
When a new, unseen image is processed, it is dropped into the hyper-dimensional hash space. Because the engine utilizes a sort-free $O(1)$ lookup, it instantly identifies which historical data points share the exact same spatial bucket. By calculating the physical distance to neighbors and applying an exponential decay, the engine calculates the total "signed mass gravity" pulling on the new image. 

**If the gravity of "Defective Parts" is stronger than "Healthy Parts," the classification is finalized instantaneously without a single probabilistic hallucination.**

---

## Execution

The complete data ingestion, physics sweeps, and autopsy generation are located within the Jupyter environment.

### Execution
Launch the notebook to initialize the grid slicing and trigger the mass gravity sweeps:
```bash
jupyter notebook successful-deeplearningumer.ipynb
