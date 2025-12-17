# WEEK 1 REPORT  
Phosphene Optimizer: Learned Encoding under Extreme Resolution Constraints

## 1. Motivation and Research Question

Visual prostheses and brain–computer interfaces operate under severe resolution constraints due to the limited number of stimulation channels. A central challenge is how to encode visual information so that useful structure is preserved, even when faithful image reconstruction is impossible.

Rather than optimizing pixel-level fidelity, this project investigates whether a learned encoding strategy can produce phosphene representations that preserve meaningful visual structure.

**Week 1 research question:**  
Can a neural network learn a useful phosphene encoding under extreme resolution constraints?

---

## 2. System Overview

The system consists of two clearly separated components:

1. **Phosphene Encoder** – a learnable CNN that maps input images to a fixed-length vector of phosphene intensities.
2. **Phosphene Renderer** – a fixed, differentiable forward model that converts phosphene intensities into a perceptual image via spatial upsampling and Gaussian blur.

Only the encoder is trained; the renderer represents a constrained perceptual model.

---

## 3. Methodology (Day 1–5)

A CNN-based encoder was trained using a pixel-wise mean squared error loss between the rendered phosphene image and the grayscale input image. The encoder outputs 256 phosphene intensities arranged on a fixed 16×16 grid.

Training results showed stable loss convergence, demonstrating that gradient-based optimization is feasible despite the severe information bottleneck.

---

## 4. Analysis: Representation Collapse

Further analysis revealed that the learned encoder produces highly similar phosphene activation patterns across different input images.

This behavior was confirmed through:
- Low per-phosphene variance across samples
- Nearly identical 16×16 activation heatmaps for different inputs

The encoder therefore exhibits **representation collapse**, converging to a stable, image-agnostic encoding. This outcome is consistent with optimization under extreme bottlenecks and pixel-wise reconstruction objectives.

---

## 5. Functional Evaluation (Day 6)

Human detection was selected as a proxy task for functional visual performance. A fixed OpenCV HOG + SVM detector was applied to original images, naive phosphene representations, and learned phosphene representations.

The detector failed across all conditions, including original images. This negative result is expected given the detector’s reliance on fine-grained gradient features, which are severely degraded in phosphene-based representations.

Although no detection advantage was observed, this result highlights the limitations of traditional hand-crafted detectors for evaluating prosthetic vision.

---

## 6. Key Findings

- A differentiable phosphene encoding system can be trained under extreme resolution constraints.
- The learned encoder converges to a stable, structure-biased encoding pattern.
- Pixel-wise reconstruction objectives promote representation collapse.
- Traditional gradient-based detectors are unsuitable for phosphene-based evaluation.

---

## 7. Limitations

- The training objective is limited to pixel-wise reconstruction loss.
- Functional evaluation used a single, non-learning-based detector.
- The dataset size and diversity are limited.

---

## 8. Next Steps (Week 2)

Future work will focus on task-driven optimization and evaluation, including:
- Introducing task-aware loss functions to mitigate representation collapse
- Evaluating encodings using learning-based detectors
- Exploring sparsity and foveation strategies for efficient phosphene allocation
