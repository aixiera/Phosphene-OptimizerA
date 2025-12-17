# Project Overview: Phosphene Optimizer

## Background

Visual prostheses and brain–computer interfaces aim to restore vision by stimulating neural tissue using a limited number of electrodes. These systems produce phosphene-based percepts that are extremely low-resolution and structurally constrained.

A key challenge is determining how to encode visual information so that useful structure is preserved under these constraints.

---

## Project Goal

The goal of this project is to explore whether artificial intelligence can learn optimal phosphene encoding strategies that maximize functional visual information under a fixed stimulation budget.

Rather than focusing on pixel-perfect reconstruction, the project emphasizes structure, task relevance, and perceptual constraints.

---

## Core Idea

The project separates the system into two components:

- **Learned Encoder:** A neural network that maps images to phosphene intensities.
- **Fixed Renderer:** A differentiable perceptual model that simulates phosphene vision.

This separation allows the encoder to adapt while enforcing biologically inspired constraints on perception.

---

## Current Progress

- A differentiable phosphene renderer has been implemented.
- A CNN-based encoder has been trained under extreme resolution constraints.
- Training stability and representation collapse have been analyzed.
- A functional evaluation pipeline has been established.
- Week 1 findings have been documented in `WEEK1_REPORT.md`.

---

## Why This Matters

Understanding how to allocate limited stimulation resources is central to the future of visual prosthetics. This project provides a controlled environment to study encoding strategies, failure modes, and evaluation methodologies for phosphene-based vision.

---

## Roadmap

- **Week 1:** Feasibility, encoding collapse analysis, functional evaluation setup
- **Week 2:** Task-driven loss functions and learning-based evaluation
- **Future:** Sparsity, foveation, hardware-aware optimization, and clinical relevance
