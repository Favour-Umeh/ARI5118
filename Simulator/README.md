# Feature Pyramid Network (FPN) Interactive Simulator

An educational Streamlit application for visualising how Feature Pyramid Networks (FPNs) perform multi-scale feature extraction and object detection.

Built as part of the ARI5118 – Deep Learning for Computer Vision assignment at the University of Malta. The simulator accompanies the study notes, literature review, walkthrough notebook, and presentation materials for the topic:

> Feature Pyramid Networks and Multi-Scale Detection

The simulator implementation is based on the architecture and concepts discussed throughout the assignment report. :contentReference[oaicite:0]{index=0}

---

# Purpose

Feature Pyramid Networks solve one of the most difficult problems in computer vision:

- detecting small, medium, and large objects simultaneously.

This simulator allows users to interactively explore:

- Bottom-up feature extraction
- Top-down feature propagation
- Lateral feature connections
- Multi-scale feature fusion
- Pyramid outputs (P2–P5)
- Detection performance tradeoffs

The goal is educational clarity rather than production-level object detection.

---

# Features

## Interactive FPN Simulation

- Simulates FPN feature propagation
- Visualises pyramid feature maps (P2–P5)
- Demonstrates semantic vs spatial tradeoffs

---

## Adjustable Parameters

Users can experiment with:

- Lateral connection strength
- Upsampling method
  - Bilinear interpolation
  - Nearest-neighbour interpolation
- Detection confidence threshold

---

## Side-by-Side Comparison Mode

Compare two FPN configurations simultaneously to observe:

- Strong vs weak lateral fusion
- Different upsampling strategies
- Changes in feature quality across scales

---

## Synthetic Multi-Scale Dataset Generation

The simulator generates synthetic scenes containing:

- Small objects
- Medium objects
- Large objects

This mimics real-world multi-scale detection challenges.

---

## CPU-Only Execution

The application runs entirely on CPU and requires no GPU.

This satisfies the assignment specification requirements. :contentReference[oaicite:1]{index=1}

---

# Project Structure

```text
simulator/
├── app.py
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd simulator
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Simulator

```bash
streamlit run app.py
```

The application will launch in your browser automatically.

---

# Core Concepts Demonstrated

The simulator illustrates the three major components of Feature Pyramid Networks:

## Bottom-Up Pathway

Simulates a CNN backbone extracting features at different scales.

- High resolution → weak semantics
- Low resolution → strong semantics

---

## Top-Down Pathway

Upsamples deeper semantic feature maps to recover spatial information.

---

## Lateral Connections

Combines:

- high-resolution spatial detail
- high-level semantic context

through feature fusion.

---

# Visualisations

The simulator provides:

- Heatmaps of feature activations
- Pyramid-level comparisons
- Simulated detection score charts
- Multi-scale feature visualisation

---

# Technologies Used

- Python
- Streamlit
- NumPy
- Matplotlib
- Pillow

---

# Educational Context

This simulator was developed as part of a broader Learning Pack on:

> Feature Pyramid Networks and Multi-Scale Detection

The complete project also includes:

- Study Notes
- Literature Review
- Annotated Jupyter Notebook
- Adversarial Quiz
- AI Usage Journal
- Presentation Slides

---

# Key Reference

Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S.  
Feature Pyramid Networks for Object Detection  
CVPR 2017  
https://doi.org/10.1109/CVPR.2017.106

---

# Author

Favour Onyedikachi Umeh  
MSc Artificial Intelligence  
University of Malta

---

# Disclaimer

This simulator is designed for educational and conceptual demonstration purposes.

It is not intended to replicate the full computational complexity or performance of industrial object detection frameworks such as Faster R-CNN, RetinaNet, YOLO, or Detectron2.
