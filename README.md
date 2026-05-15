# Feature Pyramid Networks and Multi-Scale Detection

ARI5118 – Deep Learning for Computer Vision Assignment  
University of Malta – Department of Artificial Intelligence  

By **Favour Onyedikachi Umeh**

---

##  Project Overview

This repository contains my complete Learning Pack submission for the ARI5118 Deep Learning for Computer Vision assignment on **Feature Pyramid Networks (FPNs) and Multi-Scale Detection**. The project explores the scale problem in object detection, the architecture of Feature Pyramid Networks, and their real-world applications across computer vision domains such as aerial imaging, medical analysis, surveillance, and autonomous driving.

The assignment combines theoretical understanding, technical implementation, literature synthesis, and interactive educational tools into a single structured repository.

The work was developed following the official project specification provided in the course brief. :contentReference[oaicite:0]{index=0}

---

#  Contents

This repository includes the following deliverables:

| Deliverable | Description |
|---|---|
| `study_notes.pdf` | Comprehensive technical notes covering FPN theory, mathematics, literature review, and applications |
| `quiz_with_rationale.pdf` | Adversarial quiz testing conceptual understanding with detailed rationale |
| `slides.pdf` | Presentation slides used during the in-class presentation |
| `walkthrough.ipynb` | Annotated Jupyter Notebook explaining FPN concepts and implementation |
| `ai_journal.pdf` | Transparent AI Usage Journal documenting how generative AI tools were integrated into the workflow |
| `simulator/` | Interactive Streamlit simulator for visualizing FPN feature propagation |
| `further_reading/` | Collection of research papers referenced throughout the project |

---

#  Topic Summary

Feature Pyramid Networks (FPNs) were introduced to address one of the core challenges in object detection: **detecting objects at multiple scales efficiently**.

Traditional CNN architectures face a tradeoff between:

- **Spatial precision** (strong in shallow layers)
- **Semantic richness** (strong in deep layers)

FPNs solve this by combining deep semantic features with high-resolution shallow features through:

- Bottom-up feature extraction
- Top-down feature propagation
- Lateral feature connections

The architecture allows detectors to perform well across small, medium, and large object scales simultaneously.

The project also explores several FPN extensions including:

- GraphFPN
- Latent Feature Pyramid Networks (LFPN)
- MSRA-FPN
- DM-FPN for remote sensing imagery

---

#  Repository Structure

```text
/
├── README.md
├── requirements.txt
├── study_notes.pdf
├── quiz_with_rationale.pdf
├── slides.pdf
├── walkthrough.ipynb
├── ai_journal.pdf
├── simulator/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
└── further_reading/
    └── *.pdf
