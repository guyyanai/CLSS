# CLSS: Contrastive learning unites sequence and structure in a global representation of protein space

**Paper:** [https://www.biorxiv.org/content/10.1101/2025.09.05.674454v3.full.pdf](https://www.biorxiv.org/content/10.1101/2025.09.05.674454v3.full.pdf)

**DOI:** [https://doi.org/10.1101/2025.09.05.674454](https://doi.org/10.1101/2025.09.05.674454)

**Interactive viewer:** [https://gabiaxel.github.io/clss-viewer/](https://gabiaxel.github.io/clss-viewer/)

---

## Abstract

> Amino acid sequence dictates the three-dimensional structure and biological function of proteins. Yet, despite decades of research, our understanding of the interplay between sequence and structure is incomplete. To meet this challenge, we introduce Contrastive Learning Sequence-Structure (CLSS), an AI-based contrastive learning model trained to co-embed sequence and structure information in a self-supervised manner. We trained CLSS on large and diverse sets of protein building blocks called domains. CLSS represents both sequences and structures as vectors in the same high-dimensional space, where distance relates to sequence-structure similarity. Thus, CLSS provides a natural way to represent the protein universe, reflecting evolutionary relationships, as well as structural changes. We find that CLSS refines expert knowledge about the global organization of protein space, and highlights transitional forms that resist hierarchical classification. CLSS reveals linkage between domains of seemingly separate lineages, thereby significantly improving our understanding of evolutionary design.

---

## TL;DR

**CLSS** is a self-supervised, two-tower contrastive model that co-embeds **protein sequences** and **structures** into a **shared 32‑D space**, enabling unified mapping of protein space across modalities.

---

## Key ideas

* **Two-tower architecture:** sequence tower (ESM2‑like, \~35M params) co-trained; structure tower (ESM3) kept frozen; both feed **32‑D L2‑normalized adapters**.
* **Segment-aware training:** contrastive pairs match **full-domain structures** with **random sequence sub-segments (≥10 aa)** to encode contextual compatibility.
* **Unified embeddings:** sequences, structures, and subsequences align in a **single space**; distances track ECOD hierarchy and reveal cross-fold relationships.
* **Scale & efficiency:** \~36M trainable params, compact embeddings (32‑D) supporting efficient inference and training.
* **Resources:** code + weights, and a public **CLSS viewer** for exploration.

> See paper for full details, datasets, ablations, and comparisons.

---

## Quick Start

### Installation

```bash
pip install clss-model
```

### Examples

Complete examples are available in the [`examples/`](examples/) directory:

- **[`examples/training/`](examples/training/)** - Full training pipeline
  - `train.py` - Main training script with PyTorch Lightning
  - `dataset.py` - ECOD dataset loading and preprocessing  
  - `args.py` - Command-line argument parsing
  - `infra.py` - Infrastructure setup (distributed training, logging)

- **[`examples/inference/`](examples/inference/)** - Inference and embedding
  - `infer.py` - Protein sequence and structure embedding
  - `sample-pdbs/` - Example PDB files for testing

---

## Data

* **ECOD‑AF2 domains** (training/validation set).

---

## Citation

If you use this repository, please cite:

```bibtex
@article{Yanai2025CLSS,
  title={Contrastive learning unites sequence and structure in a global representation of protein space},
  author={Yanai, Guy and Axel, Gabriel and Longo, Liam M. and Ben-Tal, Nir and Kolodny, Rachel},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.09.05.674454},
  url={https://www.biorxiv.org/content/10.1101/2025.09.05.674454v3.full.pdf}
}
```

---

## Acknowledgments & Contact

* See the paper for funding and acknowledgments.
* Correspondence: [llongo@elsi.jp](mailto:llongo@elsi.jp), [bental@tauex.tau.ac.il](mailto:bental@tauex.tau.ac.il), [trachel@cs.haifa.ac.il](mailto:trachel@cs.haifa.ac.il).
