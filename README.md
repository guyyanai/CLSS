# CLSS: Contrastive learning unites sequence and structure in a global representation of protein space

**Paper:** [https://www.biorxiv.org/content/10.1101/2025.09.05.674454.full.pdf](https://www.biorxiv.org/content/10.1101/2025.09.05.674454.full.pdf)

**DOI:** [https://doi.org/10.1101/2025.09.05.674454](https://doi.org/10.1101/2025.09.05.674454)

**GitHub repository:** [https://github.com/guyyanai/CLSS](https://github.com/guyyanai/CLSS)

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

### Architecture

<p align="center">
  <img src="figures/clss-architecture.png" alt="CLSS Architecture" width="800"/>
</p>

**Figure 1:** Overview of training, validating, and testing of CLSS to create unified maps of protein sequence and structure space. (A) Overview of the two-tower CLSS architecture. On the left is a structure tower based on the frozen, pre-trained ESM3 model (light blue) followed by a trained (yellow) adapter that averages, reduces dimension, and normalizes the embedding. On the right is the trained CLSS sequence tower, build upon a pre-trained ESM2 model, and its adapter network (yellow). The networks are trained using contrastive loss on batches of randomly chosen structures and sequence segments from the ECOD-AF2 domain database. Labels from a hierarchical classification were not using during training in any way. Once trained, we calculate the embeddings of the structures, sequences, and sequence segments from Datasets 1 and 2 using CLSS (B) and other PLMs (C). (D) Dimensionality reduction by t-SNE was used to create visual maps of protein space (upper images). Pairwise distance distributions (lower images) were calculated from the embeddings directly, rather than the t-SNE reduced space.

---

## Visualization

CLSS embeddings capture the global organization of protein space, revealing evolutionary relationships and structural similarities across diverse protein domains.

<p align="center">
  <img src="figures/ecod-tsne.png" alt="ECOD t-SNE Visualization" width="800"/>
</p>

**Figure 2:** CLSS embedding maps of ECOD domains (Dataset 1). For each domain, we calculate the embeddings by three modalities – structure, sequence, and a random sequence segment – and then compute a t-SNE projection of the embeddings. Each point represents one of the modalities of a domain colored according to the label of its ECOD architecture. (A) An overlay of all three modalities. Sequences are marked by circles, structures by ‘+’, and random sequence segments by ‘x’. (B) Structure embeddings. (C) sequence segment embeddings. (D) Sequence embeddings. We find that the maps of all three modalities are very similar to each other, with the sequence (D) and structure (B) embeddings being the closest. This shows that CLSS successfully injected structure information into the sequence modality. The global organization of the CLSS embedding space positions domains with the same ECOD architecture, and even the same structure class, near each other. 

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

- **[`examples/interactive-map/`](examples/interactive-map/)** - Interactive visualization
  - `app.py` - Complete pipeline from data to interactive HTML visualization
  - `mapper.py` - Plotly-based interactive scatter plot creation
  - `dataset.py` - Multi-modal data loading (FASTA/PDB)
  - `embeddings.py` - CLSS model inference and embedding generation
  - `dim_reducer.py` - t-SNE dimensionality reduction

---

## Data

* **ECOD‑AF2 domains** (training/validation set) - Available in `datasets/training/`
* **F40-large-folds** (Dataset 1 from paper) - Available in `datasets/F40-large-folds/`
  - Contains all ECOD-PDB-F40 domains in folds with more than 50 domains

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
