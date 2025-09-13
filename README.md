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

## Key ideas (from the paper)

* **Two-tower architecture:** sequence tower (ESM2‑like, \~35M params) co-trained; structure tower (ESM3) kept frozen; both feed **32‑D L2‑normalized adapters**.
* **Segment-aware training:** contrastive pairs match **full-domain structures** with **random sequence sub-segments (≥10 aa)** to encode contextual compatibility.
* **Unified embeddings:** sequences, structures, and subsequences align in a **single space**; distances track ECOD hierarchy and reveal cross-fold relationships.
* **Scale & efficiency:** \~36M trainable params, compact embeddings (32‑D) supporting efficient storage and search.
* **Resources:** code + weights, and a public **CLSS viewer** for exploration.

> See paper for full details, datasets, ablations, and comparisons.

---

## Installation

```bash
# clone your repository
git clone https://github.com/<your-username>/CLSS.git
cd CLSS

# create env (example with conda)
conda create -n clss python=3.10 -y
conda activate clss

# install Python deps
pip install -r requirements.txt
```

> If you use CUDA, ensure PyTorch/Lightning versions match your system. See your `requirements.txt`.

---

## Quickstart

### Inference (embeddings)

```bash
python infer.py \
  --checkpoint weights/clss.pt \
  --input data/example_sequences.fasta \
  --output outputs/embeddings.npy
```

* **Input:** FASTA sequences (optionally PDB/CIF structures if your script supports structure embeddings).
* **Output:** N×32 NumPy array of L2‑normalized embeddings.

### Training (from scratch or fine-tuning)

```bash
python train.py \
  --config configs/train.yaml \
  --data data/ecod_af2_train_index.json \
  --outdir runs/exp01
```

Key config fields (example):

```yaml
model:
  seq_encoder: esm2_35m
  str_encoder: esm3_frozen
  embed_dim: 32
  temperature: 0.5
train:
  epochs: 80
  batch_size: 1440        # effective batch across GPUs
  min_seg_len: 10         # min aa for random subsequences
  gpus: 8                 # adjust to your setup
```

---

## Data

* **ECOD‑AF2 domains** (training/validation and Dataset 1 in the paper).
* **CATH metamorphic set** (Dataset 2 in the paper).
* Prepare your own indices listing domain IDs, sequence files, and (optionally) structure files. Include minimal examples under `data/`.

---

## Repository structure (suggested)

```
CLSS/
├─ configs/               # YAML configs
├─ clss/                  # Python package (models, data, utils)
│  ├─ models/
│  ├─ data/
│  └─ nn/
├─ scripts/               # preprocessing, evaluation, export
├─ train.py               # training entry point
├─ infer.py               # inference entry point
├─ requirements.txt
└─ README.md
```

---

## Reproducing paper figures

* Explore the **CLSS viewer** for t‑SNE maps: [https://gabiaxel.github.io/clss-viewer/](https://gabiaxel.github.io/clss-viewer/)
* Provide a script (e.g., `scripts/tsne_map.py`) to project and plot embeddings for your datasets.

---

## Pretrained weights

* Provide links or bundle a release asset, e.g., `weights/clss.pt`.
* If you mirror the paper’s weights, reference the original release/source.

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

## License

* **Paper:** CC BY‑NC 4.0 (see bioRxiv page).
* **Code:** Add your chosen license in `LICENSE` (e.g., MIT/Apache‑2.0).

---

## Acknowledgments & Contact

* See the paper for funding and acknowledgments.
* Correspondence (from the paper): [llongo@elsi.jp](mailto:llongo@elsi.jp), [bental@tauex.tau.ac.il](mailto:bental@tauex.tau.ac.il), [trachel@cs.haifa.ac.il](mailto:trachel@cs.haifa.ac.il).
