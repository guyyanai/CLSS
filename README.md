# CLSS: Contrastive learning unites sequence and structure in a global representation of protein space

This repository contains the official implementation for the paper: **"Contrastive learning unites sequence and structure in a global representation of protein space"** (Yanai et al., Conference/Journal 2025).

[Link to Paper](https://example.com) (coming soon)

## Abstract

*Amino acid sequence dictates the three-dimensional structure and biological function of proteins. Yet, despite decades of research, our understanding of the interplay between sequence and structure is incomplete. To meet this challenge, we introduce Contrastive Learning Sequence-Structure (CLSS), an AI-based contrastive learning model trained to co-embed sequence and structure information in a self-supervised manner. We trained CLSS on large and diverse sets of protein building blocks called domains. CLSS represents both sequences and structures as vectors in the same high-dimensional space, where distance relates to sequence-structure similarity. Thus, CLSS provides a natural way to represent the protein universe, reflecting structural and evolutionary relationships among all known domains. We find that CLSS refines expert knowledge about the global organization of protein space, and highlights transitional forms that resist hierarchical classification. CLSS reveals linkage between domains of seemingly separate lineages, thereby significantly improving our understanding of evolutionary design.*

## Features

-   **Contrastive Learning**: Employs a powerful self-supervised learning technique to learn from unlabeled protein data.
-   **Pre-trained Models**: Leverages state-of-the-art protein language models: ESM-2 for sequences and ESM-3 for structures.
-   **Efficient Training**: Built with PyTorch Lightning for streamlined and scalable training loops.
-   **Distributed Training**: Supports multi-GPU and multi-node training using Distributed Data Parallel (DDP) and SLURM.
-   **Experiment Tracking**: Integrates with Weights & Biases (Wandb) for logging metrics, and model checkpoints.
-   **Data Augmentation**: Includes an option for using random sequence stretches as a form of data augmentation.
-   **Efficient Data Handling**: Caches pre-processed datasets to speed up subsequent training runs.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/guyyanai/CLSS.git
    cd CLSS
    ```

2.  **Create a Conda environment and install dependencies:**
    It is recommended to use a Conda environment to manage dependencies.

    ```bash
    conda create -n clss python=3.9
    conda activate clss
    ```

3.  **Install PyTorch:**
    Install PyTorch with CUDA support according to the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/).

4.  **Install other dependencies:**
    ```bash
    pip install pytorch-lightning transformers 'esm[esm3]' wandb pandas scikit-learn tqdm
    ```

## Dataset

The training script expects the data to be organized as follows:

1.  **A CSV file**: This file should contain a list of ECOD domain IDs to be used for training/validation. It should have a column named `ecod_uid`.
2.  **A directory of PDB files**: This directory should contain the 3D structure files in PDB format. The files should be organized in a way that they can be located using the ECOD ID. The default script assumes a path like `.../{ecod_id[2:7]}/{ecod_id}/{ecod_id}.pdb`.

You can specify the paths to the CSV file and the structures directory using command-line arguments.

## Usage

The main training script is `training/train.py`. You can run it from the root directory of the project.

### Reproducing Paper Results

To reproduce the results from the paper, you can run the following command:

```bash
torchrun --nproc_per_node=4 training/train.py \
    --batch-size 32 \
    --hidden-projection-dim 128 \
    --learning-rate 1e-4 \
    --epochs 50 \
    --dataset-path /path/to/your/dataset.csv \
    --structures-dir /path/to/your/structures \
    --train-pickle "/path/to/cache/train.pkl" \
    --validation-pickle "/path/to/cache/val.pkl" \
    --run-name "clss-paper-reproduction" \
    --random-sequence-stretches \
    --learn-temperature
```

### Command-Line Arguments

For a full list of arguments and their descriptions, run:
```bash
python training/train.py --help
```

## Citation

If you find this work useful in your research, please consider citing our paper:

```bibtex
@article{YourName2025CLSS,
  title={Contrastive learning unites sequence and structure in a global representation of protein space},
  author={Guy Yanai, Gabriel Axel, Liam M. Longo, Nir Ben-Tal, Rachel Kolodny},
  journal={Conference or Journal Name},
  year={2025},
  volume={1},
  number={1},
  pages={1-10}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.
