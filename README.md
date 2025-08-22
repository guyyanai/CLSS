# CLSS: Contrastive Learning for Protein Sequence and Structure

CLSS is a PyTorch Lightning-based project for learning joint representations of protein sequences and their 3D structures. It uses a contrastive learning objective to align the embeddings from a sequence model (ESM-2) and a structure model (ESM-3).

## Overview

The core idea is to train a model that can understand the relationship between a protein's amino acid sequence and its folded structure. This is achieved by:

1.  **Encoding Sequences**: Using the pre-trained ESM-2 model to generate embeddings from protein sequences.
2.  **Encoding Structures**: Using the pre-trained ESM-3 model to generate embeddings from PDB structures.
3.  **Contrastive Loss**: Training projection heads on top of the encoders using a contrastive loss function. This encourages the sequence and structure embeddings for the same protein to be close in the embedding space, while being distant from embeddings of other proteins.

The project is built with scalability in mind, leveraging PyTorch Lightning for features like multi-GPU training, mixed-precision, and easy-to-use logging.

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
    git clone <repository-url>
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

### Example Command

Here is an example command to start a training run:

```bash
torchrun --nproc_per_node=4 training/train.py 
    --batch-size 32 
    --hidden-projection-dim 128 
    --learning-rate 1e-4 
    --epochs 50 
    --dataset-path /path/to/your/dataset.csv 
    --structures-dir /path/to/your/structures 
    --train-pickle "/path/to/cache/train.pkl" 
    --validation-pickle "/path/to/cache/val.pkl" 
    --run-name "clss-training-run-1" 
    --random-sequence-stretches 
    --learn-temperature
```

### Command-Line Arguments

-   `--batch-size`: Training batch size.
-   `--hidden-projection-dim`: The dimension of the non-linear projection head.
-   `--learning-rate`: The learning rate for the optimizer.
-   `--epochs`: Number of training epochs.
-   `--dataset-path`: Path to the CSV file with ECOD IDs.
-   `--structures-dir`: Path to the directory containing PDB files.
-   `--train-pickle`: Path to save/load the cached training dataset.
-   `--validation-pickle`: Path to save/load the cached validation dataset.
-   `--run-name`: A name for the Weights & Biases run.
-   `--random-sequence-stretches`: (flag) Enable data augmentation using random sequence stretches.
-   `--learn-temperature`: (flag) Make the temperature parameter of the contrastive loss learnable.

For a full list of arguments, run:
```bash
python training/train.py --help
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.
