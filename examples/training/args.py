"""
This module defines the command-line arguments for the training script.

It uses argparse to handle arguments for batch size, model checkpoints, dataset paths,
hyperparameters, and other training configurations.
"""

import argparse


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Batch size of training set",
        required=True,
    )
    parser.add_argument(
        "--esm-checkpoint",
        dest="esm_checkpoint",
        type=str,
        help="Checkpoint of ESM model",
        default="facebook/esm2_t12_35M_UR50D",
    )
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        help="Path to contrastive dataset",
        required=True,
    )
    parser.add_argument(
        "--train-pickle",
        dest="train_pickle_file",
        type=str,
        help="Path to train pickle file",
        required=False,
    )
    parser.add_argument(
        "--validation-pickle",
        dest="validation_pickle_file",
        type=str,
        help="Path to validation pickle file",
        required=False,
    )
    parser.add_argument(
        "--structures-dir",
        dest="structures_dir",
        type=str,
        help="Path to directory containing all structures",
        required=True,
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        help="Learning rate",
        default=1e-3,
    )
    parser.add_argument(
        "--learn-temperature",
        dest="learn_temperature",
        help="Should learn temperature parameter",
        action="store_true",
    )
    parser.add_argument(
        "--init-temperature",
        dest="init_temperature",
        type=float,
        help="Initial contrastive loss temperature",
        default=0.5,
    )
    parser.add_argument(
        "--projection-dim",
        dest="hidden_projection_dim",
        type=int,
        help="Non-linear projection dimension",
        required=True,
    )
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        help="Number of training epochs",
        default=80,
    )
    parser.add_argument(
        "--dataset-limit",
        dest="dataset_size_limit",
        type=int,
        help="Limit to the size of the dataset",
        default=1000000,
    )
    parser.add_argument(
        "--validation-frac",
        dest="validation_dataset_frac",
        type=float,
        help="Fracture of the validation set out of the dataset",
        default=0.2,
    )
    parser.add_argument(
        "--checkpoint-path",
        dest="checkpoint_path",
        type=str,
        help="Path to folder which the model checkpoint will be saved to",
        required=True,
    )
    parser.add_argument("--seed", dest="seed", type=int, help="Random seed", default=0)
    parser.add_argument(
        "--run-name",
        dest="run_name",
        type=str,
        help="Name of the current run",
        required=False,
    )
    parser.add_argument(
        "--wandb-project",
        dest="wandb_project",
        type=str,
        help="WandB project name",
        required=True,
    )
    parser.add_argument(
        "--random-sequence-stretches",
        dest="random_sequence_stretches",
        help="Should use random sequence stretches",
        action="store_true",
    )
    parser.add_argument(
        "--random-stretch-min-size",
        dest="random_stretch_min_size",
        help="Minimum size for random sequence stretches",
        type=int,
        default=30,
    )
    return parser.parse_args()
