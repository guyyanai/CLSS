"""
This module defines the CLSS (Contrastive Learning for Sequence and Structure) model.

The CLSS model is a PyTorch Lightning module that learns joint representations of
protein sequences and structures using a contrastive learning approach. It utilizes
pre-trained ESM-2 for sequence encoding and ESM-3 for structure encoding.
"""

from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from transformers import EsmModel, EsmTokenizer

from .config import CLSSConfig
from .utils import download_pretrained_model


class CLSSModel(pl.LightningModule):
    def __init__(
        self,
        esm2_checkpoint: str,
        hidden_dim: int,
        learning_rate: float = 1e-3,
        init_temperature: float = 0.5,
        should_learn_temperature: bool = False,
        random_sequence_stretches: bool = True,
        random_stretch_min_size: int = 10,
        use_global_loss: bool = False,
        should_load_esm3: bool = False,
    ):
        """
        Initialize the CLSS model.
        Args:
            esm2_checkpoint (str): Path or name of the ESM2 checkpoint.
            hidden_dim (int): Dimension of the hidden layer for projections.
            learning_rate (float): Learning rate for the optimizer.
            init_temperature (float): Initial temperature for contrastive loss scaling.
            should_learn_temperature (bool): Whether temperature is learnable.
            random_sequence_stretches (bool): Enable random sequence stretches.
            random_stretch_min_size (int): Minimum size for random stretches.
            use_global_loss (bool): Use global loss across all GPUs.
            should_load_esm3 (bool): Whether to load ESM3 structure encoder.
        """
        super(CLSSModel, self).__init__()
        self.save_hyperparameters()

        # Store the hidden dimension
        self.hidden_dim = hidden_dim

        # Load the pre-trained ESM2 model
        self.load_esm2(esm2_checkpoint)

        # Create ESM2 (sequence) adapter
        self.sequence_adapter = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.sequence_encoder.config.hidden_size, hidden_dim),
        )

        # Initialize structure encoder
        self.structure_encoder = None

        # Load ESM3 (structure) model if needed
        if should_load_esm3:
            self.load_esm3()

        # Create ESM3 (structure) adapter
        self.structure_adapter = nn.Sequential(nn.Linear(1536, hidden_dim))

        # Create temperature parameter
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )

        # Disable training for temperature parameter if needed
        if not should_learn_temperature:
            self.temperature.requires_grad = False

        self.learning_rate = learning_rate
        self.random_sequence_stretches = random_sequence_stretches
        self.random_stretch_min_size = random_stretch_min_size
        self.use_global_loss = use_global_loss
        self.should_load_esm3 = should_load_esm3

    @classmethod
    def from_config(
        cls, config: CLSSConfig, should_load_esm3: bool = False, **kwargs
    ) -> "CLSSModel":
        """
        Create a CLSSModel from a CLSSConfig.

        Args:
            config: CLSSConfig instance
            should_load_esm3: Whether to load ESM3 structure encoder
            **kwargs: Additional arguments to override config values

        Returns:
            CLSSModel instance
        """
        # Convert config to dict and update with any overrides
        config_dict = config.to_dict()
        config_dict.update(kwargs)

        # Map config parameters to model parameters
        model_kwargs = {
            "esm2_checkpoint": config_dict["esm2_checkpoint"],
            "hidden_dim": config_dict["hidden_dim"],
            "learning_rate": config_dict["learning_rate"],
            "init_temperature": config_dict["init_temperature"],
            "should_learn_temperature": config_dict["should_learn_temperature"],
            "random_sequence_stretches": config_dict["random_sequence_stretches"],
            "random_stretch_min_size": config_dict["random_stretch_min_size"],
            "use_global_loss": config_dict["use_global_loss"],
            "should_load_esm3": config_dict["should_load_esm3"],
        }

        return cls(**model_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "guyyanai/CLSS",
        model_name: str = "h32_r10.lckpt",
        device: str = "cuda",
    ) -> "CLSSModel":
        """
        Load a pretrained CLSS model.

        Args:
            model_name: Name of the model file to download
            repo_id: Hugging Face repository ID

        Returns:
            CLSSModel
        """
        # Download model
        model_path = download_pretrained_model(repo_id=repo_id, model_name=model_name)

        # Load model
        return cls.load_from_checkpoint(checkpoint_path=model_path, map_location=device)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "CLSSModel":
        """
        Load CLSS model from local checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            CLSSModel
        """
        return cls.load_from_checkpoint(checkpoint_path)

    def load_esm2(self, checkpoint: str) -> None:
        # Load the pre-trained ESM2 tokenizer & model
        """
        Load the pre-trained ESM2 tokenizer and model.
        Disables training for LM and contact heads.
        Args:
            checkpoint (str): Path or name of the ESM2 checkpoint.
        Returns:
            Tuple[EsmModel, EsmTokenizer]: Loaded model and tokenizer.
        """
        tokenizer = EsmTokenizer.from_pretrained(checkpoint)
        model: EsmModel = EsmModel.from_pretrained(checkpoint)  # type: ignore

        # Disable training for both LM and contact heads
        for parameter_name, parameter in list(model.named_parameters()):
            if "lm_head" in parameter_name:
                parameter.requires_grad = False

            if "contact_head" in parameter_name:
                parameter.requires_grad = False

        self.sequence_encoder = model
        self.sequence_tokenizer = tokenizer

    def load_esm3(self, checkpoint=ESM3_OPEN_SMALL) -> None:
        """
        Load the pre-trained ESM3 structure encoder model.
        Disables training for all parameters.
        Args:
            checkpoint (str): ESM3 checkpoint identifier.
        Returns:
            ESM3: Loaded ESM3 model.
        """
        model = ESM3.from_pretrained(checkpoint)

        for parameter in model.parameters():
            parameter.requires_grad = False

        self.structure_encoder = model

    def embed_sequence_residues(self, sequences: List[str]) -> List[torch.Tensor]:
        """
        Embed each residue in a list of protein sequences using ESM2.
        Args:
            sequences (List[str]): List of protein sequences.
        Returns:
            List[torch.Tensor]: List of per-residue embeddings (N, L, D).
        """
        embedding_list = []

        for sequence in sequences:
            tokenized_sequence = self.sequence_tokenizer(
                sequence, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output = self.sequence_encoder(**tokenized_sequence)

            embedding_list.append(output.last_hidden_state[0])

        return embedding_list

    def embed_sequences(
        self,
        sequences: List[str],
        apply_adapter: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed a list of protein sequences using ESM2 and potentially project to hidden_dim.
        Args:
            sequences (List[str]): List of protein sequences.
            apply_adapter (bool): Whether to apply the adapter projection.
            normalize (bool): Whether to normalize the embeddings.
        Returns:
            torch.Tensor: Normalized sequence embeddings of shape (N, D).
        """
        esm_embeddings = torch.stack(
            [
                embedding.mean(dim=0)
                for embedding in self.embed_sequence_residues(sequences)
            ]
        )

        if not apply_adapter:
            return esm_embeddings

        embeddings = self.sequence_adapter(esm_embeddings)

        if not normalize:
            return embeddings

        normalized_embeddings = F.normalize(embeddings, dim=-1)

        return normalized_embeddings

    def embed_structure_residues(
        self, structures: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Embed each residue in a list of protein structures using ESM3.
        Args:
            structures (List[torch.Tensor]): List of structure tensors.
        Returns:
            List[torch.Tensor]: List of per-residue embeddings (N, L, D).
        """
        if self.structure_encoder is None:
            raise Exception(
                "Structure encoder (ESM3) wasn't loaded, please make sure the 'should_load_esm3' flag is enabled or call load_esm3()."
            )

        structures = [structure.to(self.device) for structure in structures]

        esm_proteins = [ESMProtein(coordinates=structure) for structure in structures]
        embedding_list = []

        for esm_protein in esm_proteins:
            protein_tensor = self.structure_encoder.encode(esm_protein)

            with torch.no_grad():
                output = self.structure_encoder.forward_and_sample(
                    protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
                )

            embedding_list.append(output.per_residue_embedding)

        return embedding_list

    def embed_structures(
        self,
        structures: List[torch.Tensor],
        apply_adapter: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Embed a list of protein structures using ESM3 and potentially project to hidden_dim.
        Args:
            structures (List[torch.Tensor]): List of structure tensors.
            apply_adapter (bool): Whether to apply the adapter projection.
            normalize (bool): Whether to normalize the embeddings.
        Returns:
            torch.Tensor: Normalized structure embeddings of shape (N, D).
        """
        esm_embeddings = torch.stack(
            [
                embedding.mean(dim=0)
                for embedding in self.embed_structure_residues(structures)
            ]
        )

        if not apply_adapter:
            return esm_embeddings

        embeddings = self.structure_adapter(esm_embeddings)

        if not normalize:
            return embeddings

        normalized_embeddings = F.normalize(embeddings, dim=-1)

        return normalized_embeddings

    def forward(
        self,
        batched_input_ids: torch.Tensor,
        batched_attention_mask: torch.Tensor,
        batched_structure_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a batch of sequences and structures.
        Args:
            batched_input_ids (torch.Tensor): Batch of input IDs for sequences.
            batched_attention_mask (torch.Tensor): Batch of attention masks.
            batched_structure_embeddings (torch.Tensor): Batch of structure embeddings.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sequence and structure projections.
        """
        # Allocate list for sequence outputs
        sequence_outputs = []

        # Create tensors for tracking sequence/stretch lengths
        sequence_lengths = torch.zeros(batched_input_ids.shape[0])
        stretch_lengths = torch.zeros(batched_input_ids.shape[0])

        # Iterate through each sequence in the batch
        for index, (input_ids, attention_mask) in enumerate(
            zip(batched_input_ids, batched_attention_mask)
        ):
            # Update sequence length
            sequence_length = attention_mask.count_nonzero().item()
            sequence_lengths[index] = sequence_length

            # Sample random sequence stretch
            if self.random_sequence_stretches:
                input_ids, attention_mask, stretch_length = (
                    self.sample_sequence_stretch(
                        input_ids, attention_mask, sequence_length
                    )
                )

                stretch_lengths[index] = stretch_length

            # Run sequence/stretch through the sequence tower
            sequence_output = self.sequence_encoder(
                input_ids=input_ids[:sequence_length].unsqueeze(0),
                attention_mask=attention_mask[:sequence_length].unsqueeze(0),
            )

            # Take the mean pooling of the last hidden state
            sequence_output = sequence_output.last_hidden_state.mean(dim=1)[0]
            sequence_outputs.append(sequence_output)

        # Log sequence metrics
        self.log("sequence_mean_length", sequence_lengths.mean(), logger=True)

        if self.random_sequence_stretches:
            self.log("random_stretch_mean_length", stretch_lengths.mean(), logger=True)

        sequence_outputs_tensor = torch.stack(sequence_outputs)

        # Apply adapters
        sequence_projections = self.sequence_adapter(sequence_outputs_tensor)
        structure_projections = self.structure_adapter(batched_structure_embeddings)

        return sequence_projections, structure_projections

    def sample_sequence_stretch(
        self, input_ids, attention_mask, sequence_length
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a random stretch from a sequence.
        Args:
            input_ids (torch.Tensor): Input IDs for the sequence.
            attention_mask (torch.Tensor): Attention mask for the sequence.
            sequence_length (int): Length of the sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: Sampled input IDs, attention mask, and stretch length.
        """
        if sequence_length < self.random_stretch_min_size:
            return input_ids, attention_mask, sequence_length

        start_index = torch.randint(
            0, sequence_length - self.random_stretch_min_size + 1, (1,)
        ).item()
        max_length = sequence_length - start_index
        stretch_length = torch.randint(
            self.random_stretch_min_size, max_length + 1, (1,)
        ).item()

        sampled_input_ids = input_ids[start_index : start_index + stretch_length]
        sampled_attention_mask = attention_mask[
            start_index : start_index + stretch_length
        ]

        return sampled_input_ids, sampled_attention_mask, stretch_length

    def gather_projections(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Gather projections from all GPUs for global loss computation.
        Args:
            projections (torch.Tensor): Projections to gather.
        Returns:
            torch.Tensor: Gathered projections.
        """
        gathered = self.all_gather(projections, sync_grads=True)

        # Reshape tensors if necessary
        if isinstance(gathered, list):
            gathered = torch.cat(gathered, dim=0)
        else:
            gathered = gathered.view(-1, projections.size(-1))  # type: ignore

        return gathered

    def contrastive_loss(
        self,
        projections1: torch.Tensor,
        projections2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between two sets of projections.
        Args:
            projections1 (torch.Tensor): First set of projections.
            projections2 (torch.Tensor): Second set of projections.
        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Gather projections from all GPUs
        if self.use_global_loss:
            projections1 = self.gather_projections(projections1)
            projections2 = self.gather_projections(projections2)

        self.log("loss_total_samples", projections1.shape[0], prog_bar=True)

        # Normalize the projections
        projections1 = F.normalize(projections1, dim=1)
        projections2 = F.normalize(projections2, dim=1)

        # Compute cosine similarity
        similarities = torch.mm(projections1, projections2.T)
        scaled_similarities = similarities * self.temperature

        # Labels for contrastive learning: diagonal elements should match
        labels = torch.arange(projections1.size(0), device=self.device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(scaled_similarities, labels)

        # Log individual components (example: log mean similarity of positive pairs)
        pos_similarity = similarities.detach().diag().mean().cpu()
        scaled_pos_similarity = scaled_similarities.detach().diag().mean().cpu()

        self.log("pos_similarity", pos_similarity, prog_bar=True, logger=True)
        self.log("scaled_pos_similarity", scaled_pos_similarity, logger=True)
        self.log(
            "temperature", self.temperature.detach().cpu(), prog_bar=True, logger=True
        )

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Training step for contrastive learning.
        Args:
            batch: Batch containing input_ids, attention_mask, structure_embeddings.
            batch_idx: Index of the batch.
        Returns:
            torch.Tensor: Training loss.
        """
        # Assume batch contains paired sequences for contrastive learning
        input_ids, attention_mask, structure_embeddings = batch

        # Log the number of samples in this GPU's batch
        self.log(
            f"train_batch_size_per_gpu_{self.trainer.global_rank}",
            structure_embeddings.shape[0],
            prog_bar=True,
        )

        # Forward pass for both pairs
        sequence_projections, structure_projections = self(
            input_ids, attention_mask, structure_embeddings
        )

        # Compute contrastive loss
        loss = self.contrastive_loss(sequence_projections, structure_projections)
        self.log("train_loss", loss.detach(), prog_bar=True, logger=True)

        # Log additional metrics
        learning_rate = self.optimizers().param_groups[0]["lr"]  # type: ignore
        self.log("learning_rate", learning_rate, prog_bar=True, logger=True)

        gradient_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.log(
            "gradient_norm", gradient_norm.detach().cpu(), prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Validation step for contrastive learning.
        Args:
            batch: Batch containing input_ids, attention_mask, structure_embeddings.
            batch_idx: Index of the batch.
        Returns:
            torch.Tensor: Validation loss.
        """
        # Assume batch contains paired sequences for contrastive learning
        input_ids, attention_mask, structure_embeddings = batch

        # Log the number of samples in this GPU's batch
        self.log(
            f"validation_batch_size_per_gpu_{self.trainer.global_rank}",
            structure_embeddings.shape[0],
            prog_bar=True,
        )

        # Forward pass for sequence and structure
        sequence_projections, structure_projections = self(
            input_ids, attention_mask, structure_embeddings
        )

        # Compute contrastive loss
        val_loss = self.contrastive_loss(sequence_projections, structure_projections)
        self.log(
            "val_loss", val_loss.detach(), prog_bar=True, logger=True, sync_dist=True
        )

        return val_loss

    def configure_optimizers(self) -> torch.optim.Adam:
        """
        Configure the optimizer for training.
        Returns:
            torch.optim.Optimizer: Adam optimizer.
        """
        # Set up optimizer
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint) -> None:
        """
        Callback to modify checkpoint before saving.
        Removes structure_encoder weights from checkpoint.
        Args:
            checkpoint (dict): Checkpoint dictionary.
        """
        # Strip frozen model's weights before saving
        for key in list(checkpoint["state_dict"]):
            if key.startswith("structure_encoder"):
                del checkpoint["state_dict"][key]
