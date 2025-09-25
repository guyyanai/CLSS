"""
Utilities for CLSS package.
"""

from huggingface_hub import hf_hub_download


def download_pretrained_model(
    repo_id: str = "guyyanai/CLSS", model_name: str = "h32_r10.lckpt"
) -> str:
    """Download pretrained model from Hugging Face Hub."""

    print(f"Downloading model {model_name} from {repo_id}")

    local_path = hf_hub_download(
        repo_id=repo_id, filename=model_name, repo_type="model"
    )

    print(f"Model downloaded to {local_path}")
    return local_path
