from huggingface_hub import hf_hub_download
from model import CLSSModel

local_path = hf_hub_download(
    repo_id="guyyanai/CLSS",
    filename="h32_r10.lckpt",
    repo_type="model"
)

model = CLSSModel.load_from_checkpoint(local_path)
