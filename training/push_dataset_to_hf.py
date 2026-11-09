"""Upload the COCO dataset to a PRIVATE Hugging Face dataset repo.

Every upload creates a new commit -> the HF webhook fires -> remote training starts automatically.

    huggingface-cli login            (or set HF_TOKEN)
    python -m training.push_dataset_to_hf --src signature_dataset_coco --repo sevenminds/signature-dataset
"""
import argparse

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="signature_dataset_coco")
ap.add_argument("--repo", required=True)
a = ap.parse_args()

api = HfApi()
api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)   # NETSOL confidential: keep private
if hasattr(api, "upload_large_folder"):          # resumable, best for big datasets
    api.upload_large_folder(repo_id=a.repo, repo_type="dataset", folder_path=a.src)
else:                                            # older/newer huggingface_hub without it
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=a.src,
                      commit_message="Update signature dataset")
print(f"Uploaded {a.src} -> https://huggingface.co/datasets/{a.repo} (private)")
