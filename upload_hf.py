from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
import os

# === CONFIGURATION ===
username = "fpadovani"  # <-- CHANGE THIS to your Hugging Face username
model_name = "communicative-baby-dpo-synthetic"
base_checkpoint_dir = "/home3/p318482/communicative_baby_dpo/dpo_outputs_complete_synthetic"
checkpoints = ["checkpoint-2000", "checkpoint-4000", "checkpoint-5630"]
full_repo_name = f"{username}/{model_name}"
visibility = "public"  # or "public"

# === STEP 1: Create the HF repo if it doesn't exist ===
api = HfApi()
token = HfFolder.get_token()

try:
    create_repo(
        repo_id=full_repo_name,
        token=token,
        exist_ok=True,
        private=(visibility == "private")
    )
    print(f"Repository '{full_repo_name}' created or already exists.")
except Exception as e:
    print(f"Error creating repo: {e}")
    exit(1)

# === STEP 2: Upload each checkpoint into its own subfolder ===
for checkpoint in checkpoints:
    folder_path = os.path.join(base_checkpoint_dir, checkpoint)
    print(f"Uploading: {folder_path} → {checkpoint}/ on Hugging Face Hub...")

    upload_folder(
        repo_id=full_repo_name,
        folder_path=folder_path,
        path_in_repo=checkpoint,  # Upload to a subdirectory (e.g., checkpoint-2000)
        commit_message=f"Upload {checkpoint}",
        token=token
    )

print(f"✅ All checkpoints uploaded to: https://huggingface.co/{full_repo_name}")
