from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="fpadovani/communicative-baby-dpo-synthetic",
    local_dir="finetuned_models/communicative-baby-dpo-synthetic",
    local_dir_use_symlinks=False  # Optional: avoids symlinks, useful for portability
)