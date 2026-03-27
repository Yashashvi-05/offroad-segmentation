from huggingface_hub import HfApi

repo_id = "Yashashvi-0508/offroad-segmentation-segformer-b2"
file_path = "runs/best_segformer_b4_v5.pth"

api = HfApi()

print(f"🧹 Scanning repository '{repo_id}' for old files...")

try:
    # Get all files currently in the repo
    existing_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    
    # Delete any old model weights so the judges don't get confused
    for file in existing_files:
        if file.endswith(".pth") or file.endswith(".bin") or file.endswith(".safetensors"):
            print(f"🗑️ Deleting old model file: {file}")
            api.delete_file(path_in_repo=file, repo_id=repo_id, repo_type="model")
            
    print("✅ Cleanup complete. Repository is ready for the final model.")
except Exception as e:
    print(f"Note on cleanup: {e}")

print(f"\n🚀 Uploading the final 0.4838 mIoU model ({file_path})...")
print("⏳ This will take a few minutes (File size: ~250MB)...")

# Upload the new model
api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo="best_segformer_b4_v5.pth",
    repo_id=repo_id,
    repo_type="model"
)

print("\n🎉 SUCCESS! Your Hugging Face repository now contains ONLY your final winning model.")