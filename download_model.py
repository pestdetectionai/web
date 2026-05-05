from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_REPO = "underdogquality/yolo11s-pest-detection"
MODEL_FILE = "best.pt"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading pest detection model...")
print(f"Repo: {MODEL_REPO}")
print(f"File: {MODEL_FILE}")

downloaded_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False
)

print("Model downloaded successfully:")
print(downloaded_path)