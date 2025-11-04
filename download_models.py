import os
import sys

# Disable XET BEFORE importing huggingface_hub
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import snapshot_download

models = [
    "facebook/mms-tts-eng",
    "facebook/mms-tts-tam",
    "ramsrigouthamg/t5_paraphraser",
    "felflare/bert-restore-punctuation"
]

for model in models:
    print(f"\n{'='*60}")
    print(f"Downloading {model}...")
    print(f"{'='*60}")
    try:
        snapshot_download(repo_id=model)
        print(f"✓ {model} downloaded successfully")
    except Exception as e:
        print(f"✗ {model} failed: {str(e)[:100]}")
        sys.exit(1)

print("\n" + "="*60)
print("All models downloaded successfully!")
print("="*60)
