import os
from pathlib import Path

# Typical Hugging Face cache location on macOS
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

if hf_cache.exists():
    total_size = sum(f.stat().st_size for f in hf_cache.rglob('*') if f.is_file())
    print(f"Cache size: {total_size / (1024**3):.2f} GB")
    
    # List recent files
    print("\nRecent files:")
    for file in sorted(hf_cache.rglob('*'), key=lambda x: x.stat().st_mtime, reverse=True)[:10]:
        if file.is_file():
            print(f"  {file.name}: {file.stat().st_size / (1024**2):.2f} MB")