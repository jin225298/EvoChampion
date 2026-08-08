"""
Shared HuggingFace cache utilities for the EvoChampion system.

Provides resolve_model_path() to map HuggingFace model IDs to local
snapshot paths, bypassing the hf-xet monkey-patch that crashes
from_pretrained on GPU nodes without internet access.
"""

import os
from pathlib import Path


def resolve_model_path(model_name_or_path: str) -> str:
    """Resolve a HuggingFace model ID to its local cache snapshot path.

    This bypasses the hf-xet issue where from_pretrained finds
    'pytorch_model.bin from cache at None' and crashes with
    AttributeError even with HF_HUB_OFFLINE=1.

    Resolution order:
      1. If model_name_or_path is an existing local directory, return as-is.
      2. If it looks like a HuggingFace model ID, resolve to local snapshot.
      3. If resolution fails, return the original string.

    Args:
        model_name_or_path: Either a local path or a HuggingFace model ID.

    Returns:
        The resolved local snapshot path, or the original string if
        resolution failed.
    """
    # If it's already a local path that exists, use it directly
    if Path(model_name_or_path).is_dir():
        return model_name_or_path

    # Try to resolve as HuggingFace model ID
    cache_dir = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"
    # Normalize repository IDs to the HuggingFace cache directory naming scheme.
    model_dir = cache_dir / f"models--{model_name_or_path.replace('/', '--')}"

    if model_dir.is_dir():
        # Prefer the revision pointed to by refs/main
        refs_main = model_dir / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text().strip()
            snapshot_dir = model_dir / "snapshots" / revision
            if snapshot_dir.is_dir():
                print(
                    f"[hf_cache] Resolved {model_name_or_path} -> {snapshot_dir}"
                )
                return str(snapshot_dir)

        # Fallback: use the latest snapshot by modification time
        snapshots = model_dir / "snapshots"
        if snapshots.is_dir():
            revisions = sorted(
                snapshots.iterdir(),
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if revisions:
                print(
                    f"[hf_cache] Resolved {model_name_or_path} -> {revisions[0]} (latest snapshot)"
                )
                return str(revisions[0])

    # Could not resolve
    print(f"[hf_cache] Could not resolve model to local path: {model_name_or_path}")
    return model_name_or_path
