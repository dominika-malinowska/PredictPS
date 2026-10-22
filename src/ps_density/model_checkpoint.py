"""Resolve the paper U-Net checkpoint from Hugging Face or a local path."""

from huggingface_hub import hf_hub_download

PAPER_HF_REPO_ID = "dominika-malinowska/ps_predictions"
PAPER_HF_FILENAME = "seed_54321_20231214-203338.ckpt"


def resolve_paper_checkpoint(cfg):
    """Return a local path to the 2-channel paper checkpoint.

    Uses ``test.checkpoint`` from ``paths.yaml`` when that string is set.
    Otherwise downloads ``PAPER_HF_FILENAME`` from Hugging Face (cached).
    """
    test_cfg = getattr(cfg, "test", None)
    local = getattr(test_cfg, "checkpoint", None) if test_cfg is not None else None
    if isinstance(local, str) and local.strip():
        path = local.strip()
        print("Using local checkpoint:", path, flush=True)
        return path
    path = hf_hub_download(repo_id=PAPER_HF_REPO_ID, filename=PAPER_HF_FILENAME)
    print(
        "Using Hugging Face checkpoint:",
        f"{PAPER_HF_REPO_ID}/{PAPER_HF_FILENAME}",
        flush=True,
    )
    print("Cached at:", path, flush=True)
    return path
