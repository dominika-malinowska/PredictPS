"""Resolve the paper U-Net checkpoint from Hugging Face or a local path.

The 2-channel paper weights are cached by ``huggingface_hub`` when
``test.checkpoint`` or ``predict.checkpoint`` is empty.
"""

from huggingface_hub import hf_hub_download

PAPER_HF_REPO_ID = "dominika-malinowska/ps_predictions"
PAPER_HF_FILENAME = "seed_54321_20231214-203338.ckpt"


def _checkpoint_string(local):
    """Extract a checkpoint path string from a YAML value or paths object."""
    if local is None:
        return None
    if isinstance(local, str):
        return local
    nested = getattr(local, "checkpoint", None)
    if nested is not None:
        return _checkpoint_string(nested)
    test_cfg = getattr(local, "test", None)
    if test_cfg is not None:
        return _checkpoint_string(getattr(test_cfg, "checkpoint", None))
    predict_cfg = getattr(local, "predict", None)
    if predict_cfg is not None:
        return _checkpoint_string(getattr(predict_cfg, "checkpoint", None))
    return None


def resolve_paper_checkpoint(local=None):
    """Return a local path to the 2-channel paper checkpoint.

    Uses ``local`` when it is a non-empty string (typically
    ``test.checkpoint`` or ``predict.checkpoint``). A paths object is also
    accepted: the function reads ``.checkpoint``, or ``test.checkpoint`` /
    ``predict.checkpoint``. Otherwise downloads ``PAPER_HF_FILENAME`` from
    Hugging Face (cached).
    """
    path = _checkpoint_string(local)
    if isinstance(path, str) and path.strip():
        path = path.strip()
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
