"""Image loader: caption images with Granite Vision, emit plain-text records.

Granite Vision is used for exactly one thing — turning an image into a short
textual description. The caption then flows through the *unchanged* pipeline
(granite-embedding / BM25 / dense retrieval, Granite instruct generation) as
an ordinary text chunk; nothing downstream ever sees pixels.

VRAM discipline (hard requirement for shared cluster nodes): the vision model
is a multi-GB guest next to the embedding and generation models, so it is
loaded on demand inside :func:`caption_images` and torn down in a ``finally``
block — ``del model`` + ``gc.collect()`` + ``torch.cuda.empty_cache()`` — so
ingestion never holds GPU memory once captioning is done, even on error.

torch/transformers are imported lazily inside the functions, so importing
this module stays cheap on machines without the ML stack.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import List, Sequence

from src.ingestion.loaders.base import LoadedDocument

DEFAULT_VISION_MODEL_ID = "ibm-granite/granite-vision-3.3-2b"

# Aimed at retrieval: pull out the text, numbers and relations a user would
# query for, not just an aesthetic one-liner.
DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail. Include any visible text and numbers, "
    "chart axes and trends, and the relationships between the objects shown."
)

#: Suffixes the directory dispatcher will treat as images.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}


def caption_images(
    paths: Sequence[str | Path],
    *,
    prompt: str = DEFAULT_CAPTION_PROMPT,
    model_id: str | None = None,
    device: str | None = None,
    max_new_tokens: int = 256,
) -> List[LoadedDocument]:
    """Caption ``paths`` with Granite Vision and return one record per image.

    Loads the vision model **once** for the whole batch (so bulk ingestion
    doesn't reload multi-GB weights per image) and releases model + VRAM
    before returning. Each record carries ``source_type="image"``,
    ``file_name`` and the absolute ``image_path`` for RAG back-links.

    Parameters
    ----------
    model_id:
        Hugging Face repo id. Defaults to the ``GRANITE_VISION_MODEL_ID`` env
        var, then :data:`DEFAULT_VISION_MODEL_ID`.
    device:
        ``"auto"`` | ``"cuda"`` | ``"cpu"``. Defaults to the ``LLM_DEVICE``
        env var, then ``"auto"`` (same convention as ``src.llm_client``).
    """
    import torch

    image_paths = [Path(p) for p in paths]
    missing = [str(p) for p in image_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Image(s) not found: {', '.join(missing)}")
    if not image_paths:
        return []

    model_id = model_id or os.getenv("GRANITE_VISION_MODEL_ID") or DEFAULT_VISION_MODEL_ID
    device = device or os.getenv("LLM_DEVICE", "auto")

    model = processor = None
    try:
        model, processor = _load_vision_model(model_id, device)
        records = []
        for image_path in image_paths:
            caption = _caption_one(model, processor, image_path, prompt, max_new_tokens)
            records.append(
                LoadedDocument(
                    doc_id=image_path.stem,
                    text=caption,
                    metadata={
                        "source_type": "image",
                        "file_name": image_path.name,
                        "image_path": str(image_path.resolve()),
                    },
                )
            )
        return records
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_vision_model(model_id: str, device: str):
    """Load Granite Vision processor + model (mirrors ``LLMClient._init_client``)."""
    from transformers import AutoProcessor

    try:  # transformers >= 4.52 name, with fallback for older releases
        from transformers import AutoModelForImageTextToText as AutoVisionModel
    except ImportError:  # pragma: no cover - depends on installed transformers
        from transformers import AutoModelForVision2Seq as AutoVisionModel

    token = os.getenv("HUGGINGFACE_API_KEY") or None
    cache_dir = os.getenv("MODEL_CACHE_DIR") or None

    processor = AutoProcessor.from_pretrained(model_id, token=token, cache_dir=cache_dir)
    model = AutoVisionModel.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
        dtype="auto",
        device_map="auto" if device == "auto" else None,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()
    return model, processor


def _caption_one(model, processor, path: Path, prompt: str, max_new_tokens: int) -> str:
    """Generate a caption for a single image (greedy, reproducible)."""
    import torch
    from PIL import Image

    image = Image.open(path).convert("RGB")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    # Decode only the newly generated tokens, not the echoed prompt.
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()
