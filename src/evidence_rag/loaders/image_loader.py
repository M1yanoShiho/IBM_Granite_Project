"""Image loader: caption images with Granite Vision, emit plain-text ``Document``s.

Granite Vision is used for exactly one thing — turning an image into a short
textual description. Because a caption is lossy, images are also run through
Docling OCR by default and any recognised text is appended to the same
document, which materially improves retrieval for text-heavy images. The
vision model lives inside :class:`VisionCaptioner`, a context manager that
loads the weights once per batch and releases GPU memory unconditionally on
exit, so cluster nodes never keep multi-GB vision weights resident between
ingestion runs.
"""

import gc
import importlib
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from evidence_rag.contracts.models import Document, SourceMetadata

logger = logging.getLogger("evidence_rag.loaders")

IMAGE_LOADER_VERSION = "image-loader-v2"

DEFAULT_VISION_MODEL_ID = "ibm-granite/granite-vision-3.3-2b"

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail. Include any visible text and numbers, "
    "chart axes and trends, and the relationships between the objects shown."
)

MAX_IMAGE_DIMENSION = 2048

OnError = Literal["skip", "raise"]


class VisionCaptioner:
    """Context manager owning the Granite Vision model lifecycle.

    ``__enter__`` loads the processor and model once for a whole ingestion
    batch; ``__exit__`` unconditionally drops them, runs ``gc.collect()`` and
    empties the CUDA cache — even when captioning raised mid-batch.
    """

    def __init__(
        self,
        *,
        prompt: str = DEFAULT_CAPTION_PROMPT,
        model_id: str | None = None,
        device: str | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.prompt = prompt
        self.model_id = model_id or os.getenv("GRANITE_VISION_MODEL_ID") or DEFAULT_VISION_MODEL_ID
        self.device = device or os.getenv("LLM_DEVICE") or "auto"
        self.max_new_tokens = max_new_tokens
        self._processor: Any = None
        self._model: Any = None

    def __enter__(self) -> "VisionCaptioner":
        self._processor, self._model = _load_vision_model(self.model_id, self.device)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        model = self._model
        processor = self._processor
        self._model = None
        self._processor = None
        del model, processor
        _release_accelerator_memory()

    def caption(self, image: Any) -> str:
        """Caption a PIL image (greedy decoding, reproducible)."""
        if self._model is None or self._processor is None:
            raise RuntimeError("VisionCaptioner must be entered before captioning")
        prepared = prepare_pil_image(image)
        return _generate_caption(
            self._processor, self._model, prepared, self.prompt, self.max_new_tokens
        ).strip()


def caption_images(
    paths: Sequence[str | Path],
    *,
    prompt: str = DEFAULT_CAPTION_PROMPT,
    model_id: str | None = None,
    device: str | None = None,
    max_new_tokens: int = 512,
    on_error: OnError = "skip",
    include_ocr: bool = True,
    converter: Any | None = None,
) -> list[Document]:
    """Caption ``paths`` with Granite Vision and return one ``Document`` per image.

    The vision model is loaded once for the whole batch and released before
    the (optional) OCR pass, keeping peak memory to one model at a time. A
    failing image is skipped with a warning by default (``on_error="skip"``);
    pass ``on_error="raise"`` to fail fast — memory is released either way.
    ``include_ocr`` appends Docling-recognised text to each caption; pass an
    existing Docling ``converter`` to reuse it.
    """
    resolved = [Path(path).resolve() for path in paths]
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Image(s) not found: {', '.join(missing)}")
    if not resolved:
        return []

    captioner = VisionCaptioner(
        prompt=prompt, model_id=model_id, device=device, max_new_tokens=max_new_tokens
    )
    with captioner:
        captions = caption_image_paths(captioner, resolved, on_error=on_error)

    # OCR runs after the vision model is released so only one model is resident.
    return [
        build_image_document(path, caption, include_ocr=include_ocr, converter=converter)
        for path, caption in captions
    ]


def caption_image_paths(
    captioner: VisionCaptioner,
    paths: Sequence[Path],
    *,
    on_error: OnError = "skip",
) -> list[tuple[Path, str]]:
    """Caption image files with an already-entered captioner, honouring ``on_error``."""
    captions: list[tuple[Path, str]] = []
    for path in paths:
        try:
            caption = captioner.caption(_open_image(path))
        except Exception:
            if on_error == "raise":
                raise
            logger.warning("Captioning failed for %s; skipping", path, exc_info=True)
            continue
        if not caption:
            logger.warning("Empty caption for %s; skipping", path)
            continue
        captions.append((path, caption))
    return captions


def build_image_document(
    path: Path, caption: str, *, include_ocr: bool, converter: Any | None
) -> Document:
    """Wrap a caption (plus optional OCR text) into a standalone image ``Document``."""
    text = caption
    if include_ocr:
        ocr_text = extract_ocr_text(path, converter=converter)
        if ocr_text:
            text = f"{caption}\n\nText in image:\n{ocr_text}"
    return Document(
        document_id=path.name,
        text=text,
        source_uri=str(path),
        metadata=SourceMetadata(
            source_type="image",
            file_name=path.name,
            image_path=str(path),
        ),
    )


def extract_ocr_text(path: Path, *, converter: Any | None = None) -> str:
    """Run Docling OCR over an image file; empty string when unavailable or failed."""
    try:
        if converter is None:
            document_converter = importlib.import_module("docling.document_converter")
            converter = document_converter.DocumentConverter()
        docling_document = converter.convert(path).document
        return str(docling_document.export_to_markdown()).strip()
    except ImportError:
        logger.warning("docling unavailable; skipping OCR for %s (caption only)", path.name)
        return ""
    except Exception:
        logger.warning("OCR failed for %s; caption only", path, exc_info=True)
        return ""


def _open_image(path: Path) -> Any:
    """Open an image, honouring the EXIF orientation tag (phone photos)."""
    try:
        pil_image = importlib.import_module("PIL.Image")
        image_ops = importlib.import_module("PIL.ImageOps")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "Image captioning requires the optional 'pillow' package "
            "(install the 'ingestion' extra)."
        ) from exc
    with pil_image.open(path) as image:
        return image_ops.exif_transpose(image).convert("RGB")


def prepare_pil_image(image: Any) -> Any:
    """Normalise a PIL image for the vision model: RGB, capped resolution."""
    prepared = image.convert("RGB")
    prepared.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION))
    return prepared


def _load_vision_model(model_id: str, device: str) -> tuple[Any, Any]:
    """Load Granite Vision processor + model (mirrors ``GraniteLLMClient._load_model``)."""
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "Image captioning requires the optional 'transformers' package "
            "(install the 'granite' extra)."
        ) from exc

    token = os.getenv("HUGGINGFACE_API_KEY") or None
    cache_dir = os.getenv("MODEL_CACHE_DIR") or None
    processor = transformers.AutoProcessor.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
    )
    model_cls = getattr(
        transformers,
        "AutoModelForImageTextToText",
        None,
    ) or transformers.AutoModelForVision2Seq
    model = model_cls.from_pretrained(
        model_id,
        token=token,
        cache_dir=cache_dir,
        dtype="auto",
        device_map="auto" if device == "auto" else None,
    )
    if device != "auto":
        model = model.to(device)
    model.eval()
    return processor, model


def _generate_caption(
    processor: Any,
    model: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate a caption for a prepared PIL image (greedy, reproducible).

    Single-image path only: the ``output_ids[0]`` / ``input_ids.shape[-1]``
    slicing assumes batch size 1. Batched captioning would need per-sequence
    prompt lengths (padding-aware) — rewrite the slicing before adding it.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise RuntimeError(
            "Image captioning requires the optional 'torch' package "
            "(install the 'granite' extra)."
        ) from exc

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
    )
    model_device = getattr(model, "device", None)
    if model_device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(model_device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return str(processor.decode(new_tokens, skip_special_tokens=True))


def _release_accelerator_memory() -> None:
    """Free the vision model's memory: collect Python refs, then empty the CUDA cache."""
    gc.collect()
    try:
        torch = importlib.import_module("torch")
    except ImportError:  # pragma: no cover - depends on optional runtime deps
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
