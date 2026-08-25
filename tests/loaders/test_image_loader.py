import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evidence_rag.loaders import image_loader


class FakeTorch:
    """Stand-in torch module recording CUDA cache releases."""

    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.cuda = SimpleNamespace(
            is_available=lambda: True,
            empty_cache=self._empty_cache,
        )

    def _empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeImage:
    def __init__(self, name: str) -> None:
        self.name = name


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> FakeTorch:
    fake = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake)  # type: ignore[arg-type]
    return fake


def _patch_vision(
    monkeypatch: pytest.MonkeyPatch, captions: dict[str, str], load_counter: list[int] | None = None
) -> None:
    def fake_load(model_id: str, device: str) -> tuple[Any, Any]:
        if load_counter is not None:
            load_counter.append(1)
        return "fake-processor", "fake-model"

    def fake_generate(
        processor: Any, model: Any, image: Any, prompt: str, max_new_tokens: int
    ) -> str:
        caption = captions[image.name]
        if caption == "<raise>":
            raise RuntimeError(f"caption failed for {image.name}")
        return caption

    monkeypatch.setattr(image_loader, "_load_vision_model", fake_load)
    monkeypatch.setattr(image_loader, "_open_image", lambda path: FakeImage(path.name))
    monkeypatch.setattr(image_loader, "prepare_pil_image", lambda image: image)
    monkeypatch.setattr(image_loader, "_generate_caption", fake_generate)


def _write_images(tmp_path: Path, *names: str) -> list[Path]:
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_bytes(b"fake-image")
        paths.append(path)
    return paths


def test_caption_images_emits_documents_and_releases_memory(
    monkeypatch: pytest.MonkeyPatch, fake_torch: FakeTorch, tmp_path: Path
) -> None:
    chart, photo = _write_images(tmp_path, "chart.png", "photo.jpg")
    loads: list[int] = []
    _patch_vision(
        monkeypatch,
        {"chart.png": "A bar chart of rising revenue.", "photo.jpg": "  "},
        load_counter=loads,
    )

    documents = image_loader.caption_images([chart, photo], include_ocr=False)

    assert [document.document_id for document in documents] == ["chart.png"]
    assert documents[0].text == "A bar chart of rising revenue."
    assert documents[0].source_uri == str(chart.resolve())
    assert documents[0].metadata is not None
    assert documents[0].metadata.source_type == "image"
    assert documents[0].metadata.image_path == str(chart.resolve())
    assert loads == [1]
    assert fake_torch.empty_cache_calls == 1


def test_on_error_skip_continues_batch_and_releases(
    monkeypatch: pytest.MonkeyPatch,
    fake_torch: FakeTorch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    broken, chart = _write_images(tmp_path, "broken.png", "chart.png")
    _patch_vision(monkeypatch, {"broken.png": "<raise>", "chart.png": "A chart."})

    with caplog.at_level("WARNING", logger="evidence_rag.loaders"):
        documents = image_loader.caption_images([broken, chart], include_ocr=False)

    assert [document.document_id for document in documents] == ["chart.png"]
    assert "Captioning failed" in caplog.text
    assert fake_torch.empty_cache_calls == 1


def test_on_error_raise_propagates_and_still_releases(
    monkeypatch: pytest.MonkeyPatch, fake_torch: FakeTorch, tmp_path: Path
) -> None:
    (broken,) = _write_images(tmp_path, "broken.png")
    _patch_vision(monkeypatch, {"broken.png": "<raise>"})

    with pytest.raises(RuntimeError, match="caption failed"):
        image_loader.caption_images([broken], include_ocr=False, on_error="raise")

    assert fake_torch.empty_cache_calls == 1


def test_missing_file_raises_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_load(model_id: str, device: str) -> tuple[Any, Any]:
        raise AssertionError("model must not be loaded for missing files")

    monkeypatch.setattr(image_loader, "_load_vision_model", unexpected_load)

    with pytest.raises(FileNotFoundError):
        image_loader.caption_images([tmp_path / "missing.png"])


def test_empty_batch_skips_model_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_load(model_id: str, device: str) -> tuple[Any, Any]:
        raise AssertionError("model must not be loaded for an empty batch")

    monkeypatch.setattr(image_loader, "_load_vision_model", unexpected_load)

    assert image_loader.caption_images([]) == []


def test_include_ocr_appends_recognised_text(
    monkeypatch: pytest.MonkeyPatch, fake_torch: FakeTorch, tmp_path: Path
) -> None:
    (chart,) = _write_images(tmp_path, "chart.png")
    _patch_vision(monkeypatch, {"chart.png": "A chart."})
    monkeypatch.setattr(image_loader, "extract_ocr_text", lambda path, converter: "Q4: 12%")

    documents = image_loader.caption_images([chart], include_ocr=True)

    assert documents[0].text == "A chart.\n\nText in image:\nQ4: 12%"


def test_extract_ocr_text_returns_empty_on_converter_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (chart,) = _write_images(tmp_path, "chart.png")

    class FailingConverter:
        def convert(self, path: Path) -> Any:
            raise RuntimeError("no OCR backend")

    with caplog.at_level("WARNING", logger="evidence_rag.loaders"):
        assert image_loader.extract_ocr_text(chart, converter=FailingConverter()) == ""
    assert "OCR failed" in caplog.text


def test_extract_ocr_text_uses_supplied_converter(tmp_path: Path) -> None:
    (chart,) = _write_images(tmp_path, "chart.png")

    class OkConverter:
        def convert(self, path: Path) -> Any:
            document = SimpleNamespace(export_to_markdown=lambda: "  recognised text  ")
            return SimpleNamespace(document=document)

    assert image_loader.extract_ocr_text(chart, converter=OkConverter()) == "recognised text"


def test_extract_ocr_text_from_image_uses_supplied_converter(tmp_path: Path) -> None:
    class FakePilImage:
        def convert(self, mode: str) -> "FakePilImage":
            return self

        def save(self, path: str, format: str) -> None:
            Path(path).write_bytes(b"png-bytes")

    class OkConverter:
        def convert(self, path: Path) -> Any:
            document = SimpleNamespace(export_to_markdown=lambda: "  chart text  ")
            return SimpleNamespace(document=document)

    assert (
        image_loader.extract_ocr_text_from_image(FakePilImage(), converter=OkConverter())
        == "chart text"
    )


def test_extract_ocr_text_from_image_empty_on_unpreparable_image(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BadImage:
        def convert(self, mode: str) -> Any:
            raise RuntimeError("not an image")

    with caplog.at_level("WARNING", logger="evidence_rag.loaders"):
        assert image_loader.extract_ocr_text_from_image(BadImage(), converter=None) == ""
    assert "caption only" in caplog.text


def test_open_image_applies_exif_orientation(tmp_path: Path) -> None:
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("RGB", (10, 20), "red")
    exif = pil.Exif()
    exif[0x0112] = 6  # rotate 90 degrees
    path = tmp_path / "photo.jpg"
    image.save(path, exif=exif)

    opened = image_loader._open_image(path)

    assert opened.size == (20, 10)


def test_prepare_pil_image_caps_resolution() -> None:
    pil = pytest.importorskip("PIL.Image")
    huge = pil.new("RGB", (5000, 2500))

    prepared = image_loader.prepare_pil_image(huge)

    assert max(prepared.size) == image_loader.MAX_IMAGE_DIMENSION
