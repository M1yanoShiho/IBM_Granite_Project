from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evidence_rag.infrastructure.config import IngestionConfig
from evidence_rag.loaders import dispatch, image_loader


class FakeImage:
    def __init__(self, name: str) -> None:
        self.name = name


class FakePicture:
    def __init__(self, image: FakeImage, page: int | None) -> None:
        self._image = image
        self.prov = [SimpleNamespace(page_no=page)] if page is not None else []

    def get_image(self, docling_document: object) -> FakeImage:
        return self._image


class FakeDoc:
    def __init__(
        self,
        pages: dict[int, str] | None = None,
        pictures: list[FakePicture] | None = None,
        markdown: str = "",
    ) -> None:
        self.pages = pages or {}
        self.pictures = pictures or []
        self._markdown = markdown

    def export_to_markdown(self, page_no: int | None = None) -> str:
        if page_no is None:
            return self._markdown
        return self.pages[page_no]


class FakeConverter:
    def __init__(self, docs: dict[str, FakeDoc]) -> None:
        self.docs = docs
        self.converted: list[str] = []

    def convert(self, path: Path) -> SimpleNamespace:
        name = Path(path).name
        self.converted.append(name)
        return SimpleNamespace(document=self.docs[name])


def _patch_vision(
    monkeypatch: pytest.MonkeyPatch, captions: dict[str, str]
) -> tuple[list[int], list[str]]:
    loads: list[int] = []
    generated: list[str] = []

    def fake_load(model_id: str, device: str) -> tuple[Any, Any]:
        loads.append(1)
        return "fake-processor", "fake-model"

    def fake_generate(
        processor: Any, model: Any, image: Any, prompt: str, max_new_tokens: int
    ) -> str:
        generated.append(image.name)
        return captions[image.name]

    monkeypatch.setattr(image_loader, "_load_vision_model", fake_load)
    monkeypatch.setattr(image_loader, "_open_image", lambda path: FakeImage(path.name))
    monkeypatch.setattr(image_loader, "prepare_pil_image", lambda image: image)
    monkeypatch.setattr(image_loader, "_generate_caption", fake_generate)
    return loads, generated


def test_routes_by_extension_and_loads_vision_model_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "notes.txt").write_text("plain notes", encoding="utf-8")
    (tmp_path / "readme.md").write_text("markdown notes", encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "chart.png").write_bytes(b"fake-png")
    (tmp_path / "photo.JPG").write_bytes(b"fake-jpg")
    (tmp_path / "data.parquet").write_bytes(b"unsupported")
    (tmp_path / "subdir").mkdir()
    converter = FakeConverter({"report.pdf": FakeDoc(pages={1: "Page one."})})
    loads, _ = _patch_vision(
        monkeypatch, {"chart.png": "Chart caption.", "photo.JPG": "Photo caption."}
    )

    documents = dispatch.load_directory(
        tmp_path, converter=converter, pdf_mode="pages", image_ocr=False
    )

    # File order is preserved; captions produced later slot back into place.
    assert [document.document_id for document in documents] == [
        "chart.png",
        "notes.txt",
        "photo.JPG",
        "readme.md",
        "report.pdf::p1",
    ]
    assert loads == [1]
    assert converter.converted == ["report.pdf"]


def test_caption_pdf_pictures_shares_the_vision_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "report.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "chart.png").write_bytes(b"fake-png")
    converter = FakeConverter(
        {
            "report.pdf": FakeDoc(
                pages={1: "Body text."},
                pictures=[FakePicture(FakeImage("embedded"), page=2)],
            )
        }
    )
    loads, generated = _patch_vision(
        monkeypatch, {"chart.png": "A chart.", "embedded": "An embedded diagram."}
    )

    documents = dispatch.load_directory(
        tmp_path,
        converter=converter,
        pdf_mode="pages",
        caption_pdf_pictures=True,
        image_ocr=False,
    )

    assert loads == [1]
    assert sorted(generated) == ["chart.png", "embedded"]
    by_id = {document.document_id: document for document in documents}
    picture_doc = by_id["report.pdf::p2::img1"]
    assert picture_doc.text == "An embedded diagram."
    assert picture_doc.source_uri.endswith("report.pdf#page=2")
    assert picture_doc.metadata is not None
    assert picture_doc.metadata.source_type == "image"
    assert picture_doc.metadata.page_number == 2
    assert "report.pdf::p1" in by_id


def test_caption_pdf_pictures_appends_ocr_text(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "report.pdf").write_bytes(b"%PDF-fake")
    converter = FakeConverter(
        {
            "report.pdf": FakeDoc(
                pages={1: "Body text."},
                pictures=[FakePicture(FakeImage("embedded"), page=2)],
            )
        }
    )
    _patch_vision(monkeypatch, {"embedded": "An embedded diagram."})
    monkeypatch.setattr(
        dispatch,
        "extract_ocr_text_from_image",
        lambda image, converter: "Revenue up 12%" if image.name == "embedded" else "",
    )

    documents = dispatch.load_directory(
        tmp_path,
        converter=converter,
        pdf_mode="pages",
        caption_pdf_pictures=True,
        image_ocr=True,
    )

    by_id = {document.document_id: document for document in documents}
    picture_doc = by_id["report.pdf::p2::img1"]
    assert picture_doc.text == "An embedded diagram.\n\nText in image:\nRevenue up 12%"


def test_caption_pdf_pictures_skips_ocr_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "report.pdf").write_bytes(b"%PDF-fake")
    converter = FakeConverter(
        {
            "report.pdf": FakeDoc(
                pages={1: "Body text."},
                pictures=[FakePicture(FakeImage("embedded"), page=2)],
            )
        }
    )
    _patch_vision(monkeypatch, {"embedded": "An embedded diagram."})

    def _fail_ocr(image: Any, converter: Any) -> str:
        raise AssertionError("OCR must not run when image_ocr=False")

    monkeypatch.setattr(dispatch, "extract_ocr_text_from_image", _fail_ocr)

    documents = dispatch.load_directory(
        tmp_path,
        converter=converter,
        pdf_mode="pages",
        caption_pdf_pictures=True,
        image_ocr=False,
    )

    by_id = {document.document_id: document for document in documents}
    assert by_id["report.pdf::p2::img1"].text == "An embedded diagram."


def test_pdf_cache_invalidated_by_image_ocr_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "report.pdf").write_bytes(b"%PDF-fake")
    cache_dir = tmp_path / "cache"

    def _converter() -> FakeConverter:
        return FakeConverter(
            {
                "report.pdf": FakeDoc(
                    pages={1: "Body text."},
                    pictures=[FakePicture(FakeImage("embedded"), page=2)],
                )
            }
        )

    loads, _ = _patch_vision(monkeypatch, {"embedded": "An embedded diagram."})
    monkeypatch.setattr(dispatch, "extract_ocr_text_from_image", lambda image, converter: "42%")

    dispatch.load_directory(
        corpus,
        converter=_converter(),
        pdf_mode="pages",
        caption_pdf_pictures=True,
        image_ocr=True,
        cache_dir=cache_dir,
    )
    # Same inputs but a different OCR flag must miss the cache and recaption.
    dispatch.load_directory(
        corpus,
        converter=_converter(),
        pdf_mode="pages",
        caption_pdf_pictures=True,
        image_ocr=False,
        cache_dir=cache_dir,
    )

    assert loads == [1, 1]


def test_load_directory_from_config_maps_switches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_load_directory(directory: Any, **kwargs: Any) -> list[Any]:
        captured["directory"] = directory
        captured.update(kwargs)
        return []

    monkeypatch.setattr(dispatch, "load_directory", fake_load_directory)
    config = IngestionConfig(
        pdf_mode="pages",
        caption_pdf_pictures=True,
        image_ocr=False,
        caption_prompt="Describe the chart.",
        cache_dir=tmp_path / "cache",
    )

    dispatch.load_directory_from_config(tmp_path, config)

    assert captured["directory"] == tmp_path
    assert captured["pdf_mode"] == "pages"
    assert captured["caption_pdf_pictures"] is True
    assert captured["image_ocr"] is False
    assert captured["caption_prompt"] == "Describe the chart."
    assert captured["cache_dir"] == tmp_path / "cache"
    # Unset optional fields must not be forwarded (keep load_directory's defaults).
    assert "vision_model_id" not in captured
    assert "vision_device" not in captured


def test_load_directory_from_config_omits_unset_optionals(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        dispatch, "load_directory", lambda directory, **kwargs: captured.update(kwargs) or []
    )

    dispatch.load_directory_from_config(tmp_path, IngestionConfig())

    for optional in ("caption_prompt", "vision_model_id", "vision_device", "cache_dir"):
        assert optional not in captured
    assert captured["caption_pdf_pictures"] is False
    assert captured["image_ocr"] is True


def test_image_ocr_appends_text_via_shared_converter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "chart.png").write_bytes(b"fake-png")
    converter = FakeConverter({"chart.png": FakeDoc(markdown="Q4: 12%")})
    _patch_vision(monkeypatch, {"chart.png": "A chart."})

    documents = dispatch.load_directory(tmp_path, converter=converter, image_ocr=True)

    assert documents[0].text == "A chart.\n\nText in image:\nQ4: 12%"


def test_cache_skips_reparse_and_recaption_until_content_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "report.pdf").write_bytes(b"%PDF-fake")
    (corpus / "chart.png").write_bytes(b"fake-png")
    converter = FakeConverter({"report.pdf": FakeDoc(pages={1: "Body text."})})
    _, generated = _patch_vision(monkeypatch, {"chart.png": "A chart."})
    cache_dir = tmp_path / "cache"

    first = dispatch.load_directory(
        corpus, converter=converter, pdf_mode="pages", image_ocr=False, cache_dir=cache_dir
    )
    second = dispatch.load_directory(
        corpus, converter=converter, pdf_mode="pages", image_ocr=False, cache_dir=cache_dir
    )

    assert second == first
    assert converter.converted == ["report.pdf"]  # parsed once, cached afterwards
    assert generated == ["chart.png"]  # captioned once, cached afterwards

    (corpus / "chart.png").write_bytes(b"fake-png-v2")
    third = dispatch.load_directory(
        corpus, converter=converter, pdf_mode="pages", image_ocr=False, cache_dir=cache_dir
    )

    assert generated == ["chart.png", "chart.png"]  # content change re-captions
    assert [document.document_id for document in third] == [
        "chart.png",
        "report.pdf::p1",
    ]


def test_captioner_uses_the_same_model_id_as_the_cache_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "chart.png").write_bytes(b"fake-png")
    monkeypatch.setenv("GRANITE_VISION_MODEL_ID", "org/custom-vision-model")
    received: list[str] = []

    def fake_load(model_id: str, device: str) -> tuple[Any, Any]:
        received.append(model_id)
        return "fake-processor", "fake-model"

    monkeypatch.setattr(image_loader, "_load_vision_model", fake_load)
    monkeypatch.setattr(image_loader, "_open_image", lambda path: FakeImage(path.name))
    monkeypatch.setattr(image_loader, "prepare_pil_image", lambda image: image)
    monkeypatch.setattr(
        image_loader,
        "_generate_caption",
        lambda processor, model, image, prompt, max_new_tokens: "A chart.",
    )

    dispatch.load_directory(tmp_path, image_ocr=False)

    # dispatch resolves the env fallback once and hands that exact id to the
    # captioner, so cache fingerprints can never drift from the model in use.
    assert received == ["org/custom-vision-model"]


def test_unsupported_extensions_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "data.parquet").write_bytes(b"unsupported")

    with caplog.at_level("DEBUG", logger="evidence_rag.loaders"):
        documents = dispatch.load_directory(tmp_path)

    assert documents == []
    assert "Skipping unsupported file" in caplog.text


def test_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        dispatch.load_directory(tmp_path / "missing")


def test_nested_files_are_ingested_and_keyed_by_relative_path(tmp_path: Path) -> None:
    (tmp_path / "top.txt").write_text("top", encoding="utf-8")
    nested = tmp_path / "2024" / "q1"
    nested.mkdir(parents=True)
    (nested / "notes.txt").write_text("nested", encoding="utf-8")

    documents = dispatch.load_directory(tmp_path)

    assert [document.document_id for document in documents] == [
        "2024/q1/notes.txt",
        "top.txt",
    ]


def test_same_file_name_in_two_subdirectories_keeps_distinct_ids(
    tmp_path: Path,
) -> None:
    # The reason ids became relative paths: a bare name collides once the walk is
    # recursive, and duplicate document_ids break the corpus contract downstream.
    for folder in ("alpha", "beta"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "report.txt").write_text(folder, encoding="utf-8")

    documents = dispatch.load_directory(tmp_path)

    ids = [document.document_id for document in documents]
    assert ids == ["alpha/report.txt", "beta/report.txt"]
    assert len(set(ids)) == len(ids)


def test_flat_directory_ids_are_unchanged_by_recursion(tmp_path: Path) -> None:
    # Backward-compat guard: corpora that predate recursive scanning must keep the
    # exact ids they had, or every persisted index signature built on them breaks.
    (tmp_path / "notes.txt").write_text("flat", encoding="utf-8")

    recursive = dispatch.load_directory(tmp_path)
    flat = dispatch.load_directory(tmp_path, recursive=False)

    assert [document.document_id for document in recursive] == ["notes.txt"]
    assert [document.document_id for document in flat] == ["notes.txt"]


def test_recursive_false_still_skips_nested_files(tmp_path: Path) -> None:
    (tmp_path / "top.txt").write_text("top", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.txt").write_text("nested", encoding="utf-8")

    documents = dispatch.load_directory(tmp_path, recursive=False)

    assert [document.document_id for document in documents] == ["top.txt"]


def test_pdf_chunk_ids_keep_their_suffix_under_a_relative_id(tmp_path: Path) -> None:
    nested = tmp_path / "reports"
    nested.mkdir()
    (nested / "report.pdf").write_bytes(b"%PDF-fake")
    converter = FakeConverter({"report.pdf": FakeDoc(pages={1: "Page one."})})

    documents = dispatch.load_directory(
        tmp_path, converter=converter, pdf_mode="pages", image_ocr=False
    )

    # The file-name part is re-keyed, the ::p1 chunk suffix survives intact.
    assert [document.document_id for document in documents] == ["reports/report.pdf::p1"]


def test_a_failed_file_is_skipped_and_the_rest_still_load(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Before this, a parse error propagated: one bad file aborted the whole ingest
    # partway through, discarding every document already parsed.
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "fine.txt").write_text("readable", encoding="utf-8")

    class ExplodingConverter:
        converted: list[str] = []

        def convert(self, path: Path) -> Any:
            raise RuntimeError("corrupt pdf")

    with caplog.at_level("WARNING", logger="evidence_rag.loaders"):
        documents = dispatch.load_directory(tmp_path, converter=ExplodingConverter())

    assert [document.document_id for document in documents] == ["fine.txt"]
    assert "Failed to parse" in caplog.text
    assert "1 file(s) failed to parse" in caplog.text


def test_on_error_raise_still_fails_fast_on_a_bad_file(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-fake")

    class ExplodingConverter:
        def convert(self, path: Path) -> Any:
            raise RuntimeError("corrupt pdf")

    with pytest.raises(RuntimeError, match="corrupt pdf"):
        dispatch.load_directory(
            tmp_path, converter=ExplodingConverter(), on_error="raise"
        )


def test_unreadable_text_file_is_skipped_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("fine", encoding="utf-8")
    # Invalid UTF-8 makes read_text raise, standing in for any unreadable file.
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00broken")

    documents = dispatch.load_directory(tmp_path)

    assert [document.document_id for document in documents] == ["good.txt"]


def test_summary_reports_how_many_files_were_ingested(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")

    with caplog.at_level("INFO", logger="evidence_rag.loaders"):
        dispatch.load_directory(tmp_path)

    assert "Ingested 2 file(s)" in caplog.text
