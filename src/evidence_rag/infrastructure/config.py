import math
import tomllib
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

NonEmpty = Annotated[str, Field(min_length=1)]
PositiveInteger = Annotated[int, Field(gt=0)]


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def _validate_json_value(value: object) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if math.isfinite(value):
            return
        raise ValueError("module parameters must be JSON-compatible")
    if type(value) is list:
        for item in value:
            _validate_json_value(item)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("module parameters must be JSON-compatible")
            _validate_json_value(item)
        return
    raise ValueError("module parameters must be JSON-compatible")


class ModuleConfig(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    name: NonEmpty
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters", mode="before")
    @classmethod
    def parameters_must_be_json_compatible(cls, value: object) -> object:
        _validate_json_value(value)
        return value


class ExperimentConfig(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    dataset_manifest_path: Path
    output_directory: Path
    retriever: ModuleConfig
    selector: ModuleConfig
    generator: ModuleConfig
    top_k: PositiveInteger
    max_selected: PositiveInteger
    seed: int


class _DatasetToml(FrozenModel):
    manifest: NonEmpty


class _OutputToml(FrozenModel):
    directory: NonEmpty


class _RunToml(FrozenModel):
    top_k: PositiveInteger
    max_selected: PositiveInteger
    seed: int


class _TomlExperimentConfig(FrozenModel):
    dataset: _DatasetToml
    output: _OutputToml
    retriever: ModuleConfig
    selector: ModuleConfig
    generator: ModuleConfig
    run: _RunToml


class IngestionConfig(FrozenModel):
    """Config-file switches for directory ingestion (see ``loaders.load_directory``).

    ``caption_pdf_pictures`` turns on Granite Vision captioning of embedded PDF
    figures; with ``image_ocr`` (on by default) their in-figure text is OCR'd and
    appended too. The optional string fields fall back to ``load_directory``'s own
    defaults / environment variables when left unset.
    """

    schema_version: Literal["1.0"] = "1.0"
    pdf_mode: Literal["chunks", "pages"] = "chunks"
    caption_pdf_pictures: bool = False
    image_ocr: bool = True
    caption_prompt: NonEmpty | None = None
    vision_model_id: NonEmpty | None = None
    vision_device: NonEmpty | None = None
    on_error: Literal["skip", "raise"] = "skip"
    cache_dir: Path | None = None


class _TomlIngestionSection(FrozenModel):
    schema_version: Literal["1.0"] = "1.0"
    pdf_mode: Literal["chunks", "pages"] = "chunks"
    caption_pdf_pictures: bool = False
    image_ocr: bool = True
    caption_prompt: NonEmpty | None = None
    vision_model_id: NonEmpty | None = None
    vision_device: NonEmpty | None = None
    on_error: Literal["skip", "raise"] = "skip"
    cache_dir: NonEmpty | None = None


class _TomlIngestionConfig(FrozenModel):
    ingestion: _TomlIngestionSection = Field(default_factory=_TomlIngestionSection)


def load_ingestion_config(path: Path) -> IngestionConfig:
    """Load an ``[ingestion]`` TOML section into an :class:`IngestionConfig`.

    A missing ``[ingestion]`` table yields all defaults. ``cache_dir`` is resolved
    relative to the config file, matching ``load_experiment_config``.
    """
    config_path = Path(path).resolve()
    try:
        raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"unable to load ingestion config {config_path}: {error}") from error

    section = _TomlIngestionConfig.model_validate(raw_config).ingestion
    cache_dir = (
        (config_path.parent / section.cache_dir).resolve()
        if section.cache_dir is not None
        else None
    )
    return IngestionConfig(
        pdf_mode=section.pdf_mode,
        caption_pdf_pictures=section.caption_pdf_pictures,
        image_ocr=section.image_ocr,
        caption_prompt=section.caption_prompt,
        vision_model_id=section.vision_model_id,
        vision_device=section.vision_device,
        on_error=section.on_error,
        cache_dir=cache_dir,
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    try:
        raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"unable to load experiment config {config_path}: {error}") from error

    parsed = _TomlExperimentConfig.model_validate(raw_config)
    root = config_path.parent
    return ExperimentConfig(
        dataset_manifest_path=(root / parsed.dataset.manifest).resolve(),
        output_directory=(root / parsed.output.directory).resolve(),
        retriever=parsed.retriever,
        selector=parsed.selector,
        generator=parsed.generator,
        top_k=parsed.run.top_k,
        max_selected=parsed.run.max_selected,
        seed=parsed.run.seed,
    )
