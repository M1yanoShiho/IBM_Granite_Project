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
