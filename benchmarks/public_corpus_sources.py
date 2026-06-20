import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLIC_SOURCES_PATH = REPO_ROOT / "benchmarks" / "corpora" / "public_sources.json"
VALID_DOMAINS = {"legal", "financial"}
REQUIRED_SOURCE_KEYS = {
    "source_id",
    "source_name",
    "domain",
    "license_or_usage_note",
    "source_url",
    "provider_name",
    "expected_document_format",
    "raw_files_may_be_committed",
    "attribution_requirement",
    "limitations",
    "recommended_local_storage_path",
    "default_sample_size",
}


class PublicSourceRegistryError(ValueError):
    """Raised when public corpus source registry metadata is invalid."""


@dataclass(frozen=True)
class PublicCorpusSource:
    source_id: str
    source_name: str
    domain: str
    license_or_usage_note: str
    source_url: str
    provider_name: str
    expected_document_format: list[str]
    raw_files_may_be_committed: bool
    attribution_requirement: str
    limitations: list[str]
    recommended_local_storage_path: str
    default_sample_size: int

    def to_manifest_source_metadata(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "provider_name": self.provider_name,
            "source_url": self.source_url,
            "license_or_usage_note": self.license_or_usage_note,
            "attribution_requirement": self.attribution_requirement,
        }


@dataclass(frozen=True)
class PublicSourceRegistry:
    schema_version: str
    sources: list[PublicCorpusSource]

    def by_id(self, source_id: str) -> PublicCorpusSource:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise PublicSourceRegistryError(f"Unknown public source_id: {source_id}")

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_count": len(self.sources),
            "sources": [
                {
                    "source_id": source.source_id,
                    "source_name": source.source_name,
                    "domain": source.domain,
                    "recommended_local_storage_path": source.recommended_local_storage_path,
                    "raw_files_may_be_committed": source.raw_files_may_be_committed,
                }
                for source in self.sources
            ],
        }


def load_public_source_registry(path: Path = DEFAULT_PUBLIC_SOURCES_PATH) -> PublicSourceRegistry:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_public_source_registry(payload)


def validate_public_source_registry(payload: dict[str, Any]) -> PublicSourceRegistry:
    if not isinstance(payload, dict):
        raise PublicSourceRegistryError("Registry payload must be an object")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise PublicSourceRegistryError("Registry schema_version must be a non-empty string")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise PublicSourceRegistryError("Registry sources must be a non-empty list")

    sources = [_parse_source(raw, index) for index, raw in enumerate(raw_sources)]
    ids = [source.source_id for source in sources]
    duplicates = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    if duplicates:
        raise PublicSourceRegistryError(f"Duplicate public source IDs: {duplicates}")
    return PublicSourceRegistry(schema_version=schema_version.strip(), sources=sources)


def _parse_source(raw: dict[str, Any], index: int) -> PublicCorpusSource:
    context = f"sources[{index}]"
    if not isinstance(raw, dict):
        raise PublicSourceRegistryError(f"{context} must be an object")
    missing = sorted(REQUIRED_SOURCE_KEYS - set(raw))
    if missing:
        raise PublicSourceRegistryError(f"{context} missing required keys: {missing}")

    domain = _required_string(raw, "domain", context)
    if domain not in VALID_DOMAINS:
        raise PublicSourceRegistryError(f"{context}.domain must be one of {sorted(VALID_DOMAINS)}")

    expected_formats = _string_list(raw["expected_document_format"], f"{context}.expected_document_format")
    limitations = _string_list(raw["limitations"], f"{context}.limitations")
    raw_files_may_be_committed = raw["raw_files_may_be_committed"]
    if not isinstance(raw_files_may_be_committed, bool):
        raise PublicSourceRegistryError(f"{context}.raw_files_may_be_committed must be a boolean")
    default_sample_size = raw["default_sample_size"]
    if not isinstance(default_sample_size, int) or default_sample_size <= 0:
        raise PublicSourceRegistryError(f"{context}.default_sample_size must be a positive integer")

    return PublicCorpusSource(
        source_id=_required_string(raw, "source_id", context),
        source_name=_required_string(raw, "source_name", context),
        domain=domain,
        license_or_usage_note=_required_string(raw, "license_or_usage_note", context),
        source_url=_required_string(raw, "source_url", context),
        provider_name=_required_string(raw, "provider_name", context),
        expected_document_format=expected_formats,
        raw_files_may_be_committed=raw_files_may_be_committed,
        attribution_requirement=_required_string(raw, "attribution_requirement", context),
        limitations=limitations,
        recommended_local_storage_path=_required_string(raw, "recommended_local_storage_path", context),
        default_sample_size=default_sample_size,
    )


def _required_string(raw: dict[str, Any], key: str, context: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PublicSourceRegistryError(f"{context}.{key} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise PublicSourceRegistryError(f"{context} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise PublicSourceRegistryError(f"{context} must contain only non-empty strings")
    return [item.strip() for item in value]
