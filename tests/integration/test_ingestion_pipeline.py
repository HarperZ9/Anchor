from __future__ import annotations

from pathlib import Path

import pytest

from anchor import Anchor
from anchor.extractors.markdown import MarkdownExtractor
from anchor.extractors.text import TextExtractor
from tests.unit_harness import FakeMemoryStore, deterministic_embedding

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "ingestion"


def _offline_model(_messages: list[dict]) -> str:
    return "What is Anchor about?"


def _run_ingest(filename: str) -> tuple[list[str], Path, FakeMemoryStore]:
    filepath = (_FIXTURES / filename).resolve()
    store = FakeMemoryStore(allow_fallback_query=True)
    anchor = Anchor(
        ai_fn=_offline_model,
        light_ai_fn=_offline_model,
        memory_store=store,
        embed_fn=deterministic_embedding,
    )

    chunk_ids = anchor.ingest_file(filepath)
    return chunk_ids, filepath, store


def _assert_document_metadata(metadata: dict, expected: dict[str, str | int]) -> None:
    for key, value in expected.items():
        assert key in metadata, f"missing metadata field: {key}"
        assert metadata[key] == value


@pytest.mark.unit
def test_fixture_extractors_emit_the_pipeline_metadata_contract() -> None:
    text_path = (_FIXTURES / "sample.txt").resolve()
    text = TextExtractor().extract(text_path)
    assert text.source_format == "text"
    assert text.metadata["char_start"] == 0
    assert text.metadata["char_end"] == len(text.content)

    markdown_path = (_FIXTURES / "sample.md").resolve()
    markdown = MarkdownExtractor().extract(markdown_path)
    assert len(markdown) == 1
    assert markdown[0].source_format == "markdown"
    assert markdown[0].metadata["heading"] == "# About Anchor"


@pytest.mark.unit
@pytest.mark.parametrize("filename", ["sample.txt", "sample.md"])
def test_ingest_file_preserves_existing_generated_metadata(filename: str) -> None:
    chunk_ids, filepath, store = _run_ingest(filename)

    assert len(chunk_ids) == 1
    for chunk_id in chunk_ids:
        chunk = store.get(chunk_id)
        assert chunk is not None
        assert chunk["metadata"]["source"] == str(filepath)
        assert chunk["metadata"]["questions"] == "What is Anchor about?"
        assert chunk["metadata"]["timestamp"]

    results = store.query(
        deterministic_embedding(filepath.read_text(encoding="utf-8")),
        top_k=len(chunk_ids),
    )
    assert {result["id"] for result in results} == set(chunk_ids)


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="Anchor.ingest_file currently drops TextExtractor source_format before MemoryStore.add",
)
def test_plain_text_source_format_survives_ingestion() -> None:
    chunk_ids, _filepath, store = _run_ingest("sample.txt")
    assert len(chunk_ids) == 1

    chunk = store.get(chunk_ids[0])
    assert chunk is not None
    _assert_document_metadata(chunk["metadata"], {"source_format": "text"})


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="Anchor.ingest_file currently drops TextExtractor character offsets before MemoryStore.add",
)
def test_plain_text_character_offsets_survive_ingestion() -> None:
    chunk_ids, filepath, store = _run_ingest("sample.txt")
    assert len(chunk_ids) == 1

    chunk = store.get(chunk_ids[0])
    assert chunk is not None
    content = filepath.read_text(encoding="utf-8")
    _assert_document_metadata(
        chunk["metadata"], {"char_start": 0, "char_end": len(content)}
    )


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="Anchor.ingest_file currently drops MarkdownExtractor source_format before MemoryStore.add",
)
def test_markdown_source_format_survives_ingestion() -> None:
    chunk_ids, _filepath, store = _run_ingest("sample.md")
    assert len(chunk_ids) == 1

    chunk = store.get(chunk_ids[0])
    assert chunk is not None
    _assert_document_metadata(chunk["metadata"], {"source_format": "markdown"})


@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="Anchor.ingest_file currently drops MarkdownExtractor heading metadata before MemoryStore.add",
)
def test_markdown_heading_survives_ingestion() -> None:
    chunk_ids, _filepath, store = _run_ingest("sample.md")
    assert len(chunk_ids) == 1

    chunk = store.get(chunk_ids[0])
    assert chunk is not None
    _assert_document_metadata(chunk["metadata"], {"heading": "# About Anchor"})


@pytest.mark.unit
def test_document_metadata_assertion_detects_dropped_extractor_fields() -> None:
    source = (_FIXTURES / "sample.txt").resolve()
    dropped_metadata = {
        "source": str(source),
        "questions": "What is Anchor about?",
        "timestamp": "2026-09-04T00:00:00+00:00",
    }

    with pytest.raises(AssertionError, match="source_format"):
        _assert_document_metadata(dropped_metadata, {"source_format": "text"})
