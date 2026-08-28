from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from anchor import Anchor
from tests.unit_harness import FakeMemoryStore, deterministic_embedding

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "ingestion"
_LOST_METADATA_REASON = (
    "Issue #77: Anchor.ingest_file currently drops extracted metadata before "
    "MemoryStore.add"
)


_QUESTIONS = "What is Anchor?\nWhat does Anchor store?"


def _questions(_messages: list[dict]) -> str:
    return _QUESTIONS


def _run_ingest(filename: str) -> tuple[list[str], str, FakeMemoryStore]:
    filepath = str((_FIXTURES / filename).resolve())
    store = FakeMemoryStore(allow_fallback_query=True)
    anchor = Anchor(
        ai_fn=_questions,
        light_ai_fn=_questions,
        memory_store=store,
        embed_fn=deterministic_embedding,
    )
    chunk_ids = anchor.ingest_file(filepath)
    return chunk_ids, filepath, store


def _assert_common_chunk(chunk: dict, filepath: str) -> dict:
    content = chunk.get("content")
    assert isinstance(content, str)
    assert content

    metadata = chunk.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("source") == filepath
    assert metadata.get("questions") == _QUESTIONS

    timestamp_value = metadata.get("timestamp")
    assert isinstance(timestamp_value, str)
    timestamp = datetime.fromisoformat(timestamp_value)
    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() is not None
    return metadata


@pytest.mark.unit
@pytest.mark.parametrize("filename", ["sample.txt", "sample.md"])
def test_ingest_file_stores_retrievable_content_with_provenance(filename: str) -> None:
    chunk_ids, filepath, store = _run_ingest(filename)

    assert chunk_ids
    for chunk_id in chunk_ids:
        chunk = store.get(chunk_id)
        assert isinstance(chunk, dict)
        _assert_common_chunk(chunk, filepath)

        results = store.query(
            deterministic_embedding(chunk.get("content", "")), top_k=len(chunk_ids)
        )
        assert any(result["id"] == chunk_id for result in results)


@pytest.mark.unit
@pytest.mark.xfail(reason=_LOST_METADATA_REASON, strict=True, raises=KeyError)
@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [("sample.txt", "text"), ("sample.md", "markdown")],
)
def test_ingest_file_preserves_extracted_source_format(
    filename: str, expected_format: str
) -> None:
    chunk_ids, filepath, store = _run_ingest(filename)

    assert chunk_ids
    for chunk_id in chunk_ids:
        chunk = store.get(chunk_id)
        assert isinstance(chunk, dict)
        metadata = _assert_common_chunk(chunk, filepath)
        assert metadata["source_format"] == expected_format


@pytest.mark.unit
@pytest.mark.xfail(reason=_LOST_METADATA_REASON, strict=True, raises=KeyError)
def test_ingest_text_file_preserves_fixture_offsets() -> None:
    chunk_ids, filepath, store = _run_ingest("sample.txt")
    fixture_content = Path(filepath).read_text(encoding="utf-8")

    assert len(chunk_ids) == 1
    chunk = store.get(chunk_ids[0])
    assert isinstance(chunk, dict)
    assert chunk.get("content") == fixture_content
    metadata = _assert_common_chunk(chunk, filepath)
    expected_offsets = {
        "line_start": 1,
        "line_end": len(fixture_content.splitlines()),
        "char_start": 0,
        "char_end": len(fixture_content),
    }
    for key, expected_value in expected_offsets.items():
        assert metadata[key] == expected_value


@pytest.mark.unit
@pytest.mark.xfail(reason=_LOST_METADATA_REASON, strict=True, raises=KeyError)
def test_ingest_markdown_file_preserves_heading_on_heading_section() -> None:
    chunk_ids, filepath, store = _run_ingest("sample.md")
    assert chunk_ids
    chunks = []
    for chunk_id in chunk_ids:
        chunk = store.get(chunk_id)
        assert isinstance(chunk, dict)
        _assert_common_chunk(chunk, filepath)
        chunks.append(chunk)

    heading_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("content", "").startswith("# About Anchor")
    ]
    assert len(heading_chunks) == 1
    heading_metadata = heading_chunks[0].get("metadata")
    assert isinstance(heading_metadata, dict)

    assert heading_metadata["heading"] == "# About Anchor"
