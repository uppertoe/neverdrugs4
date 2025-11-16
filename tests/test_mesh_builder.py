from __future__ import annotations

from pathlib import Path

import pytest

from app.services.mesh_builder import NIHMeshBuilder

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _SequentialHttpClient:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, str]) -> _FakeResponse:
        if not self._responses:
            raise RuntimeError("No more responses queued")
        self.calls.append((url, params))
        return self._responses.pop(0)


def _build_docsum_chunk(mesh_id: str, terms: list[str]) -> str:
    term_items = "".join(
        f"        <Item Name=\"string\" Type=\"String\">{term}</Item>\n" for term in terms
    )
    return (
        "    <DocSum>\n"
        f"        <Id>{mesh_id}</Id>\n"
        "        <Item Name=\"DS_MeshTerms\" Type=\"List\">\n"
        f"{term_items}"
        "        </Item>\n"
        "    </DocSum>\n"
    )


def _build_esummary_response(entries: dict[str, list[str]]) -> str:
    docs = "".join(_build_docsum_chunk(mesh_id, terms) for mesh_id, terms in entries.items())
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<eSummaryResult>\n"
        f"{docs}"
        "</eSummaryResult>\n"
    )


def _build_esearch(ids: list[str]) -> str:
    id_items = "".join(f"        <Id>{mesh_id}</Id>\n" for mesh_id in ids)
    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<eSearchResult>\n"
        f"    <Count>{len(ids)}</Count>\n"
        "    <IdList>\n"
        f"{id_items}"
        "    </IdList>\n"
        "</eSearchResult>\n"
    )


def test_mesh_builder_returns_ranked_terms() -> None:
    esearch_xml = (FIXTURES / "esearch_duchenne.xml").read_text(encoding="utf-8")
    esummary_xml = (FIXTURES / "esummary_68020388.xml").read_text(encoding="utf-8")
    extra_docsum = _build_docsum_chunk(
        "67030635",
        [
            "Brachial Plexus Neuropathies",
            "Erb-Duchenne Paralysis",
            "Brachial Plexus Palsy",
        ],
    )
    combined_esummary = esummary_xml.replace("</eSummaryResult>\n", f"{extra_docsum}</eSummaryResult>\n")
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(combined_esummary),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=4)

    result = builder("duchenne muscular dystrophy")

    assert result.source == "nih-esearch+esummary"
    assert result.mesh_terms == ["Muscular Dystrophy, Duchenne"]
    selected_terms = result.query_payload.get("selected_mesh_terms")
    assert selected_terms[:4] == [
        "Muscular Dystrophy, Duchenne",
        "Duchenne Muscular Dystrophy",
        "Duchenne-Type Progressive Muscular Dystrophy",
        "Progressive Muscular Dystrophy, Duchenne Type",
    ]
    assert result.query_payload.get("canonical_mesh_term") == "Muscular Dystrophy, Duchenne"
    assert result.query_payload["esearch"]["primary_id"] == "68020388"
    assert result.query_payload["esearch"]["resolved_mesh_terms"] == ["Muscular Dystrophy, Duchenne"]
    assert len(result.query_payload["esearch"]["candidates"]) == 2
    assert result.query_payload["ranked_mesh_terms"][0]["term"] == "Muscular Dystrophy, Duchenne"
    assert http_client.calls[0][1]["term"] == "duchenne muscular dystrophy"
    assert http_client.calls[1][1]["id"] == "68020388,67030635"


def test_mesh_builder_filters_disallowed_extra_tokens() -> None:
    esearch_xml = """<?xml version='1.0' encoding='UTF-8'?>
<eSearchResult>
    <Count>3</Count>
    <IdList>
        <Id>12345</Id>
    </IdList>
</eSearchResult>
"""
    esummary_xml = """<?xml version='1.0' encoding='UTF-8'?>
<eSummaryResult>
    <DocSum>
        <Id>12345</Id>
        <Item Name="DS_MeshTerms" Type="List">
            <Item Name="string" Type="String">Rare Muscle Disease</Item>
            <Item Name="string" Type="String">Rare Muscle Disease Veterinary</Item>
            <Item Name="string" Type="String">Rare Muscle Disease Mice</Item>
            <Item Name="string" Type="String">Rare Muscle Disease Virology</Item>
            <Item Name="string" Type="String">Rare Muscle Disease Adult</Item>
        </Item>
    </DocSum>
</eSummaryResult>
"""
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=5)

    result = builder("rare muscle disease")

    assert result.mesh_terms == ["Rare Muscle Disease"]
    selected_terms = result.query_payload.get("selected_mesh_terms")
    assert selected_terms[0] == "Rare Muscle Disease"
    assert "Rare Muscle Disease Adult" in selected_terms
    for banned in ("veterinary", "mice", "virology"):
        assert all(banned not in term.lower() for term in result.mesh_terms)


def test_mesh_builder_prefers_close_matches() -> None:
    esearch_xml = (FIXTURES / "esearch_duchenne.xml").read_text(encoding="utf-8")
    esummary_xml = (FIXTURES / "esummary_68020388.xml").read_text(encoding="utf-8")
    extra_docsum = _build_docsum_chunk(
        "67030635",
        [
            "Brachial Plexus Neuropathies",
            "Erb-Duchenne Paralysis",
            "Brachial Plexus Palsy",
        ],
    )
    combined_esummary = esummary_xml.replace("</eSummaryResult>\n", f"{extra_docsum}</eSummaryResult>\n")
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(combined_esummary),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=6)

    result = builder("duchenne muscular dystrophy")

    assert result.mesh_terms == ["Muscular Dystrophy, Duchenne"]
    selected_terms = result.query_payload.get("selected_mesh_terms")
    assert selected_terms[:4] == [
        "Muscular Dystrophy, Duchenne",
        "Duchenne Muscular Dystrophy",
        "Duchenne-Type Progressive Muscular Dystrophy",
        "Progressive Muscular Dystrophy, Duchenne Type",
    ]
    ranked_terms = result.query_payload["ranked_mesh_terms"]
    becker_index = next(
        idx for idx, entry in enumerate(ranked_terms) if entry["term"] == "Duchenne-Becker Muscular Dystrophy"
    )
    assert becker_index > 3
    top_entry = ranked_terms[0]
    assert top_entry["term"] == "Muscular Dystrophy, Duchenne"
    assert top_entry["score"] >= 0.8


def test_mesh_builder_handles_docsum_v2_schema() -> None:
    esearch_xml = """<?xml version='1.0' encoding='UTF-8'?>
<eSearchResult>
    <Count>5</Count>
    <IdList>
        <Id>68046350</Id>
    </IdList>
</eSearchResult>
"""
    esummary_v2 = """<?xml version='1.0' encoding='UTF-8'?>
<eSummaryResult>
    <DocumentSummarySet>
        <DocumentSummary uid="68046350">
            <DS_MeshTerms>
                <string>Porphyria, Variegate</string>
                <string>Variegate Porphyria</string>
                <string>Porphyria Variegata</string>
            </DS_MeshTerms>
        </DocumentSummary>
    </DocumentSummarySet>
</eSummaryResult>
"""
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_v2),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=3)

    result = builder("porphyria")

    assert result.mesh_terms == ["Porphyria, Variegate"]
    selected_terms = result.query_payload.get("selected_mesh_terms")
    assert "Variegate Porphyria" in selected_terms
    assert http_client.calls[1][1]["version"] == "2.0"


def test_mesh_builder_skips_non_matching_descriptors() -> None:
    esearch_xml = _build_esearch(["9001", "9002"])
    esummary_xml = _build_esummary_response(
        {
            "9001": [
                "Example Condition Primary",
                "Primary Example Disorder",
            ],
            "9002": [
                "Unrelated Neuropathy",
                "Peripheral Nerve Injury",
            ],
        }
    )
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=3)

    result = builder("example condition")

    assert result.mesh_terms == ["Example Condition Primary"]
    resolved = result.query_payload["esearch"]["resolved_mesh_terms"]
    assert resolved == ["Example Condition Primary"]
    candidates = result.query_payload["esearch"]["candidates"]
    assert len(candidates) == 2
    assert candidates[1]["canonical_overlap"] == 0


def test_mesh_builder_returns_multiple_terms_when_ambiguous() -> None:
    esearch_xml = _build_esearch(["9101", "9102"])
    esummary_xml = _build_esummary_response(
        {
            "9101": [
                "Example Condition Alpha",
                "Alpha Example Condition",
            ],
            "9102": [
                "Example Condition Beta",
                "Beta Example Condition",
            ],
        }
    )
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=5)

    result = builder("example condition")

    assert len(result.mesh_terms) == 2
    assert set(result.mesh_terms) == {
        "Example Condition Alpha",
        "Example Condition Beta",
    }
    selected_terms = result.query_payload["selected_mesh_terms"]
    assert len(selected_terms) == 2
    assert set(selected_terms) == {
        "Example Condition Alpha",
        "Example Condition Beta",
    }
    resolved = result.query_payload["esearch"]["resolved_mesh_terms"]
    assert len(resolved) == 2
    assert set(resolved) == {
        "Example Condition Alpha",
        "Example Condition Beta",
    }
    query = result.query_payload["esearch"]["query"]
    assert query is not None
    assert '"Example Condition Alpha"' in query
    assert '"Example Condition Beta"' in query


def test_mesh_builder_includes_alias_terms_in_query() -> None:
    esearch_xml = """<?xml version='1.0' encoding='UTF-8'?>
<eSearchResult>
    <Count>1</Count>
    <IdList>
        <Id>111111</Id>
    </IdList>
</eSearchResult>
"""
    esummary_xml = """<?xml version='1.0' encoding='UTF-8'?>
<eSummaryResult>
    <DocSum>
        <Id>111111</Id>
        <Item Name=\"DS_MeshTerms\" Type=\"List\">
            <Item Name=\"string\" Type=\"String\">Example Condition Syndrome</Item>
            <Item Name=\"string\" Type=\"String\">Example Condition Variant</Item>
            <Item Name=\"string\" Type=\"String\">Sample Disorder Alpha</Item>
            <Item Name=\"string\" Type=\"String\">Alpha Neuro Disorder</Item>
            <Item Name=\"string\" Type=\"String\">Example Condition Type 1</Item>
            <Item Name=\"string\" Type=\"String\">Example Condition Type 2</Item>
        </Item>
    </DocSum>
</eSummaryResult>
"""
    http_client = _SequentialHttpClient([
        _FakeResponse(esearch_xml),
        _FakeResponse(esummary_xml),
    ])
    builder = NIHMeshBuilder(http_client=http_client, max_terms=5)

    result = builder("example condition")

    assert result.mesh_terms == ["Example Condition Syndrome"]
    selected_terms = result.query_payload.get("selected_mesh_terms")
    assert selected_terms[:4] == [
        "Example Condition Syndrome",
        "Example Condition Variant",
        "Sample Disorder Alpha",
        "Alpha Neuro Disorder",
    ]
    query = result.query_payload["esearch"]["query"]
    assert query is not None
    assert '"Sample Disorder Alpha"[mesh]' in query
    assert '"Alpha Neuro Disorder"[mesh]' in query
    assert '"Sample Disorder Alpha"[tiab]' in query
    assert '"Alpha Neuro Disorder"[tiab]' in query


def test_mesh_builder_handles_missing_ids() -> None:
    empty_esearch = """<?xml version='1.0' encoding='UTF-8'?><eSearchResult><Count>0</Count><IdList></IdList></eSearchResult>"""
    http_client = _SequentialHttpClient([
        _FakeResponse(empty_esearch),
    ])
    builder = NIHMeshBuilder(http_client=http_client)

    result = builder("condition")

    assert result.mesh_terms == []
    assert result.query_payload["esearch"]["ids"] == []


def test_mesh_builder_propagates_http_error() -> None:
    http_client = _SequentialHttpClient([
        _FakeResponse("fail", status_code=500),
    ])
    builder = NIHMeshBuilder(http_client=http_client)

    with pytest.raises(RuntimeError):
        builder("condition")
