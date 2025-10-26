from pathlib import Path

import pytest

from app.services.espell import ESpellSuggestion, NIHESpellClient, extract_espell_correction


FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_espell_correction_returns_suggestion() -> None:
    xml_payload = (FIXTURES / "espell_duchene.xml").read_text(encoding="utf-8")

    suggestion = extract_espell_correction(xml_payload)

    assert isinstance(suggestion, ESpellSuggestion)
    assert suggestion.query == "duchene"
    assert suggestion.corrected_query == "duchenne"
    assert suggestion.replaced == "duchenne"
    assert suggestion.suggestion == "duchenne"


@pytest.mark.parametrize(
    "xml_payload",
    [
        "<?xml version='1.0'?><eSpellResult><Database>mesh</Database><Query>ok</Query><ERROR/></eSpellResult>",
        "<?xml version='1.0'?><eSpellResult><Database>mesh</Database><Query>ok</Query></eSpellResult>",
    ],
)
def test_extract_espell_correction_handles_missing_suggestion(xml_payload: str) -> None:
    suggestion = extract_espell_correction(xml_payload)

    assert suggestion.corrected_query is None
    assert suggestion.replaced is None
    assert suggestion.suggestion is None


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class _FakeHttpxClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, params: dict[str, str]) -> _FakeResponse:
        self.calls.append((url, params))
        return self.response


def test_nih_espell_client_returns_suggestion() -> None:
    xml_payload = (FIXTURES / "espell_duchene.xml").read_text(encoding="utf-8")
    fake_client = _FakeHttpxClient(_FakeResponse(xml_payload))
    client = NIHESpellClient(http_client=fake_client)

    suggestion = client("duchene")

    assert suggestion == "duchenne"
    assert fake_client.calls == [
        (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi",
            {"db": "mesh", "term": "duchene"},
        )
    ]


def test_nih_espell_client_handles_error_status() -> None:
    fake_client = _FakeHttpxClient(_FakeResponse("oops", status_code=500))
    client = NIHESpellClient(http_client=fake_client)

    suggestion = client("anything")

    assert suggestion is None
