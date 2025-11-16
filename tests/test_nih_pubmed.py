from __future__ import annotations

from app.services.nih_pubmed import MeshBuilderTermExpander
from app.services.search import MeshBuildResult


class _RecordingBuilder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, term: str) -> MeshBuildResult:
        self.calls.append(term)
        return MeshBuildResult(
            mesh_terms=[
                "Primary Condition",
                "Primary Alias",
                "Condition Variant",
            ],
            query_payload={
                "esummary": {
                    "mesh_terms": [
                        "Primary Condition",
                        "Primary Alias",
                        "Condition Variant",
                    ]
                }
            },
            source="test",
        )


def test_mesh_builder_term_expander_caches_aliases_and_variants() -> None:
    builder = _RecordingBuilder()
    expander = MeshBuilderTermExpander(mesh_builder=builder)

    first = expander("Primary Condition")
    assert builder.calls == ["primary condition"]
    assert first.mesh_terms == ("Primary Condition",)

    second = expander("Primary Alias")
    third = expander("Condition Variant")

    assert second.mesh_terms == first.mesh_terms
    assert third.mesh_terms == first.mesh_terms
    assert builder.calls == ["primary condition"]