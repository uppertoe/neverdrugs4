from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services import query_terms
from app.services.query_terms import build_nih_search_query
from app.services.nih_pubmed import MeshBuilderTermExpander
from scripts.validate_mesh_terms import BASE_URL, fetch_mesh_descriptor_xml, load_mesh_descriptors


@dataclass
class TermRecord:
    section: str
    field: str
    term: str
    is_mesh_field: bool
    valid_mesh: bool | None
    override_applied: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "section": self.section,
            "field": self.field,
            "term": self.term,
            "is_mesh_field": self.is_mesh_field,
            "valid_mesh": self.valid_mesh,
            "override_applied": self.override_applied,
        }


def _normalize_sequence(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        normalized = value.strip()
        if not normalized:
            continue
        if (normalized.startswith('"') and normalized.endswith('"')) or (
            normalized.startswith("'") and normalized.endswith("'")
        ):
            normalized = normalized[1:-1].strip()
            if not normalized:
                continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _collect_sections(
    condition_mesh: Sequence[str],
    additional_text_terms: Sequence[str],
    term_expander: MeshBuilderTermExpander,
) -> tuple[list[tuple[str, str, Sequence[str]]], dict[str, query_terms.ConditionTermExpansion]]:
    condition_mesh_terms = _normalize_sequence(condition_mesh)
    extra_text_terms = _normalize_sequence(additional_text_terms)

    condition_text_terms = _normalize_sequence([*condition_mesh_terms, *extra_text_terms])

    expanded_mesh_terms: list[str] = []
    expanded_text_terms: list[str] = []
    expansion_map: dict[str, query_terms.ConditionTermExpansion] = {}
    for term in condition_mesh_terms:
        expansion = term_expander(term)
        if not expansion:
            continue
        expansion_map[term.lower()] = expansion
        expanded_mesh_terms.extend(expansion.mesh_terms)
        expanded_text_terms.extend(expansion.alias_terms)

    sections: list[tuple[str, str, Sequence[str]]] = [
        ("condition_mesh", "mesh", condition_mesh_terms),
        ("condition_text", "tiab", condition_text_terms),
        ("additional_text", "tiab", extra_text_terms),
        ("anesthesia_mesh", "mesh", query_terms.ANESTHESIA_MESH_TERMS),
        ("anesthesia_text", "tiab", query_terms.ANESTHESIA_TEXT_TERMS),
        ("drug_mesh", "mesh", query_terms.DRUG_MESH_TERMS),
        ("drug_text", "tiab", query_terms.DRUG_TEXT_TERMS),
        ("clinical_context_mesh", "mesh", query_terms.CLINICAL_CONTEXT_MESH_TERMS),
        ("clinical_context_text", "tiab", query_terms.CLINICAL_CONTEXT_TEXT_TERMS),
        ("publication_types", "pt", query_terms.FOCUSED_PUBLICATION_TYPES),
        ("human_subjects", "mesh", query_terms.HUMAN_SUBJECTS_TERMS),
    ]
    if expanded_mesh_terms:
        sections.insert(1, ("condition_mesh_expanded", "mesh", _normalize_sequence(expanded_mesh_terms)))
    if expanded_text_terms:
        sections.insert(2, ("condition_text_aliases", "tiab", _normalize_sequence(expanded_text_terms)))
    return sections, expansion_map


def _annotate_terms(
    sections: Sequence[tuple[str, str, Sequence[str]]],
    descriptors: set[str],
    expansion_map: dict[str, query_terms.ConditionTermExpansion],
) -> list[TermRecord]:
    annotated: list[TermRecord] = []
    for section, field, terms in sections:
        is_mesh_field = field in {"mesh", "pt"}
        for term in terms:
            key = term.lower()
            valid_mesh = (key in descriptors) if is_mesh_field else None
            override_applied = False
            if section == "condition_mesh" and key in expansion_map:
                override_applied = True
            elif section in {"condition_mesh_expanded", "condition_text_aliases"}:
                override_applied = True
            annotated.append(
                TermRecord(
                    section=section,
                    field=field,
                    term=term,
                    is_mesh_field=is_mesh_field,
                    valid_mesh=valid_mesh,
                    override_applied=override_applied,
                )
            )
    return annotated


def _print_report(records: Sequence[TermRecord], query: str) -> None:
    print("PubMed query:\n----------------")
    print(query)
    print("\nTerm validation:\n----------------")
    header = f"{'Section':24} {'Field':6} {'Term':50} {'Valid Mesh':11}"
    print(header)
    print("-" * len(header))
    for record in records:
        status = "N/A"
        if record.valid_mesh is True:
            status = "yes"
        elif record.valid_mesh is False:
            status = "NO"
        line = f"{record.section:24} {record.field:6} {record.term:50} {status:>11}"
        print(line)

    invalid_records = [rec for rec in records if rec.valid_mesh is False and not rec.override_applied]
    if invalid_records:
        print("\nInvalid MeSH terms detected:")
        for rec in invalid_records:
            print(f"  - {rec.term} (section={rec.section})")
    else:
        print("\nAll mesh-tagged terms resolved to valid descriptors.")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the generated NIH search query and validate which components correspond to valid MeSH descriptors."
        )
    )
    parser.add_argument(
        "condition_mesh",
        nargs="+",
        help="MeSH descriptors representing the condition of interest.",
    )
    parser.add_argument(
        "--additional-text",
        nargs="*",
        default=[],
        help="Additional free-text terms supplied alongside condition MeSH descriptors.",
    )
    parser.add_argument(
        "--mesh-url",
        type=str,
        help="Optional MeSH descriptor dataset URL; latest dataset auto-detected when omitted.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".mesh_cache"),
        help="Directory used to cache the MeSH dataset.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    descriptor_path = fetch_mesh_descriptor_xml(args.mesh_url, cache_dir=args.cache_dir)
    descriptors = load_mesh_descriptors(descriptor_path)

    term_expander = MeshBuilderTermExpander()

    sections, expansion_map = _collect_sections(args.condition_mesh, args.additional_text, term_expander)
    records = _annotate_terms(sections, descriptors, expansion_map)

    query = build_nih_search_query(
        args.condition_mesh,
        additional_text_terms=args.additional_text,
        term_expander=term_expander,
    )

    if args.json:
        payload = {
            "query": query,
            "terms": [record.as_dict() for record in records],
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_report(records, query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
