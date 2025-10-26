from __future__ import annotations

from typing import Iterable, Sequence

ANESTHESIA_MESH_TERMS: tuple[str, ...] = (
    "Anesthesia",
    "Anesthesia, General",
    "Anesthesia, Conduction",
    "Anesthetics",
    "Anesthesiology",
    "Analgesia",
    "Analgesia, Epidural",
    "Conscious Sedation",
    "Monitoring, Intraoperative",
    "Pain, Postoperative",
)

ANESTHESIA_TEXT_TERMS: tuple[str, ...] = (
    "anesthesia",
    "anaesthesia",
    "anesthetic",
    "anesthetics",
    "sedation",
    "conscious sedation",
    "regional anesthesia",
    "nerve block",
    "perioperative",
    "induction of anesthesia",
    "general anesthesia",
    "postoperative pain",
)

DRUG_MESH_TERMS: tuple[str, ...] = (
    "Anesthetics, Intravenous",
    "Anesthetics, Local",
    "Anesthetics, Inhalation",
    "Analgesics, Opioid",
    "Analgesics, Non-Narcotic",
    "Neuromuscular Blocking Agents",
    "Hypnotics and Sedatives",
)

DRUG_TEXT_TERMS: tuple[str, ...] = (
    "propofol",
    "ketamine",
    "midazolam",
    "fentanyl",
    "remifentanil",
    "dexmedetomidine",
    "sevoflurane",
    "isoflurane",
    "desflurane",
    "rocuronium",
    "succinylcholine",
    "bupivacaine",
    "lidocaine",
)


def build_nih_search_query(
    condition_mesh_terms: Sequence[str],
    *,
    additional_text_terms: Iterable[str] | None = None,
) -> str:
    """Build a PubMed-compatible query targeting anesthetic drugs for a condition."""
    normalized_mesh_terms = [term for term in condition_mesh_terms if term]
    if not normalized_mesh_terms:
        raise ValueError("At least one condition MeSH term is required to build a query")

    condition_clause = _build_or_clause(normalized_mesh_terms, field="MeSH Terms")

    anesthesia_mesh_clause = _build_or_clause(ANESTHESIA_MESH_TERMS, field="MeSH Terms")
    anesthesia_text_clause = _build_or_clause(ANESTHESIA_TEXT_TERMS, field="Title/Abstract")

    drug_mesh_clause = _build_or_clause(DRUG_MESH_TERMS, field="MeSH Terms")
    drug_text_clause = _build_or_clause(DRUG_TEXT_TERMS, field="Title/Abstract")

    extra_text_clause = None
    if additional_text_terms:
        extra_text_clause = _build_or_clause(list(additional_text_terms), field="Title/Abstract")

    anesthesia_clause = _wrap_or(anesthesia_mesh_clause, anesthesia_text_clause)
    drug_clause = _wrap_or(drug_mesh_clause, drug_text_clause)

    support_clauses = [anesthesia_clause, drug_clause]
    if extra_text_clause:
        support_clauses.append(extra_text_clause)

    support_clause = _wrap_or(*support_clauses)

    return f"{condition_clause} AND {support_clause}"


def _build_or_clause(terms: Sequence[str], *, field: str) -> str:
    quoted_terms = [f'"{term}"[{field}]' for term in terms]
    if len(quoted_terms) == 1:
        return quoted_terms[0]
    return f"({ ' OR '.join(quoted_terms) })"


def _wrap_or(*clauses: str) -> str:
    filtered = [clause for clause in clauses if clause]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    return f"({ ' OR '.join(filtered) })"
