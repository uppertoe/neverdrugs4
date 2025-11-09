from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class ConditionTermExpansion:
    mesh_terms: tuple[str, ...]
    alias_terms: tuple[str, ...] = ()


ConditionTermExpander = Callable[[str], ConditionTermExpansion | None]

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
    "Cholinesterase Inhibitors",
    "Cyclodextrins",
    "Parasympatholytics",
    "Adrenergic Agonists",
    "Adrenergic beta-Antagonists",
    "Vasodilator Agents",
    "Cephalosporins",
    "Cyclooxygenase 2 Inhibitors",
    "Antiemetics",
    "Serotonin Antagonists",
    "Adrenal Cortex Hormones",
    "Glucagon",
    "Calcium Compounds",
    "Ringer's Lactate",
    "Anticonvulsants",
    "Anti-Arrhythmia Agents",
    "Nitrates",
    "Adenosine",
    "Muscle Relaxants, Central",
    "Sodium Chloride",
)

# Expand class and agent coverage to mirror the stocked perioperative drug trolley.
DRUG_TEXT_TERMS: tuple[str, ...] = (
    # Analgesics and sedatives
    "analgesic opioid",
    "opioid",
    "opioids",
    "morphine",
    "fentanyl",
    "remifentanil",
    "parecoxib",
    "benzodiazepine",
    "benzodiazepines",
    "midazolam",
    "versed",
    "diazepam",
    "lorazepam",
    "propofol",
    "diprivan",
    "ketamine",
    "ketalar",
    # Volatile and inhalational agents
    "volatile anaesthetic",
    "volatile anaesthetics",
    "volatile anesthetic",
    "volatile anesthetics",
    "volatile agent",
    "volatile agents",
    "inhalational anaesthetic",
    "inhalational anesthetic",
    "desflurane",
    "isoflurane",
    "sevoflurane",
    "halothane",
    "nitrous oxide",
    "nitrous",
    "laughing gas",
    "n2o",
    # Neuromuscular blockade and reversal
    "neuromuscular blockade",
    "neuromuscular blocker",
    "neuromuscular blockers",
    "neuromuscular blocking agent",
    "neuromuscular blocking agents",
    "muscle relaxant",
    "muscle relaxants",
    "succinylcholine",
    "rocuronium",
    "atracurium",
    "sugammadex",
    "bridion",
    "neostigmine",
    "prostigmin",
    "atropine",
    "glycopyrrolate",
    # Crisis and rescue agents
    "dantrolene",
    "adrenaline",
    "epinephrine",
    "noradrenaline",
    "norepinephrine",
    "glucagon",
    "calcium gluconate",
    # Perioperative adjuncts
    "lidocaine",
    "lignocaine",
    "xylocaine",
    "bupivacaine",
    "marcaine",
    "levobupivacaine",
    "ropivacaine",
    "naropin",
    "mepivacaine",
    "prilocaine",
    "cefazolin",
    "dexamethasone",
    "ondansetron",
    "metoclopramide",
    "hyoscine",
    "hartmann's solution",
    "hartmanns solution",
    "sodium chloride",
    # Hemodynamic agents
    "hydralazine",
    "gtn",
    "glyceryl trinitrate",
    "sodium nitroprusside",
    "amiodarone",
    "metoprolol",
    "adenosine",
    # Neurologic and anticonvulsant
    "levetiracetam",
)

# Focus query narrowing on class-level language while leaving the richer list above
# available for downstream snippet extraction.
DRUG_QUERY_TEXT_TERMS: tuple[str, ...] = (
    "analgesic opioid",
    "opioid",
    "opioids",
    "benzodiazepine",
    "benzodiazepines",
    "volatile anaesthetic",
    "volatile anesthetic",
    "volatile anaesthetics",
    "volatile anesthetics",
    "volatile agent",
    "volatile agents",
    "inhalational anaesthetic",
    "inhalational anesthetic",
    "nitrous oxide",
    "neuromuscular blockade",
    "neuromuscular blocker",
    "neuromuscular blockers",
    "neuromuscular blocking agent",
    "neuromuscular blocking agents",
    "muscle relaxant",
    "muscle relaxants",
    "neuromuscular reversal agent",
    "hartmann's solution",
    "hartmanns solution",
    "sodium chloride",
)

CLINICAL_CONTEXT_MESH_TERMS: tuple[str, ...] = (
    "Critical Care",
    "Critical Care Outcomes",
    "Emergency Medicine",
    "Emergency Treatment",
    "Intensive Care Units",
    "Perioperative Care",
    "Perioperative Period",
    "Resuscitation",
)

CLINICAL_CONTEXT_TEXT_TERMS: tuple[str, ...] = (
    "critical care",
    "intensive care",
    "icu",
    "anaesthetic emergency",
    "emergency department",
    "emergency medicine",
    "perioperative",
    "rapid sequence induction",
    "resuscitation",
)

FOCUSED_PUBLICATION_TYPES: tuple[str, ...] = (
    "Review",
    "Systematic Review",
    "Meta-Analysis",
    "Case Reports",
    "Clinical Study",
    "Observational Study",
)

HUMAN_SUBJECTS_TERMS: tuple[str, ...] = (
    "Humans",
)


def build_nih_search_query(
    condition_mesh_terms: Sequence[str],
    *,
    additional_text_terms: Iterable[str] | None = None,
    term_expander: ConditionTermExpander | None = None,
) -> str:
    """Build a PubMed-compatible query targeting anesthetic drugs for a condition."""
    mesh_terms: list[str] = []
    text_terms: list[str] = []

    for raw_term in condition_mesh_terms:
        if not raw_term:
            continue
        term = raw_term.strip()
        if not term:
            continue
        expansion = term_expander(term) if term_expander else None
        if expansion:
            mesh_terms.extend(expansion.mesh_terms)
            if expansion.alias_terms:
                text_terms.extend(expansion.alias_terms)
            text_terms.append(term)
            continue
        mesh_terms.append(term)
        text_terms.append(term)

    normalized_mesh_terms = _dedupe_preserving_order(mesh_terms)
    if not normalized_mesh_terms:
        raise ValueError("At least one condition MeSH term is required to build a query")

    condition_mesh_clause = _build_or_clause(normalized_mesh_terms, field="mesh")

    condition_text_terms: list[str] = _dedupe_preserving_order(text_terms)
    if additional_text_terms:
        condition_text_terms.extend(term.strip() for term in additional_text_terms if term)
    condition_text_clause = _build_or_clause(sorted(set(filter(None, condition_text_terms))), field="tiab")

    condition_clause = _wrap_or(condition_mesh_clause, condition_text_clause)

    anesthesia_mesh_clause = _build_or_clause(ANESTHESIA_MESH_TERMS, field="mesh")
    anesthesia_text_clause = _build_or_clause(ANESTHESIA_TEXT_TERMS, field="tiab")

    drug_mesh_clause = _build_or_clause(DRUG_MESH_TERMS, field="mesh")
    drug_text_clause = _build_or_clause(DRUG_QUERY_TEXT_TERMS, field="tiab")

    context_mesh_clause = _build_or_clause(CLINICAL_CONTEXT_MESH_TERMS, field="mesh")
    context_text_clause = _build_or_clause(CLINICAL_CONTEXT_TEXT_TERMS, field="tiab")

    extra_text_clause = None
    if additional_text_terms:
        extra_text_clause = _build_or_clause(sorted(set(additional_text_terms)), field="tiab")

    anesthesia_clause = _wrap_or(anesthesia_mesh_clause, anesthesia_text_clause)
    drug_clause = _wrap_or(drug_mesh_clause, drug_text_clause)
    context_clause = _wrap_or(context_mesh_clause, context_text_clause)

    support_clauses = [anesthesia_clause, drug_clause, context_clause]
    if extra_text_clause:
        support_clauses.append(extra_text_clause)

    support_clause = _wrap_or(*support_clauses)

    human_clause = _build_or_clause(HUMAN_SUBJECTS_TERMS, field="mesh")
    publication_clause = _build_or_clause(FOCUSED_PUBLICATION_TYPES, field="pt")

    return _combine_and(condition_clause, support_clause, human_clause, publication_clause)


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


def _combine_and(*clauses: str) -> str:
    filtered = [clause for clause in clauses if clause]
    if not filtered:
        return ""
    if len(filtered) == 1:
        return filtered[0]
    return " AND ".join(filtered)


def _dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered
