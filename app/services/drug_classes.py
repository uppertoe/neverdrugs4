from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class DrugGroup:
    key: str
    label: str
    classes: Tuple[str, ...] = ()
    roles: Tuple[str, ...] = ()


_VOLATILE_ANESTHETICS = (
    "sevoflurane",
    "desflurane",
    "isoflurane",
    "enflurane",
    "halothane",
    "methoxyflurane",
    "volatile",
    "volatiles",
    "volatile agent",
    "volatile agents",
    "volatile anesthetic",
    "volatile anaesthetic",
    "volatile anesthetics",
    "volatile anaesthetics",
)

_DEPOLARISING_BLOCKERS = (
    "succinylcholine",
    "suxamethonium",
)

_AMINOSTEROID_BLOCKERS = (
    "rocuronium",
    "vecuronium",
    "pancuronium",
    "pipecuronium",
    "rapacuronium",
)

_BENZYLISOQUINOLINIUM_BLOCKERS = (
    "atracurium",
    "cisatracurium",
    "mivacurium",
    "doxacurium",
    "gallamine",
)

_GENERAL_NEUROMUSCULAR = (
    "neuromuscular",
    "neuromuscular blocker",
    "neuromuscular blockers",
    "neuromuscular blocking agent",
    "neuromuscular blocking agents",
    "neuromuscular blocking drug",
    "neuromuscular blocking drugs",
    "neuromuscular block",
    "neuromuscular blockade",
    "neuromuscular blockade (intervention)",
    "nmdb",
    "paralytic agent",
    "paralytic agents",
    "paralytic drug",
    "paralytic drugs",
    "muscle relaxant",
    "muscle relaxants",
)

_INTRAVENOUS_ANESTHETICS = (
    "propofol",
    "diprivan",
    "etomidate",
    "amidate",
    "ketamine",
    "ketalar",
    "thiopental",
    "methohexital",
)

_LOCAL_ANESTHETICS = (
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
)

_OPIOID_ANALGESICS = (
    "morphine",
    "duramorph",
    "fentanyl",
    "sublimaze",
    "remifentanil",
    "ultiva",
    "sufentanil",
    "alfentanil",
    "hydromorphone",
)

_BENZODIAZEPINE_SEDATIVES = (
    "midazolam",
    "versed",
    "diazepam",
    "lorazepam",
)

_NITROUS_ANALGESICS = (
    "nitrous oxide",
    "nitrous",
    "laughing gas",
    "n2o",
)

_REVERSAL_AGENTS = (
    "sugammadex",
    "bridion",
    "neostigmine",
    "prostigmin",
    "atropine",
    "glycopyrrolate",
)


def _build_default_groups() -> Dict[str, DrugGroup]:
    groups: Dict[str, DrugGroup] = {}

    def register(
        names: Iterable[str], *, key: str, label: str, classes: Tuple[str, ...], roles: Tuple[str, ...] = ()
    ) -> None:
        for name in names:
            normalized = name.lower()
            groups[normalized] = DrugGroup(key=key, label=label, classes=classes, roles=roles)

    register(
        _VOLATILE_ANESTHETICS,
        key="volatile-anesthetics",
        label="volatile anesthetics",
        classes=("volatile anesthetic",),
    )

    register(
        _DEPOLARISING_BLOCKERS,
        key="depolarising-neuromuscular-blockers",
        label="depolarising neuromuscular blockers",
        classes=("depolarising neuromuscular blocker",),
    )

    register(
        _AMINOSTEROID_BLOCKERS,
        key="aminosteroid-neuromuscular-blockers",
        label="aminosteroid neuromuscular blockers",
        classes=("neuromuscular blocker (non-depolarising)", "neuromuscular blocker (aminosteroid)"),
    )

    register(
        _BENZYLISOQUINOLINIUM_BLOCKERS,
        key="benzylisoquinolinium-neuromuscular-blockers",
        label="benzylisoquinolinium neuromuscular blockers",
        classes=(
            "neuromuscular blocker (non-depolarising)",
            "neuromuscular blocker (benzylisoquinolinium)",
        ),
    )

    register(
        _GENERAL_NEUROMUSCULAR,
        key="neuromuscular-blockers",
        label="neuromuscular blocking agents",
        classes=("neuromuscular blocking agent",),
        roles=("generic-class",),
    )

    register(
        _INTRAVENOUS_ANESTHETICS,
        key="intravenous-anesthetics",
        label="intravenous anesthetics",
        classes=("intravenous anesthetic",),
    )

    register(
        _LOCAL_ANESTHETICS,
        key="local-anesthetics",
        label="local anesthetics",
        classes=("local anesthetic",),
    )

    register(
        _OPIOID_ANALGESICS,
        key="opioid-analgesics",
        label="opioid analgesics",
        classes=("opioid analgesic",),
    )

    register(
        _BENZODIAZEPINE_SEDATIVES,
        key="benzodiazepine-sedatives",
        label="benzodiazepine sedatives",
        classes=("benzodiazepine sedative",),
    )

    register(
        _NITROUS_ANALGESICS,
        key="inhaled-analgesics",
        label="inhaled analgesics",
        classes=("inhaled analgesic",),
    )

    register(
        _REVERSAL_AGENTS,
        key="neuromuscular-reversal-agents",
        label="neuromuscular reversal agents",
        classes=("neuromuscular reversal agent",),
    )

    register(
        ("dantrolene",),
        key="dantrolene",
        label="dantrolene",
        classes=("ryr1 modulator",),
        roles=("mh-therapy",),
    )

    return groups


_DEFAULT_GROUPS = _build_default_groups()


def resolve_drug_group(drug_name: str) -> DrugGroup:
    normalized = drug_name.strip().lower()
    if not normalized:
        return DrugGroup(key="unknown", label="unknown", classes=())

    group = _DEFAULT_GROUPS.get(normalized)
    if group is not None:
        return group

    return DrugGroup(key=normalized, label=drug_name.strip(), classes=())


def list_known_groups() -> Iterable[DrugGroup]:
    seen = {}
    for group in _DEFAULT_GROUPS.values():
        if group.key not in seen:
            seen[group.key] = group
    return seen.values()
