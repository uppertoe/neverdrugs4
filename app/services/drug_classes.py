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

_NON_DEPOLARISING_BLOCKERS = (
    "rocuronium",
    "vecuronium",
    "pancuronium",
    "atracurium",
    "cisatracurium",
    "mivacurium",
    "pipecuronium",
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
    "nmdb",
    "paralytic agent",
    "paralytic agents",
    "paralytic drug",
    "paralytic drugs",
    "muscle relaxant",
    "muscle relaxants",
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
        _NON_DEPOLARISING_BLOCKERS,
        key="non-depolarising-neuromuscular-blockers",
        label="non-depolarising neuromuscular blockers",
        classes=("non-depolarising neuromuscular blocker",),
    )

    register(
        _GENERAL_NEUROMUSCULAR,
        key="neuromuscular-blockers",
        label="neuromuscular blocking agents",
        classes=("neuromuscular blocking agent",),
        roles=("generic-class",),
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
