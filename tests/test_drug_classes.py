from app.services.drug_classes import resolve_drug_group


def test_resolve_drug_group_aminosteroid() -> None:
    group = resolve_drug_group("Rocuronium")

    assert group.key == "aminosteroid-neuromuscular-blockers"
    assert group.label == "aminosteroid neuromuscular blockers"
    assert group.classes == (
        "neuromuscular blocker (non-depolarising)",
        "neuromuscular blocker (aminosteroid)",
    )
    assert "generic-class" not in group.roles


def test_resolve_drug_group_benzylisoquinolinium() -> None:
    group = resolve_drug_group("Atracurium")

    assert group.key == "benzylisoquinolinium-neuromuscular-blockers"
    assert group.label == "benzylisoquinolinium neuromuscular blockers"
    assert group.classes == (
        "neuromuscular blocker (non-depolarising)",
        "neuromuscular blocker (benzylisoquinolinium)",
    )


def test_resolve_drug_group_generic_neuromuscular_blockade() -> None:
    group = resolve_drug_group("neuromuscular blockade")

    assert group.key == "neuromuscular-blockers"
    assert "generic-class" in group.roles