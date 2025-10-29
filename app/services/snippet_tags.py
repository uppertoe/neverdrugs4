from __future__ import annotations

from dataclasses import dataclass

NEGATING_RISK_PATTERNS: tuple[str, ...] = (
    "no {}",
    "not {}",
    "without {}",
    "absence of {}",
    "does not cause {}",
    "did not cause {}",
    "doesn't cause {}",
    "didn't cause {}",
    "not associated with {}",
    "no evidence of {}",
)

DEFAULT_RISK_CUES: tuple[str, ...] = (
    "adverse",
    "adverse event",
    "adverse events",
    "arrhythmia",
    "arrhythmias",
    "avoid",
    "avoided",
    "cardiac arrest",
    "complication",
    "complications",
    "contraindicated",
    "contraindication",
    "contraindications",
    "caution",
    "danger",
    "deterioration",
    "do not use",
    "exacerbate",
    "exacerbated",
    "fatal",
    "harmful",
    "hazard",
    "hyperkalemia",
    "hyperkalaemia",
    "life-threatening",
    "malignant hyperthermia",
    "mh",
    "precaution",
    "precipitate",
    "risk",
    "risk of",
    "rhabdomyolysis",
    "serious adverse",
    "should not",
    "side effect",
    "side effects",
    "toxic",
    "toxicity",
    "trigger",
    "triggered",
    "triggering",
    "unsafe",
    "worsened",
)

DEFAULT_SAFETY_CUES: tuple[str, ...] = (
    "acceptable safety",
    "beneficial",
    "did not cause",
    "effective",
    "generally safe",
    "no adverse event",
    "no adverse events",
    "no complications",
    "no major complications",
    "no reported complications",
    "no significant adverse",
    "recommended",
    "safe",
    "safe option",
    "safely",
    "safety",
    "successfully",
    "tolerated well",
    "well tolerated",
    "well-tolerated",
    "without complications",
    "ameliorated",
    "ameliorate",
    "prevent",
    "preventative",
    "indicated",
    "efficacious",
    "efficacy",
)

SEVERE_REACTION_ALWAYS_CUES: tuple[str, ...] = (
    "arrhythmia",
    "atrial fibrillation",
    "ventricular arrhythmia",
    "ventricular tachycardia",
    "torsades de pointes",
    "ventricular fibrillation",
    "cardiac arrest",
    "asystole",
    "pulseless electrical activity",
    "hemodynamic collapse",
    "cardiovascular collapse",
    "anaphylaxis",
    "anaphylactic shock",
    "refractory hypotension",
    "refractory bradycardia",
)

SEVERE_REACTION_CONDITIONAL_CUES: tuple[str, ...] = (
    "status epilepticus",
    "seizure",
    "seizures",
    "convulsion",
    "convulsions",
)

SEVERE_REACTION_QUALIFIERS: tuple[str, ...] = (
    "unexpected",
    "unanticipated",
    "unforeseen",
    "severe",
    "sudden",
    "prolonged",
    "refractory",
    "life-threatening",
    "catastrophic",
    "critical",
)

IGNORED_SEVERE_REACTION_CUES: tuple[str, ...] = (
    "respiratory depression",
    "mild hypotension",
    "transient hypotension",
    "expected hypotension",
    "sedation",
    "drowsiness",
)

THERAPY_ROLE_CUES: dict[str, tuple[str, ...]] = {
    "rescue": (
        "rescue therapy",
        "rescue treatment",
        "used as rescue",
        "served as rescue",
    ),
    "maintenance": (
        "maintenance infusion",
        "maintenance therapy",
    ),
    "induction": (
        "induction agent",
        "used for induction",
        "induction dose",
    ),
    "prophylaxis": (
        "prophylaxis",
        "prophylactic",
    ),
    "adjunct": (
        "adjunct therapy",
        "adjunctive",
    ),
    "alternative": (
        "alternative to",
        "used instead of",
    ),
}

MECHANISM_ALERT_CUES: dict[str, tuple[str, ...]] = {
    "pseudocholinesterase deficiency": (
        "pseudocholinesterase deficiency",
        "butyrylcholinesterase deficiency",
    ),
    "ryanodine receptor dysfunction": (
        "ryr1 mutation",
        "ryanodine receptor",
    ),
    "mitochondrial toxicity": (
        "mitochondrial toxicity",
        "mitochondrial dysfunction",
    ),
    "sodium channelopathy": (
        "scn1a mutation",
        "nav1.1",
        "sodium channelopathy",
    ),
    "malignant hyperthermia susceptibility": (
        "malignant hyperthermia susceptibility",
        "mh-susceptible",
    ),
}


@dataclass(frozen=True)
class Tag:
    kind: str
    label: str
    confidence: float
    source: str



