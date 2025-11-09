from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any, Mapping

_DRUG_FIRST_PAYLOAD_RESOURCE = "drug_first_payload.schema.json"


@lru_cache(maxsize=1)
def load_drug_first_payload_schema() -> Mapping[str, Any]:
    """Return the locked drug-first JSON schema used for LLM responses."""
    data = resources.files(__package__).joinpath(_DRUG_FIRST_PAYLOAD_RESOURCE).read_text("utf-8")
    return json.loads(data)


DRUG_FIRST_PAYLOAD_SCHEMA: Mapping[str, Any] = load_drug_first_payload_schema()

__all__ = [
    "DRUG_FIRST_PAYLOAD_SCHEMA",
    "load_drug_first_payload_schema",
]
