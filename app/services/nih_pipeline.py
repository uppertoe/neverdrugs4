from __future__ import annotations

import hashlib
from typing import Dict, Tuple

from sqlalchemy.orm import Session

from app.services.espell import NIHESpellClient
from app.services.mesh_builder import NIHMeshBuilder
from app.services.nih_pubmed import NIHPubMedSearcher
from app.services.search import SearchResolution, resolve_search_input
from app.settings import get_app_settings


def resolve_condition_via_nih(
    raw_condition: str,
    *,
    session: Session,
    mesh_builder: NIHMeshBuilder | None = None,
    espell_client: NIHESpellClient | None = None,
    pubmed_searcher: NIHPubMedSearcher | None = None,
    refresh_ttl_seconds: int | None = None,
) -> SearchResolution:
    builder = mesh_builder or NIHMeshBuilder()
    espell = espell_client or NIHESpellClient()
    searcher = pubmed_searcher or NIHPubMedSearcher()
    cached_signatures: Dict[Tuple[str, ...], str] = {}
    settings = get_app_settings()
    refresh_ttl = refresh_ttl_seconds or settings.search.refresh_ttl_seconds

    def _result_signature_provider(mesh_terms: list[str], _: str) -> str:
        key = tuple(mesh_terms)
        if key not in cached_signatures:
            result = searcher(mesh_terms)
            fingerprint = "|".join(result.pmids)
            cached_signatures[key] = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return cached_signatures[key]

    return resolve_search_input(
        raw_condition,
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
        refresh_ttl_seconds=refresh_ttl,
        result_signature_provider=_result_signature_provider,
    )
