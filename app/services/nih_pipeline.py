from __future__ import annotations

from sqlalchemy.orm import Session

from app.services.espell import NIHESpellClient
from app.services.mesh_builder import NIHMeshBuilder
from app.services.search import SearchResolution, resolve_search_input


def resolve_condition_via_nih(
    raw_condition: str,
    *,
    session: Session,
    mesh_builder: NIHMeshBuilder | None = None,
    espell_client: NIHESpellClient | None = None,
) -> SearchResolution:
    builder = mesh_builder or NIHMeshBuilder()
    espell = espell_client or NIHESpellClient()
    return resolve_search_input(
        raw_condition,
        session=session,
        mesh_builder=builder,
        espell_fetcher=espell,
    )
