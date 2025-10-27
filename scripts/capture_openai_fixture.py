#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import ArticleArtefact, ArticleSnippet, SearchTerm
from app.services.llm_batches import build_llm_batches
from app.services.openai_client import OpenAIChatClient

FIXTURE_PATH = Path("tests/fixtures/openai_duchenne_response.json")


def _seed_sample_data(session) -> int:
    term = SearchTerm(canonical="duchenne muscular dystrophy")
    session.add(term)
    session.flush()

    article = ArticleArtefact(
        search_term_id=term.id,
        pmid="11111111",
        rank=1,
        score=4.8,
        citation={
            "pmid": "11111111",
            "preferred_url": "https://pubmed.ncbi.nlm.nih.gov/11111111/",
            "title": "Safety considerations for neuromuscular blockade in dystrophinopathies",
        },
    )
    session.add(article)
    session.flush()

    snippet = ArticleSnippet(
        article_artefact_id=article.id,
        snippet_hash=f"{article.pmid}-succinylcholine-fixture",
        drug="succinylcholine",
        classification="risk",
        snippet_text=(
            "Case reports highlight malignant hyperthermia-like reactions when patients with Duchenne muscular dystrophy "
            "receive succinylcholine during anesthesia, suggesting elevated perioperative risk."
        ),
        snippet_score=5.0,
        cues=["malignant hyperthermia", "elevated risk"],
    )
    session.add(snippet)
    session.flush()

    return term.id


def capture_fixture() -> Path:
    engine = create_engine("sqlite:///:memory:", future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    Base.metadata.create_all(engine)
    session = SessionLocal()
    try:
        search_term_id = _seed_sample_data(session)
        batches = build_llm_batches(
            session,
            search_term_id=search_term_id,
            condition_label="Duchenne muscular dystrophy",
            mesh_terms=["Duchenne Muscular Dystrophy"],
        )
        if not batches:
            raise RuntimeError("No LLM batches were generated from the sample data")

        client = OpenAIChatClient()
        results = client.run_batches(batches)

        output_records = []
        for index, result in enumerate(results):
            output_records.append(
                {
                    "batch_index": index,
                    "model": result.model,
                    "response_id": result.response_id,
                    "usage": result.usage,
                    "content": result.content,
                    "parsed": result.parsed_json(),
                }
            )

        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
        return FIXTURE_PATH
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


if __name__ == "__main__":
    path = capture_fixture()
    print(f"Captured OpenAI fixture to {path}")
