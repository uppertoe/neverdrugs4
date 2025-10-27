from __future__ import annotations

from datetime import datetime, timezone
from typing import List

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.database import Base


class SearchTerm(Base):
    __tablename__ = "search_terms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )

    variants: Mapped[List["SearchTermVariant"]] = relationship(
        back_populates="term",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    artefacts: Mapped[List["SearchArtefact"]] = relationship(
        back_populates="term",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    articles: Mapped[List["ArticleArtefact"]] = relationship(
        back_populates="term",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class SearchTermVariant(Base):
    __tablename__ = "search_term_variants"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    search_term_id: Mapped[int] = mapped_column(ForeignKey("search_terms.id"), nullable=False)
    value: Mapped[str] = mapped_column(String(512), nullable=False)
    normalized_value: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )

    term: Mapped[SearchTerm] = relationship(back_populates="variants")


class SearchArtefact(Base):
    __tablename__ = "search_artefacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    search_term_id: Mapped[int] = mapped_column(ForeignKey("search_terms.id"), index=True, nullable=False)
    query_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    mesh_terms: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    mesh_signature: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    result_signature: Mapped[str | None] = mapped_column(String(512), index=True, nullable=True)
    ttl_policy_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=86_400)
    last_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )

    term: Mapped[SearchTerm] = relationship(back_populates="artefacts")


class ArticleArtefact(Base):
    __tablename__ = "article_artefacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    search_term_id: Mapped[int] = mapped_column(ForeignKey("search_terms.id"), index=True, nullable=False)
    pmid: Mapped[str] = mapped_column(String(32), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    citation: Mapped[dict] = mapped_column(JSON, nullable=False)
    content: Mapped[str | None] = mapped_column(Text)
    content_source: Mapped[str | None] = mapped_column(String(32))
    token_estimate: Mapped[int | None] = mapped_column(Integer)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    term: Mapped[SearchTerm] = relationship(back_populates="articles")
    snippets: Mapped[List["ArticleSnippet"]] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (UniqueConstraint("search_term_id", "pmid", name="uq_article_searchterm_pmid"),)


class ArticleSnippet(Base):
    __tablename__ = "article_snippets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_artefact_id: Mapped[int] = mapped_column(ForeignKey("article_artefacts.id"), index=True, nullable=False)
    snippet_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    drug: Mapped[str] = mapped_column(String(128), nullable=False)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    snippet_text: Mapped[str] = mapped_column(Text, nullable=False)
    snippet_score: Mapped[float] = mapped_column(Float, nullable=False)
    cues: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    article: Mapped[ArticleArtefact] = relationship(back_populates="snippets")

    __table_args__ = (
        UniqueConstraint("article_artefact_id", "snippet_hash", name="uq_snippet_article_hash"),
    )


class ClaimSetRefresh(Base):
    __tablename__ = "claim_set_refreshes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mesh_signature: Mapped[str] = mapped_column(String(512), unique=True, index=True, nullable=False)
    job_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    error_message: Mapped[str | None] = mapped_column(Text)
    progress_state: Mapped[str | None] = mapped_column(String(64))
    progress_payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )


class ProcessedClaimSet(Base):
    __tablename__ = "processed_claim_sets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mesh_signature: Mapped[str] = mapped_column(String(512), unique=True, index=True, nullable=False)
    condition_label: Mapped[str] = mapped_column(String(512), nullable=False)
    last_search_term_id: Mapped[int | None] = mapped_column(ForeignKey("search_terms.id"), index=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    claims: Mapped[List["ProcessedClaim"]] = relationship(
        back_populates="claim_set",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ProcessedClaim(Base):
    __tablename__ = "processed_claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_set_id: Mapped[int] = mapped_column(ForeignKey("processed_claim_sets.id"), index=True, nullable=False)
    claim_id: Mapped[str] = mapped_column(String(128), nullable=False)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str] = mapped_column(String(16), nullable=False)
    drugs: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    drug_classes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    source_claim_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    claim_set: Mapped[ProcessedClaimSet] = relationship(back_populates="claims")
    evidence: Mapped[List["ProcessedClaimEvidence"]] = relationship(
        back_populates="claim",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    drug_links: Mapped[List["ProcessedClaimDrugLink"]] = relationship(
        back_populates="claim",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ProcessedClaimEvidence(Base):
    __tablename__ = "processed_claim_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("processed_claims.id"), index=True, nullable=False)
    snippet_id: Mapped[str] = mapped_column(String(64), nullable=False)
    pmid: Mapped[str] = mapped_column(String(32), nullable=False)
    article_title: Mapped[str | None] = mapped_column(String(512))
    citation_url: Mapped[str | None] = mapped_column(String(512))
    key_points: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    claim: Mapped[ProcessedClaim] = relationship(back_populates="evidence")


class ProcessedClaimDrugLink(Base):
    __tablename__ = "processed_claim_drug_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("processed_claims.id"), index=True, nullable=False)
    term: Mapped[str] = mapped_column(String(256), nullable=False)
    term_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False
    )

    claim: Mapped[ProcessedClaim] = relationship(back_populates="drug_links")

    __table_args__ = (
        UniqueConstraint("claim_id", "term", "term_kind", name="uq_claim_term_kind"),
    )
