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

    __table_args__ = (UniqueConstraint("search_term_id", "pmid", name="uq_article_searchterm_pmid"),)
