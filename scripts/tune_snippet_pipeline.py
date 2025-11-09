from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import get_sessionmaker
from app.models import ArticleArtefact, SearchArtefact, SearchTerm
from app.services.full_text import FullTextSelectionPolicy, NIHFullTextFetcher, collect_pubmed_articles
from app.services.nih_pipeline import MeshTermsNotFoundError, resolve_condition_via_nih
from app.services.nih_pubmed import NIHPubMedSearcher
from app.services.snippet_pipeline import (
    SnippetExtractionPipeline,
    SnippetPipelineConfig,
    SnippetPostProcessor,
)
from app.services.snippet_postprocessors import LimitPerDrugPostProcessor
from app.services.snippet_tuning import (
    SnippetArticleInput,
    generate_quota_grid,
    grid_search_pipeline_configs,
)
from app.services.snippets import ArticleSnippetExtractor, SnippetResult
from app.services.snippet_scoring import (
    DEFAULT_SCORING_CONFIG,
    SnippetScoringConfig,
)
from app.services.snippet_tags import DEFAULT_RISK_CUES, DEFAULT_SAFETY_CUES
from app.settings import get_app_settings
from app.services.search import SearchResolution


class ConditionMatchFilter:
    def process(self, results: Sequence[SnippetResult]) -> Sequence[SnippetResult]:
        return [result for result in results if result.metadata.get("condition_matched")]


class MinCueCountFilter:
    def __init__(self, minimum_cues: int) -> None:
        self.minimum_cues = max(0, minimum_cues)

    def process(self, results: Sequence[SnippetResult]) -> Sequence[SnippetResult]:
        if self.minimum_cues <= 1:
            return results
        return [result for result in results if len(result.candidate.cues) >= self.minimum_cues]


def _build_condition_terms(session, search_term: SearchTerm) -> list[str]:
    stmt = (
        select(SearchArtefact)
        .where(SearchArtefact.search_term_id == search_term.id)
        .order_by(SearchArtefact.created_at.desc())
        .limit(1)
    )
    artefact = session.execute(stmt).scalar_one_or_none()
    terms: list[str] = []
    if search_term.canonical:
        terms.append(search_term.canonical)
    if artefact is not None and artefact.mesh_terms:
        terms.extend(term for term in artefact.mesh_terms if term)
    return terms


def _load_articles(
    session,
    search_term: SearchTerm,
    *,
    max_articles: int,
) -> list[ArticleArtefact]:
    stmt = (
        select(ArticleArtefact)
        .where(ArticleArtefact.search_term_id == search_term.id)
        .order_by(ArticleArtefact.rank.asc())
        .limit(max_articles)
    )
    return list(session.execute(stmt).scalars())


def _to_snippet_input(
    article: ArticleArtefact,
    condition_terms: Sequence[str],
) -> SnippetArticleInput | None:
    text = article.content or ""
    if not text:
        return None

    citation = article.citation or {}
    preferred_url = citation.get("preferred_url") if isinstance(citation, dict) else None
    if not preferred_url:
        preferred_url = f"https://pubmed.ncbi.nlm.nih.gov/{article.pmid}/"

    pmc_ref_count_val = citation.get("pmc_ref_count") if isinstance(citation, dict) else None
    try:
        pmc_ref_count = int(pmc_ref_count_val) if pmc_ref_count_val is not None else 0
    except (TypeError, ValueError):
        pmc_ref_count = 0

    return SnippetArticleInput(
        article_text=text,
        pmid=article.pmid,
        condition_terms=list(condition_terms),
        article_rank=article.rank,
        article_score=article.score,
        preferred_url=preferred_url,
        pmc_ref_count=pmc_ref_count,
    )


def _build_evaluator(
    *,
    weight_score: float,
    weight_count: float,
    weight_diversity: float,
    weight_coverage: float,
    weight_risk: float,
    weight_safety: float,
    weight_condition_match: float,
    weight_cues: float,
    required_classes: Sequence[str],
):
    required_set = tuple(required_classes)

    def evaluate(results: Sequence[SnippetResult]) -> float:
        if not results:
            return -1.0

        scores = [result.candidate.snippet_score for result in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        unique_drugs = len({result.candidate.drug for result in results})
        counts = Counter(result.candidate.classification for result in results)
        coverage_hits = sum(1 for cls in required_set if counts.get(cls, 0) > 0)
        coverage_ratio = coverage_hits / len(required_set) if required_set else 1.0
        risk_ratio = counts.get("risk", 0) / len(results)
        safety_ratio = counts.get("safety", 0) / len(results)
        condition_hits = sum(1 for result in results if result.metadata.get("condition_matched"))
        condition_ratio = condition_hits / len(results)
        avg_cues = sum(len(result.candidate.cues) for result in results) / len(results)

        total = 0.0
        total += weight_score * avg_score
        total += weight_diversity * unique_drugs
        total += weight_count * len(results)
        total += weight_coverage * coverage_ratio
        total += weight_risk * risk_ratio
        total += weight_safety * safety_ratio
        total += weight_condition_match * condition_ratio
        total += weight_cues * avg_cues
        return total

    return evaluate


def _render_results(
    *,
    pipeline: SnippetExtractionPipeline,
    articles: Sequence[tuple[ArticleArtefact, SnippetArticleInput]],
    top_k: int,
) -> list[dict[str, object]]:
    rendered: list[dict[str, object]] = []
    for artefact, article_input in articles:
        results = pipeline.run_results(**asdict(article_input))
        preview = []
        for result in results[:top_k]:
            tags = [
                {
                    "kind": tag.kind,
                    "label": tag.label,
                    "confidence": tag.confidence,
                    "source": tag.source,
                }
                for tag in result.candidate.tags
            ]
            preview.append(
                {
                    "pmid": result.candidate.pmid,
                    "drug": result.candidate.drug,
                    "classification": result.candidate.classification,
                    "score": result.candidate.snippet_score,
                    "snippet": result.candidate.snippet_text,
                    "cues": result.candidate.cues,
                    "tags": tags,
                    "metadata": result.metadata,
                }
            )
        rendered.append(
            {
                "pmid": artefact.pmid,
                "article_rank": artefact.rank,
                "article_score": artefact.score,
                "preferred_url": article_input.preferred_url,
                "snippets": preview,
            }
        )
    return rendered


def _load_cues(
    defaults: Sequence[str],
    *,
    file_path: str | None,
    extras: Sequence[str],
) -> tuple[str, ...]:
    cues = [cue for cue in defaults if cue]
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise SystemExit(f"Cue file '{file_path}' not found")
        for line in path.read_text(encoding="utf-8").splitlines():
            normalized = line.strip()
            if normalized and not normalized.startswith("#"):
                cues.append(normalized)
    cues.extend(extras or [])
    seen: set[str] = set()
    ordered: list[str] = []
    for cue in cues:
        normalized = cue.strip()
        if not normalized:
            continue
        lower = normalized.lower()
        if lower in seen:
            continue
        seen.add(lower)
        ordered.append(normalized)
    return tuple(ordered)


def _build_scoring_config(args: argparse.Namespace) -> SnippetScoringConfig:
    return SnippetScoringConfig(
        article_score_weight=args.scoring_article_weight,
        pmc_ref_weight=args.scoring_pmc_weight,
        pmc_ref_cap=args.scoring_pmc_cap,
        risk_bonus=args.scoring_risk_bonus,
        safety_bonus=args.scoring_safety_bonus,
        cue_weight=args.scoring_cue_weight,
        condition_bonus=args.scoring_condition_bonus,
        condition_penalty=args.scoring_condition_penalty,
    )
def _build_post_processors(
    limit_per_drug: int,
    *,
    require_condition_match: bool,
    min_cue_count: int,
) -> list[SnippetPostProcessor]:
    processors: list[SnippetPostProcessor] = []
    if require_condition_match:
        processors.append(ConditionMatchFilter())
    if min_cue_count and min_cue_count > 1:
        processors.append(MinCueCountFilter(min_cue_count))
    if limit_per_drug > 0:
        processors.append(LimitPerDrugPostProcessor(max_per_drug=limit_per_drug))
    return processors


def _build_extractor(
    args: argparse.Namespace,
    *,
    scoring_config: SnippetScoringConfig,
) -> ArticleSnippetExtractor:
    risk_cues = _load_cues(
        DEFAULT_RISK_CUES,
        file_path=args.risk_cue_file,
        extras=args.risk_cue or [],
    )
    safety_cues = _load_cues(
        DEFAULT_SAFETY_CUES,
        file_path=args.safety_cue_file,
        extras=args.safety_cue or [],
    )
    return ArticleSnippetExtractor(
        risk_cues=risk_cues,
        safety_cues=safety_cues,
        window_chars=args.window_chars,
        min_snippet_chars=args.min_snippet_chars,
        scoring_config=scoring_config,
    )


def _count_articles(session: Session, search_term_id: int) -> int:
    stmt = (
        select(func.count())
        .select_from(ArticleArtefact)
        .where(ArticleArtefact.search_term_id == search_term_id)
    )
    return int(session.execute(stmt).scalar_one())


def _run_fetch_pipeline(
    session: Session,
    *,
    resolution: SearchResolution,
    raw_term: str,
    post_processors: Sequence[SnippetPostProcessor],
    fetch_base_quota: int,
    fetch_max_quota: int,
    pubmed_retmax: int | None,
    full_text_base: int | None,
    full_text_max: int | None,
    extractor: ArticleSnippetExtractor,
) -> None:
    settings = get_app_settings()
    article_settings = settings.article_selection

    resolved_pubmed_retmax = (
        pubmed_retmax
        if pubmed_retmax is not None and pubmed_retmax > 0
        else article_settings.pubmed_retmax
    )
    resolved_base_full_text = (
        full_text_base
        if full_text_base is not None and full_text_base > 0
        else article_settings.base_full_text_articles
    )
    resolved_max_full_text = (
        full_text_max
        if full_text_max is not None and full_text_max > 0
        else article_settings.max_full_text_articles
    )

    selection_policy = FullTextSelectionPolicy(
        base_full_text=resolved_base_full_text,
        max_full_text=resolved_max_full_text,
    )
    searcher = NIHPubMedSearcher(retmax=resolved_pubmed_retmax)
    fetcher = NIHFullTextFetcher()

    per_drug_limit = max(1, fetch_base_quota)
    total_cap = None
    if fetch_max_quota and fetch_max_quota > 0:
        total_cap = max(per_drug_limit, fetch_max_quota)
    pipeline = SnippetExtractionPipeline(
        extractor=extractor,
        post_processors=tuple(post_processors),
        config=SnippetPipelineConfig(
            per_drug_limit=per_drug_limit,
            max_total_snippets=total_cap,
        ),
    )

    additional_terms = [raw_term]
    if resolution.normalized_condition and resolution.normalized_condition not in additional_terms:
        additional_terms.append(resolution.normalized_condition)

    collect_pubmed_articles(
        resolution,
        session=session,
        pubmed_searcher=lambda mesh_terms: searcher(
            mesh_terms,
            additional_text_terms=additional_terms,
        ),
        full_text_fetcher=fetcher,
        selection_policy=selection_policy,
        snippet_pipeline=pipeline,
    )
    session.commit()


def _ensure_articles_available(
    session: Session,
    *,
    raw_term: str,
    skip_fetch: bool,
    force_refresh: bool,
    post_processors: Sequence[SnippetPostProcessor],
    fetch_base_quota: int,
    fetch_max_quota: int,
    pubmed_retmax: int | None,
    full_text_base: int | None,
    full_text_max: int | None,
    extractor: ArticleSnippetExtractor,
) -> tuple[SearchTerm, SearchResolution]:
    settings = get_app_settings()
    try:
        resolution = resolve_condition_via_nih(
            raw_term,
            session=session,
            refresh_ttl_seconds=settings.search.refresh_ttl_seconds,
        )
    except MeshTermsNotFoundError as exc:
        suggestions = ", ".join(exc.suggestions) if exc.suggestions else "none"
        raise SystemExit(
            f"No MeSH terms found for '{raw_term}'. Suggested alternatives: {suggestions}"
        ) from exc

    search_term = session.get(SearchTerm, resolution.search_term_id)
    if search_term is None:
        raise SystemExit("Resolved search term could not be loaded from database")

    article_count = _count_articles(session, search_term.id)
    needs_fetch = force_refresh or article_count == 0

    if needs_fetch and skip_fetch:
        print(
            "Warning: no articles available for the term and fetching is disabled; tuning may fail.",
            flush=True,
        )
    elif needs_fetch:
        print("Fetching articles via NIH pipeline...", flush=True)
        _run_fetch_pipeline(
            session,
            resolution=resolution,
            raw_term=raw_term,
            post_processors=post_processors,
            fetch_base_quota=fetch_base_quota,
            fetch_max_quota=fetch_max_quota,
            pubmed_retmax=pubmed_retmax,
            full_text_base=full_text_base,
            full_text_max=full_text_max,
            extractor=extractor,
        )
        session.expire_all()
        search_term = session.get(SearchTerm, resolution.search_term_id)
        if search_term is None:
            raise SystemExit("Failed to reload search term after fetching articles")
    else:
        print("Using existing stored articles (use --force-refresh to re-fetch).", flush=True)

    return search_term, resolution


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune snippet pipeline limits on live data.")
    parser.add_argument("search_term", help="Canonical search term to evaluate (case-insensitive).")
    parser.add_argument("--max-articles", type=int, default=25, help="Maximum number of articles to load.")
    parser.add_argument("--base-range", nargs="*", type=int, default=[2, 3, 4], help="Candidate per-drug limits to evaluate.")
    parser.add_argument("--max-range", nargs="*", type=int, default=[0, 6, 9, 12], help="Candidate total snippet caps (use 0 to allow auto cap).")
    parser.add_argument("--limit-per-drug", type=int, default=2, help="Post-processor cap per drug.")
    parser.add_argument("--score-weight", type=float, default=1.0, help="Weight for average snippet score.")
    parser.add_argument("--count-weight", type=float, default=0.1, help="Weight for snippet count.")
    parser.add_argument("--diversity-weight", type=float, default=0.25, help="Weight for drug diversity.")
    parser.add_argument("--coverage-weight", type=float, default=0.5, help="Weight for classification coverage.")
    parser.add_argument("--risk-weight", type=float, default=0.5, help="Weight for risk-class coverage.")
    parser.add_argument("--safety-weight", type=float, default=0.3, help="Weight for safety-class coverage.")
    parser.add_argument("--condition-match-weight", type=float, default=0.6, help="Weight for proportion of snippets matching the condition context.")
    parser.add_argument("--cue-weight", type=float, default=0.1, help="Weight for average cue count per snippet.")
    parser.add_argument(
        "--required-classes",
        nargs="*",
        default=["risk", "safety"],
        help="Classifications to treat as required when computing coverage.",
    )
    parser.add_argument("--window-chars", type=int, default=600, help="Window size used when extracting snippet candidates.")
    parser.add_argument("--min-snippet-chars", type=int, default=60, help="Minimum snippet length after trimming.")
    parser.add_argument("--risk-cue-file", type=str, help="Newline-delimited risk cues to merge before extraction.")
    parser.add_argument("--risk-cue", action="append", default=[], metavar="CUE", help="Additional risk cue to include.")
    parser.add_argument("--safety-cue-file", type=str, help="Newline-delimited safety cues to merge before extraction.")
    parser.add_argument("--safety-cue", action="append", default=[], metavar="CUE", help="Additional safety cue to include.")
    parser.add_argument("--scoring-article-weight", type=float, default=DEFAULT_SCORING_CONFIG.article_score_weight, help="Multiplier applied to article score when computing snippet score.")
    parser.add_argument("--scoring-pmc-weight", type=float, default=DEFAULT_SCORING_CONFIG.pmc_ref_weight, help="Weight applied per PMC reference.")
    parser.add_argument("--scoring-pmc-cap", type=float, default=DEFAULT_SCORING_CONFIG.pmc_ref_cap, help="Maximum PMC contribution to snippet score.")
    parser.add_argument("--scoring-risk-bonus", type=float, default=DEFAULT_SCORING_CONFIG.risk_bonus, help="Bonus applied to risk snippets.")
    parser.add_argument("--scoring-safety-bonus", type=float, default=DEFAULT_SCORING_CONFIG.safety_bonus, help="Bonus applied to safety snippets.")
    parser.add_argument("--scoring-cue-weight", type=float, default=DEFAULT_SCORING_CONFIG.cue_weight, help="Per-cue weight added to snippet score.")
    parser.add_argument("--scoring-condition-bonus", type=float, default=DEFAULT_SCORING_CONFIG.condition_bonus, help="Bonus applied when snippet matches condition context.")
    parser.add_argument("--scoring-condition-penalty", type=float, default=DEFAULT_SCORING_CONFIG.condition_penalty, help="Penalty applied when condition context is missing.")
    parser.add_argument("--top-configs", type=int, default=5, help="Number of top configs to display.")
    parser.add_argument("--preview", type=int, default=3, help="Snippets to show per article for best config.")
    parser.add_argument("--output", type=str, help="Optional path to write JSON preview output.")
    parser.add_argument(
        "--require-condition-match",
        action="store_true",
        help="Filter out snippets that do not explicitly match the resolved condition.",
    )
    parser.add_argument(
        "--min-cue-count",
        type=int,
        default=1,
        help="Discard snippets with fewer than this many cues (after post-processing).",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip running the NIH fetching pipeline; rely solely on stored articles.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Always run the NIH fetching pipeline even if stored articles exist.",
    )
    parser.add_argument(
        "--pubmed-retmax",
        type=int,
        default=None,
        help="Override PubMed retmax when fetching articles.",
    )
    parser.add_argument(
        "--full-text-base",
        type=int,
        default=None,
        help="Override base number of full-text articles to fetch.",
    )
    parser.add_argument(
        "--full-text-max",
        type=int,
        default=None,
        help="Override maximum number of full-text articles to fetch.",
    )
    parser.add_argument(
        "--fetch-base-quota",
        type=int,
        default=None,
        help="Per-drug limit applied when persisting snippets during fetching.",
    )
    parser.add_argument(
        "--fetch-max-quota",
        type=int,
        default=None,
        help="Overall snippet cap applied during fetching (<=0 uses auto cap).",
    )

    args = parser.parse_args()

    scoring_config = _build_scoring_config(args)
    extractor = _build_extractor(args, scoring_config=scoring_config)

    post_processors = _build_post_processors(
        args.limit_per_drug,
        require_condition_match=args.require_condition_match,
        min_cue_count=args.min_cue_count,
    )
    base_range = args.base_range or [3]
    max_range = args.max_range or [4]
    default_fetch_base = max(base_range)
    default_fetch_max = max(max_range + [default_fetch_base])
    fetch_base_quota = (
        args.fetch_base_quota if args.fetch_base_quota is not None else default_fetch_base
    )
    fetch_max_quota = (
        args.fetch_max_quota if args.fetch_max_quota is not None else default_fetch_max
    )

    SessionLocal = get_sessionmaker()
    with SessionLocal() as session:
        raw_term = args.search_term.strip()
        search_term, resolution = _ensure_articles_available(
            session,
            raw_term=raw_term,
            skip_fetch=args.skip_fetch,
            force_refresh=args.force_refresh,
            post_processors=post_processors,
            fetch_base_quota=fetch_base_quota,
            fetch_max_quota=fetch_max_quota,
            pubmed_retmax=args.pubmed_retmax,
            full_text_base=args.full_text_base,
            full_text_max=args.full_text_max,
            extractor=extractor,
        )

        condition_terms = _build_condition_terms(session, search_term)
        if not condition_terms:
            fallback_terms = list(resolution.mesh_terms)
            if resolution.normalized_condition:
                fallback_terms.append(resolution.normalized_condition)
            condition_terms = fallback_terms
        if not condition_terms:
            raise SystemExit("No condition terms available after fetching; cannot tune pipeline")
        condition_terms = list(dict.fromkeys(condition_terms))

        artefacts = _load_articles(session, search_term, max_articles=args.max_articles)
        articles: list[tuple[ArticleArtefact, SnippetArticleInput]] = []
        for artefact in artefacts:
            snippet_input = _to_snippet_input(artefact, condition_terms)
            if snippet_input is None:
                continue
            articles.append((artefact, snippet_input))

        if not articles:
            raise SystemExit("No articles with full text available for tuning")

        configs = generate_quota_grid(
            per_drug_limits=base_range,
            max_total_results=max_range,
        )
        if not configs:
            raise SystemExit("No pipeline configurations generated; check --base-range/--max-range")

        evaluator = _build_evaluator(
            weight_score=args.score_weight,
            weight_count=args.count_weight,
            weight_diversity=args.diversity_weight,
            weight_coverage=args.coverage_weight,
            weight_risk=args.risk_weight,
            weight_safety=args.safety_weight,
            weight_condition_match=args.condition_match_weight,
            weight_cues=args.cue_weight,
            required_classes=args.required_classes,
        )

        extractor_pipeline = SnippetExtractionPipeline(
            extractor=extractor,
            post_processors=tuple(post_processors),
        )
        tuning_results = grid_search_pipeline_configs(
            configs,
            articles=[article_input for _, article_input in articles],
            evaluate_results=evaluator,
            extractor=extractor_pipeline.extractor,
            post_processors=extractor_pipeline.post_processors,
        )

        if not tuning_results:
            raise SystemExit("Tuning produced no results")

        top_configs = tuning_results[: args.top_configs]
        print("Top configurations:")
        for result in top_configs:
            print(
                f"  per_drug_limit={result.config.per_drug_limit:>2} total_cap={result.config.max_total_snippets if result.config.max_total_snippets is not None else 'auto':>4} "
                f"score={result.score:.4f}"
            )

        best_config = top_configs[0].config
        print("\nPreviewing snippets using best configuration...")
        pipeline = SnippetExtractionPipeline(
            extractor=extractor_pipeline.extractor,
            post_processors=extractor_pipeline.post_processors,
            config=best_config,
        )
        preview = _render_results(
            pipeline=pipeline,
            articles=articles,
            top_k=args.preview,
        )
        for entry in preview:
            print(f"\nPMID {entry['pmid']} (rank={entry['article_rank']} score={entry['article_score']:.2f})")
            for snippet in entry["snippets"]:
                tag_summary = ", ".join(
                    f"{tag['kind']}:{tag['label']}" for tag in snippet.get("tags", [])
                )
                if tag_summary:
                    tag_summary = f" tags=[{tag_summary}]"
                print(
                    f"  [{snippet['classification']}] {snippet['drug']} score={snippet['score']:.2f}{tag_summary}\n"
                    f"    {snippet['snippet']}"
                )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump({
                    "search_term": search_term.canonical,
                    "best_config": {
                        "per_drug_limit": best_config.per_drug_limit,
                        "max_total_snippets": best_config.max_total_snippets,
                    },
                    "preview": preview,
                }, handle, indent=2)
            print(f"\nPreview written to {args.output}")


if __name__ == "__main__":
    main()
