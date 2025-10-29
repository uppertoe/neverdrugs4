from __future__ import annotations

import argparse
import ast
import html.parser
import re
import shutil
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse
from zipfile import ZipFile

BASE_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/"
INVALID_MARKER = "# INVALID_MESH:"

__all__ = [
    "BASE_URL",
    "fetch_mesh_descriptor_xml",
    "load_mesh_descriptors",
    "extract_mesh_terms",
]


class MeshTermVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.mesh_terms: dict[str, list[str]] = {}
        self.term_positions: dict[str, list[int]] = defaultdict(list)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: D401 - simple ast visitor
        if len(node.targets) != 1:
            return
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return
        name = target.id
        if not name.endswith("_MESH_TERMS"):
            return
        value = node.value
        if not isinstance(value, (ast.Tuple, ast.List)):
            return
        terms: list[str] = []
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                term = element.value
                terms.append(term)
                self.term_positions[term].append(element.lineno - 1)
        if terms:
            self.mesh_terms[name] = terms


def _strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _get_remote_last_modified(url: str) -> str | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:  # noqa: S310 external request
            return response.headers.get("Last-Modified")
    except Exception:  # pragma: no cover - network/head failures fall back to cached copy
        return None


def _download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, open(destination, "wb") as dest:  # noqa: S310 - trusted source
        shutil.copyfileobj(response, dest)


class _MeshIndexParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:  # noqa: D401 - HTML parser hook
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.links.append(href)


def _resolve_mesh_dataset_url(base_url: str, explicit_url: str | None) -> str:
    if explicit_url:
        return explicit_url

    try:
        with urllib.request.urlopen(base_url) as response:  # noqa: S310
            content = response.read().decode("utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - network failures fallback to default
        raise SystemExit(
            "Failed to retrieve MeSH index page; specify --mesh-url manually."
        ) from exc

    parser = _MeshIndexParser()
    parser.feed(content)

    candidates: list[tuple[int, str]] = []
    for link in parser.links:
        if not link.lower().endswith(('.zip', '.xml')):
            continue
        name = Path(link).name
        if not name.startswith("desc"):
            continue
        try:
            year_part = name[4:8]
            year = int(year_part)
        except ValueError:
            continue
        candidates.append((year, link))

    if not candidates:
        raise SystemExit("Could not locate descriptor dataset on the MeSH index page.")

    latest_year, latest_link = max(candidates, key=lambda item: item[0])
    resolved = urljoin(base_url, latest_link)
    print(f"Resolved MeSH descriptor dataset for {latest_year}: {resolved}")
    return resolved


def fetch_mesh_descriptor_xml(url: str | None, *, cache_dir: Path | None) -> Path:
    """Download (or reuse) the MeSH descriptor XML, supporting zipped archives."""

    resolved_url = _resolve_mesh_dataset_url(BASE_URL, url)

    parsed_name = Path(urlparse(resolved_url).path).name or "mesh_dataset"
    suffix = Path(parsed_name).suffix.lower()

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = cache_dir / parsed_name
        metadata_path = dataset_path.with_suffix(dataset_path.suffix + ".meta")
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="mesh_"))
        dataset_path = temp_dir / parsed_name
        metadata_path = temp_dir / (parsed_name + ".meta")

    need_download = True
    downloaded_this_run = False
    remote_last_modified = _get_remote_last_modified(resolved_url) if dataset_path.exists() else None
    if dataset_path.exists():
        if remote_last_modified and metadata_path.exists():
            cached_last_modified = metadata_path.read_text(encoding="utf-8").strip()
            if cached_last_modified == remote_last_modified:
                need_download = False
        elif cache_dir is not None:
            # Without remote metadata, reuse cached asset.
            need_download = False

    if need_download:
        _download_file(resolved_url, dataset_path)
        downloaded_this_run = True
        if remote_last_modified:
            metadata_path.write_text(remote_last_modified, encoding="utf-8")

    if suffix == ".zip":
        try:
            archive = ZipFile(dataset_path)
        except Exception:
            _download_file(resolved_url, dataset_path)
            downloaded_this_run = True
            archive = ZipFile(dataset_path)
        with archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".xml")]
            if not members:
                raise SystemExit(f"Archive '{dataset_path}' does not contain an XML descriptor file")
            xml_name = sorted(members)[0]
            if cache_dir is not None:
                xml_target = dataset_path.with_suffix(".xml")
            else:
                xml_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
                xml_target = Path(xml_handle.name)
                xml_handle.close()
            needs_extract = downloaded_this_run or not xml_target.exists()
            if xml_target.exists():
                archive_info = archive.getinfo(xml_name)
                needs_extract = needs_extract or archive_info.file_size != xml_target.stat().st_size
            if needs_extract:
                with archive.open(xml_name) as source, open(xml_target, "wb") as dest:
                    shutil.copyfileobj(source, dest)
            if cache_dir is None and dataset_path.exists():
                dataset_path.unlink(missing_ok=True)
            return xml_target

    return dataset_path


def load_mesh_descriptors(path: Path) -> set[str]:
    descriptors: set[str] = set()
    try:
        for _event, elem in ET.iterparse(path, events=("end",)):
            tag = _strip_namespace(elem.tag)
            if tag != "DescriptorRecord":
                continue
            name_elem = elem.find("DescriptorName/String")
            if name_elem is not None and name_elem.text:
                descriptors.add(name_elem.text.strip().lower())
            elem.clear()
    except ET.ParseError as exc:  # pragma: no cover - unexpected XML format
        raise SystemExit(f"Failed to parse MeSH descriptor file '{path}': {exc}") from exc
    if not descriptors:
        raise SystemExit(f"No descriptors loaded from '{path}'")
    return descriptors


def extract_mesh_terms(source: Path) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    module_ast = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    visitor = MeshTermVisitor()
    visitor.visit(module_ast)
    return visitor.mesh_terms, visitor.term_positions


def update_query_file(
    file_path: Path,
    *,
    invalid_terms: Iterable[str],
    all_terms: Iterable[str],
    term_positions: dict[str, list[int]],
    dry_run: bool,
) -> bool:
    invalid_set = {term.lower() for term in invalid_terms}
    all_terms_lower = {term.lower() for term in all_terms}
    lines = file_path.read_text(encoding="utf-8").splitlines()
    modified = False

    # Comment out invalid terms
    for term in term_positions:
        if term.lower() not in invalid_set:
            continue
        for index in term_positions[term]:
            if index >= len(lines):
                continue
            current = lines[index]
            if INVALID_MARKER in current:
                continue
            indent = current[: len(current) - len(current.lstrip())]
            lines[index] = f"{indent}{INVALID_MARKER} \"{term}\","  # keep trailing comma for context
            modified = True

    invalid_pattern = re.compile(rf"^\s*{re.escape(INVALID_MARKER)}\s*\"(.*?)\",\s*$")

    # Restore previously commented terms that are now valid
    for idx, line in enumerate(lines):
        match = invalid_pattern.match(line)
        if not match:
            continue
        term = match.group(1)
        if term.lower() in invalid_set:
            continue
        if term.lower() not in all_terms_lower:
            continue
        indent = line[: len(line) - len(line.lstrip())]
        lines[idx] = f"{indent}\"{term}\","  # restore original entry
        modified = True

    if modified and not dry_run:
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return modified


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MeSH terms used in query_terms.py")
    parser.add_argument(
        "--query-file",
        type=Path,
        default=Path("app/services/query_terms.py"),
        help="Path to the module containing *_MESH_TERMS constants.",
    )
    parser.add_argument(
        "--mesh-url",
        type=str,
        help="Optional URL to the MeSH descriptor XML (or zipped) file; latest dataset used when omitted.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".mesh_cache"),
        help="Directory to cache the downloaded MeSH dataset (ignored by git).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without modifying the query file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.query_file.exists():
        raise SystemExit(f"Query file '{args.query_file}' not found")

    mesh_terms, term_positions = extract_mesh_terms(args.query_file)
    if not mesh_terms:
        print("No *_MESH_TERMS constants found; nothing to validate.")
        return 0

    descriptor_path = fetch_mesh_descriptor_xml(args.mesh_url, cache_dir=args.cache_dir)
    descriptors = load_mesh_descriptors(descriptor_path)

    invalid_terms: dict[str, list[str]] = {}
    for name, terms in mesh_terms.items():
        invalid = [term for term in terms if term.lower() not in descriptors]
        if invalid:
            invalid_terms[name] = invalid

    if invalid_terms:
        print("Invalid MeSH terms detected:")
        for name, terms in invalid_terms.items():
            for term in terms:
                print(f"  - {name}: '{term}'")
        update_query_file(
            args.query_file,
            invalid_terms=(term for terms in invalid_terms.values() for term in terms),
            all_terms=(term for terms in mesh_terms.values() for term in terms),
            term_positions=term_positions,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print("Dry run completed; no files modified.")
        else:
            print(
                "Invalid terms have been commented out with an 'INVALID_MESH' marker."
            )
        return 1

    # Ensure we restore any previously commented terms that are now valid
    restored = update_query_file(
        args.query_file,
        invalid_terms=(),
        all_terms=(term for terms in mesh_terms.values() for term in terms),
        term_positions=term_positions,
        dry_run=args.dry_run,
    )
    if restored:
        print("Previously flagged terms restored; query file updated.")
        return 0

    print("All MeSH terms are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
