"""
drug_synonym_resolver.py
-------------------------
Builds and manages a bidirectional lookup between brand names (biệt dược)
and active ingredients (hoạt chất / thảo dược), enabling query expansion
for the RAG retrieval pipeline.

Architecture:
  - Auto-builds the index from existing cleaned JSON data on disk.
  - Provides `expand_query()` which returns the original query plus
    all relevant synonyms (brand -> ingredient or ingredient -> brands).
  - Thread-safe singleton pattern: build once, reuse everywhere.

Usage:
    resolver = DrugSynonymResolver()
    resolver.build_from_data()
    expanded = resolver.expand_query("Zitromax")
    # -> "Zitromax Azithromycin"
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _normalize_name(name: str) -> str:
    """Lowercase, NFC-normalize, strip whitespace."""
    return unicodedata.normalize("NFC", name.strip().lower())


def _strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for accent-insensitive matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    # Keep only ASCII chars (strip combining marks)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Fix Vietnamese special chars not decomposed by NFKD
    for src, dst in [("đ", "d"), ("Đ", "D")]:
        stripped = stripped.replace(src, dst)
    return stripped.lower().strip()


def _is_valid_ingredient(name: str) -> bool:
    """Filter out junk ingredient names from the raw data."""
    name = name.strip()
    if len(name) < 3:
        return False
    # Reject if it's just a dosage or number pattern
    if re.match(r'^[\d.,\s/]+(?:mg|g|ml|mcg|iu|%)?$', name, re.IGNORECASE):
        return False
    # Reject if it starts with a dose (e.g. "50mg Cefixime/")
    if re.match(r'^\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg)', name, re.IGNORECASE):
        return False
    # Reject common non-ingredient words
    junk = {'bột', 'nước', 'dung dịch', 'hỗn dịch', 'viên', 'gói', 'ống',
            'dạng', 'khan', 'base', 'acid', 'null', 'none', 'n/a', 'chưa rõ'}
    if name.lower() in junk:
        return False
    # Reject if mostly digits/punctuation
    alpha_ratio = sum(1 for c in name if c.isalpha()) / max(len(name), 1)
    if alpha_ratio < 0.4:
        return False
    # Reject overly long entries (likely parsing errors or verbose salt forms)
    if len(name) > 80:
        return False
    return True


def _tokenize_drug_name(name: str) -> str:
    """
    Extract the core brand name by stripping common suffixes
    like strength, dosage form, manufacturer info.
    E.g. "Augmentin 1g dang vien nen" -> "augmentin"
    """
    name = _normalize_name(name)
    # Remove strength suffixes: 500mg, 1g, 200mg/5ml, etc.
    name = re.sub(r'\s*\d+(?:[.,]\d+)?\s*(?:mg|g|mcg|ml|iu|miu|%)(?:/\S+)?', '', name)
    # Remove common dosage form keywords (Vietnamese)
    name = re.sub(
        r'\s+(?:dang|dạng)\s+.*$', '', name
    )
    # Remove trailing manufacturer/country info after " - "
    name = re.sub(r'\s+-\s+.*$', '', name)
    return name.strip()


class DrugSynonymResolver:
    """
    Bidirectional synonym resolver for pharmaceutical names.

    Lookup directions:
        brand_name  ->  Set[ingredient_name]
        ingredient  ->  Set[brand_name]
    """

    def __init__(self) -> None:
        # brand (normalized) -> set of ingredient names (original casing)
        self._brand_to_ingredients: Dict[str, Set[str]] = defaultdict(set)
        # ingredient (normalized) -> set of brand names (original casing)
        self._ingredient_to_brands: Dict[str, Set[str]] = defaultdict(set)
        # brand (normalized) -> original brand name (for display)
        self._brand_original: Dict[str, str] = {}
        # ingredient (normalized) -> original ingredient name
        self._ingredient_original: Dict[str, str] = {}
        # Track all known names for fast membership test
        self._all_names_lower: Set[str] = set()
        self._built = False

    def is_built(self) -> bool:
        return self._built

    # ──────────────────────────────────────────────────────────────────
    # Index building
    # ──────────────────────────────────────────────────────────────────

    def build_from_data(
        self,
        western_dir: Optional[str] = None,
        traditional_dir: Optional[str] = None,
    ) -> None:
        """
        Scan cleaned JSON files and populate the synonym maps.

        Args:
            western_dir:     Path to cleaned western drug JSONs.
            traditional_dir: Path to raw traditional drug JSONs.
        """
        project_root = Path(__file__).parent.parent.parent

        # Defaults
        if western_dir is None:
            western_dir = str(project_root / "src" / "database" / "Datasets" / "thuoc" / "cleaned_json")
        if traditional_dir is None:
            traditional_dir = str(project_root / "data" / "raw" / "traditional")

        # Process western medicine
        w_path = Path(western_dir)
        if w_path.exists():
            self._index_western(w_path)

        # Process traditional medicine
        t_path = Path(traditional_dir)
        if t_path.exists():
            self._index_traditional(t_path)

        self._built = True
        logger.info(
            "[SynonymResolver] Built index: %d brands, %d ingredients",
            len(self._brand_to_ingredients),
            len(self._ingredient_to_brands),
        )

    def _index_western(self, directory: Path) -> None:
        """Index western medicine cleaned JSONs (list of Drug dicts)."""
        for f in directory.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = [data]
                for d in data:
                    meta = d.get("metadata", {})
                    brand = meta.get("name", "")
                    ingredients = meta.get("active_ingredient_list", [])
                    if not brand or not ingredients:
                        continue

                    brand_key = _normalize_name(brand)
                    brand_core = _tokenize_drug_name(brand)
                    brand_stripped = _strip_diacritics(brand)
                    self._brand_original[brand_key] = brand
                    self._brand_original[brand_core] = brand
                    self._brand_original[brand_stripped] = brand
                    self._all_names_lower.add(brand_key)
                    self._all_names_lower.add(brand_core)
                    self._all_names_lower.add(brand_stripped)

                    for ing in ingredients:
                        ing_name = ing.get("name", "")
                        if not ing_name or not _is_valid_ingredient(ing_name):
                            continue
                        ing_key = _normalize_name(ing_name)
                        ing_stripped = _strip_diacritics(ing_name)
                        self._ingredient_original[ing_key] = ing_name
                        self._ingredient_original[ing_stripped] = ing_name
                        self._all_names_lower.add(ing_key)
                        self._all_names_lower.add(ing_stripped)
                        self._brand_to_ingredients[brand_key].add(ing_name)
                        self._brand_to_ingredients[brand_core].add(ing_name)
                        self._ingredient_to_brands[ing_key].add(brand)
                        self._ingredient_to_brands[ing_stripped].add(brand)
            except Exception as exc:
                logger.debug("Failed to index %s: %s", f.name, exc)

    def _index_traditional(self, directory: Path) -> None:
        """Index traditional medicine raw JSONs (single Drug dict per file)."""
        for f in directory.glob("*.json"):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                brand = d.get("name", "")
                herbs = d.get("herbal_ingredients", [])
                if not brand or not herbs:
                    continue

                brand_key = _normalize_name(brand)
                brand_core = _tokenize_drug_name(brand)
                self._brand_original[brand_key] = brand
                self._brand_original[brand_core] = brand
                self._all_names_lower.add(brand_key)
                self._all_names_lower.add(brand_core)

                for herb in herbs:
                    herb_name = herb.get("name", "")
                    if not herb_name or herb_name == "null":
                        continue
                    herb_key = _normalize_name(herb_name)
                    self._ingredient_original[herb_key] = herb_name
                    self._all_names_lower.add(herb_key)
                    self._brand_to_ingredients[brand_key].add(herb_name)
                    self._brand_to_ingredients[brand_core].add(herb_name)
                    self._ingredient_to_brands[herb_key].add(brand)
            except Exception as exc:
                logger.debug("Failed to index traditional %s: %s", f.name, exc)

    # ──────────────────────────────────────────────────────────────────
    # Lookup API
    # ──────────────────────────────────────────────────────────────────

    def get_ingredients(self, brand_name: str) -> List[str]:
        """Return ingredient names for a given brand name."""
        key = _normalize_name(brand_name)
        core = _tokenize_drug_name(brand_name)
        stripped = _strip_diacritics(brand_name)
        result = (
            self._brand_to_ingredients.get(key, set())
            | self._brand_to_ingredients.get(core, set())
            | self._brand_to_ingredients.get(stripped, set())
        )
        return [r for r in result if _is_valid_ingredient(r)]

    def get_brands(self, ingredient_name: str) -> List[str]:
        """Return brand names for a given ingredient."""
        key = _normalize_name(ingredient_name)
        stripped = _strip_diacritics(ingredient_name)
        result = (
            self._ingredient_to_brands.get(key, set())
            | self._ingredient_to_brands.get(stripped, set())
        )
        return list(result)

    def resolve(self, name: str) -> List[str]:
        """
        Given any drug-related name (brand or ingredient),
        return all synonyms/related names (not including the input itself).
        """
        synonyms: Set[str] = set()

        # Try as brand -> get ingredients
        ingredients = self.get_ingredients(name)
        synonyms.update(ingredients)

        # Try as ingredient -> get brands (limit to avoid noise)
        brands = self.get_brands(name)
        if len(brands) <= 10:
            synonyms.update(brands)
        else:
            # Too many brands — just keep the ingredient name expansion
            pass

        # Also try the reverse: if we got ingredients, get their related brand names
        # (limited scope to avoid query explosion)
        for ing in ingredients:
            related_brands = self.get_brands(ing)
            if len(related_brands) <= 5:
                synonyms.update(related_brands)

        # Remove the input itself
        synonyms.discard(name)
        key = _normalize_name(name)
        synonyms = {s for s in synonyms if _normalize_name(s) != key}

        return list(synonyms)

    def expand_query(self, query: str, max_expansion_terms: int = 5) -> str:
        """
        Expand a user query by appending synonym terms.

        Strategy:
          1. Check if any known drug/ingredient name appears in the query.
          2. For each match, resolve synonyms.
          3. Append the most relevant synonyms (ingredients preferred over brands)
             to the original query.

        Args:
            query: Original user query string.
            max_expansion_terms: Maximum number of synonym terms to append.

        Returns:
            Expanded query string.
        """
        if not self._built:
            return query

        query_lower = _normalize_name(query)
        expansion_terms: List[str] = []

        # Strategy 1: Direct lookup — check if query IS a known name
        ingredients = self.get_ingredients(query)
        if ingredients:
            # User typed a brand name -> expand with active ingredients
            expansion_terms.extend(ingredients)
        else:
            brands = self.get_brands(query)
            if brands:
                # User typed an ingredient -> keep as is (embedding already covers it)
                # But add a few representative brand names for BM25 matching
                expansion_terms.extend(brands[:3])

        # Strategy 2: Substring matching — find drug names within longer queries
        if not expansion_terms:
            # Sort known names by length (longest first) for greedy matching
            for known_name in sorted(self._all_names_lower, key=len, reverse=True):
                if len(known_name) < 3:
                    continue
                if known_name in query_lower:
                    # Found a known name in the query
                    matched_ings = self.get_ingredients(known_name)
                    if matched_ings:
                        expansion_terms.extend(matched_ings)
                    else:
                        matched_brands = self.get_brands(known_name)
                        expansion_terms.extend(matched_brands[:3])
                    break  # Only expand the first (longest) match

        if not expansion_terms:
            return query

        # Deduplicate and limit
        seen: Set[str] = set()
        unique_terms: List[str] = []
        for term in expansion_terms:
            key = _normalize_name(term)
            if key not in seen and key not in query_lower:
                seen.add(key)
                unique_terms.append(term)
            if len(unique_terms) >= max_expansion_terms:
                break

        if unique_terms:
            expanded = query + " " + " ".join(unique_terms)
            logger.info("[SynonymResolver] Expanded query: '%s' -> '%s'", query, expanded)
            return expanded

        return query

    # ──────────────────────────────────────────────────────────────────
    # Stats / Debug
    # ──────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, int]:
        """Return index statistics."""
        return {
            "total_brands": len(self._brand_to_ingredients),
            "total_ingredients": len(self._ingredient_to_brands),
            "total_names": len(self._all_names_lower),
        }


# ─── Module-level singleton ────────────────────────────────────────────────────

_global_resolver: Optional[DrugSynonymResolver] = None


def get_synonym_resolver() -> DrugSynonymResolver:
    """Return the global singleton DrugSynonymResolver, building it lazily."""
    global _global_resolver
    if _global_resolver is None or not _global_resolver.is_built():
        _global_resolver = DrugSynonymResolver()
        _global_resolver.build_from_data()
    return _global_resolver
