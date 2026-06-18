"""
Data Pipeline — SE365 Pharma-RAG
Orchestrates: Load raw JSON → Normalize → Validate SDK → Build Drug object → Upsert to Qdrant

Usage:
    python -m src.pipeline.data_pipeline --type traditional --pages 5
    python -m src.pipeline.data_pipeline --type western     --pages 10
    python -m src.pipeline.data_pipeline --type both        --pages 5
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from src.models.drug import (
    Drug, DrugMetadata, DrugSections,
    ActiveIngredient, HerbalIngredient,
    Manufacturer, Packaging,
)
from src.crawler.tbd_crawler import TBDTraditionalCrawler
from src.crawler.tbd_western_crawler import TBDWesternCrawler
from src.crawler.dav_validator import DAVValidator
from src.crawler.ydct_validator import YDCTValidator
from src.database.qdrant_client import PharmaQdrantClient

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ──────────────────────────────────────────────────────────────────────

def _normalize_traditional(raw: Dict[str, Any]) -> Drug:
    """Convert a raw TBDTraditionalCrawler dict into a Drug Pydantic object."""
    herbal_list = [
        HerbalIngredient(
            name=h["name"],
            amount=h.get("amount"),
            role=h.get("role") or "Thành phần",
        )
        for h in raw.get("herbal_ingredients", [])
        if h.get("name")
    ]

    sections_raw = raw.get("sections", {})
    sections = DrugSections(
        indication=sections_raw.get("indication"),
        contraindication=sections_raw.get("contraindication"),
        dosage=sections_raw.get("dosage"),
        side_effects=sections_raw.get("side_effects"),
        interactions=sections_raw.get("interactions"),
        warnings=sections_raw.get("warnings"),
        pharmacology=sections_raw.get("pharmacology"),
        pharmacokinetics=sections_raw.get("pharmacokinetics"),
    )

    return Drug(
        metadata=DrugMetadata(
            name=raw["name"],
            registration_number=raw.get("registration_number") or "UNKNOWN",
            drug_type="TRADITIONAL_MEDICINE",
            herbal_ingredient_list=herbal_list,
            manufacturer=Manufacturer(
                name=raw.get("manufacturer") or "Chưa rõ",
                country=raw.get("manufacturer_country"),
            ),
            packagings=[Packaging(unit_name=raw["dosage_form"])] if raw.get("dosage_form") else [],
        ),
        sections=sections,
    )


def _normalize_western(raw: Dict[str, Any]) -> Drug:
    """Convert a raw TBDWesternCrawler dict into a Drug Pydantic object."""
    active_list = [
        ActiveIngredient(
            name=a["name"],
            is_main_active_ingredient=a.get("is_main_active_ingredient", True),
        )
        for a in raw.get("active_ingredients", [])
        if a.get("name")
    ]

    sections_raw = raw.get("sections", {})
    sections = DrugSections(
        indication=sections_raw.get("indication"),
        contraindication=sections_raw.get("contraindication"),
        dosage=sections_raw.get("dosage"),
        side_effects=sections_raw.get("side_effects"),
        interactions=sections_raw.get("interactions"),
        warnings=sections_raw.get("warnings"),
        pharmacology=sections_raw.get("pharmacology"),
        pharmacokinetics=sections_raw.get("pharmacokinetics"),
    )

    return Drug(
        metadata=DrugMetadata(
            name=raw["name"],
            registration_number=raw.get("registration_number") or "UNKNOWN",
            drug_type="WESTERN_MEDICINE",
            active_ingredient_list=active_list,
            strength=raw.get("strength"),
            manufacturer=Manufacturer(
                name=raw.get("manufacturer") or "Chưa rõ",
                country=raw.get("manufacturer_country"),
            ),
            packagings=[Packaging(unit_name=raw["dosage_form"])] if raw.get("dosage_form") else [],
        ),
        sections=sections,
    )


# ──────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────

def _validate_sdk(
    sdk: str,
    drug_type: str,
    dav: DAVValidator,
    ydct: YDCTValidator,
) -> Tuple[bool, Optional[Dict]]:
    """
    Attempt to validate an SDK against DAV or YDCT.
    Returns (is_valid, validation_result_dict).
    An SDK of 'UNKNOWN' is skipped (treated as valid to allow indexing raw crawled data).
    """
    if sdk == "UNKNOWN" or not sdk:
        return True, None

    try:
        if drug_type == "TRADITIONAL_MEDICINE":
            result = ydct.validate(sdk) or dav.validate(sdk)
        else:
            result = dav.validate(sdk) or ydct.validate(sdk)

        if result:
            return True, result
        else:
            logger.warning("[Pipeline] SDK %s not found in validator — skipping.", sdk)
            return False, None
    except Exception as exc:
        logger.warning("[Pipeline] Validation error for SDK %s: %s", sdk, exc)
        return True, None   # Network error → still index the drug


# ──────────────────────────────────────────────────────────────────────
# Main pipeline class
# ──────────────────────────────────────────────────────────────────────

class DataPipeline:
    """
    End-to-end pipeline: crawl → normalize → validate → upsert Qdrant.
    """

    def __init__(self, validate: bool = True, delay: float = 1.2):
        self.dav = DAVValidator()
        self.ydct = YDCTValidator()
        self.db = PharmaQdrantClient()
        self.validate = validate
        self.delay = delay

        # Stats
        self.stats = {
            "crawled": 0,
            "normalized": 0,
            "valid": 0,
            "invalid": 0,
            "indexed": 0,
            "errors": 0,
        }

    def run_traditional(self, max_pages: int = 5, save_raw: bool = True):
        """Run pipeline for Category N29 (Traditional/Đông y)."""
        logger.info("[Pipeline] === Starting Traditional Medicine Pipeline (N29) ===")
        crawler = TBDTraditionalCrawler(delay=self.delay)
        raw_drugs = crawler.run(max_pages=max_pages, save=save_raw)
        self.stats["crawled"] += len(raw_drugs)

        for raw in raw_drugs:
            self._process(raw, _normalize_traditional)

        self._print_stats()

    def run_western(self, max_pages: int = 10, save_raw: bool = True):
        """Run pipeline for Category N06 (Western/Tây y)."""
        logger.info("[Pipeline] === Starting Western Medicine Pipeline (N06) ===")
        crawler = TBDWesternCrawler(delay=self.delay)
        raw_drugs = crawler.run(max_pages=max_pages, save=save_raw)
        self.stats["crawled"] += len(raw_drugs)

        for raw in raw_drugs:
            self._process(raw, _normalize_western)

        self._print_stats()

    def index_from_directory(self, directory: str, drug_type: str):
        """
        Re-index previously scraped JSON files from a directory.
        drug_type: 'WESTERN_MEDICINE' | 'TRADITIONAL_MEDICINE'
        """
        path = Path(directory)
        files = list(path.glob("*.json"))
        logger.info("[Pipeline] Indexing %d files from %s", len(files), directory)

        normalizer = _normalize_traditional if drug_type == "TRADITIONAL_MEDICINE" else _normalize_western

        for fpath in files:
            try:
                with open(fpath, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        self.stats["crawled"] += 1
                        self._process(item, normalizer)
                else:
                    self.stats["crawled"] += 1
                    self._process(data, normalizer)
            except Exception as exc:
                logger.error("[Pipeline] Failed to load %s: %s", fpath, exc)
                self.stats["errors"] += 1

        self._print_stats()

    def _process(self, raw: Dict[str, Any], normalizer) -> bool:
        """Normalize, optionally validate, then upsert a single raw drug record."""
        try:
            if "metadata" in raw and "sections" in raw:
                drug = Drug.model_validate(raw)
            else:
                drug = normalizer(raw)
            self.stats["normalized"] += 1
        except Exception as exc:
            name = raw.get("name") or raw.get("metadata", {}).get("name") or "Unknown"
            logger.error("[Pipeline] Normalization failed for %s: %s", name, exc)
            self.stats["errors"] += 1
            return False

        if self.validate:
            is_valid, _ = _validate_sdk(
                drug.metadata.registration_number,
                drug.metadata.drug_type,
                self.dav,
                self.ydct,
            )
            if not is_valid:
                self.stats["invalid"] += 1
                return False
            self.stats["valid"] += 1
            time.sleep(0.3)   # Be gentle with the gov API
        else:
            self.stats["valid"] += 1

        try:
            n = self.db.upsert_drug(drug)
            self.stats["indexed"] += n
            logger.info("[Pipeline] Indexed '%s' → %d chunks.", drug.metadata.name, n)
        except Exception as exc:
            logger.error("[Pipeline] Qdrant upsert failed for '%s': %s", drug.metadata.name, exc)
            self.stats["errors"] += 1
            return False

        return True

    def _print_stats(self):
        s = self.stats
        logger.info(
            "[Pipeline] ── Summary ──\n"
            "  Crawled   : %d\n"
            "  Normalized: %d\n"
            "  Valid SDK  : %d\n"
            "  Invalid SDK: %d\n"
            "  Indexed   : %d chunks\n"
            "  Errors    : %d",
            s["crawled"], s["normalized"],
            s["valid"], s["invalid"],
            s["indexed"], s["errors"],
        )


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Pharma-RAG Data Pipeline")
    parser.add_argument("--type", choices=["western", "traditional", "both"], default="both",
                        help="Which drug category to crawl and index")
    parser.add_argument("--pages", type=int, default=5,
                        help="Number of category list pages to crawl per type")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip SDK validation against gov APIs")
    parser.add_argument("--from-dir", type=str, default=None,
                        help="Re-index from a local directory of JSON files (skips crawling)")
    parser.add_argument("--delay", type=float, default=1.2,
                        help="Delay in seconds between requests")
    args = parser.parse_args()

    pipeline = DataPipeline(validate=not args.no_validate, delay=args.delay)

    if args.from_dir:
        drug_type = (
            "TRADITIONAL_MEDICINE" if args.type == "traditional" else "WESTERN_MEDICINE"
        )
        pipeline.index_from_directory(args.from_dir, drug_type)
    else:
        if args.type in ("traditional", "both"):
            pipeline.run_traditional(max_pages=args.pages)
        if args.type in ("western", "both"):
            pipeline.run_western(max_pages=args.pages)


if __name__ == "__main__":
    main()
