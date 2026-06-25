import os
import json
import time
import random
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from crawler.tbd_crawler import TBDTraditionalCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recrawl_empty_sections")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recrawl traditional medicine records with empty sections (any file size)")
    parser.add_argument("--limit", type=int, default=50, help="Max files to recrawl in this run")
    parser.add_argument("--delay-min", type=float, default=0.5, help="Min delay between requests")
    parser.add_argument("--delay-max", type=float, default=1.5, help="Max delay between requests")
    parser.add_argument("--dry-run", action="store_true", help="Only count candidates, don't crawl")
    args = parser.parse_args()

    dir_path = Path("data/raw/traditional")
    if not dir_path.exists():
        logger.error("Directory %s does not exist!", dir_path)
        return

    # Find ALL files with empty sections, regardless of file size
    candidates = []
    total_files = 0
    for f in dir_path.glob("*.json"):
        total_files += 1
        try:
            with open(f, "r", encoding="utf-8") as file:
                data = json.load(file)
                sections = data.get("sections", {})
                url = data.get("source_url")
                if not sections and url:
                    candidates.append((f, url))
        except Exception as exc:
            logger.warning("Error inspecting file %s: %s", f, exc)

    logger.info("Scanned %d files. Found %d with empty sections.", total_files, len(candidates))
    
    if args.dry_run:
        logger.info("Dry run mode. Exiting.")
        return
    
    if not candidates:
        logger.info("No candidates found. All files have sections!")
        return

    to_crawl = candidates[:args.limit]
    logger.info("Will process %d files in this run.", len(to_crawl))

    crawler = TBDTraditionalCrawler()
    success_count = 0
    gained_sections = 0

    for idx, (file_path, url) in enumerate(to_crawl, 1):
        logger.info("[%d/%d] Recrawling %s", idx, len(to_crawl), file_path.name)
        try:
            drug = crawler.scrape_drug_detail(url)
            if drug:
                has_sections = bool(drug.get("sections"))
                section_keys = list(drug.get("sections", {}).keys())
                logger.info("  -> sections: %s, ingredients: %d", section_keys, len(drug.get("herbal_ingredients", [])))
                
                if has_sections:
                    gained_sections += 1
                
                with open(file_path, "w", encoding="utf-8") as f_out:
                    json.dump(drug, f_out, ensure_ascii=False, indent=2)
                success_count += 1
            else:
                logger.warning("  -> Failed to scrape detail for %s", url)
        except Exception as exc:
            logger.error("  -> Error processing %s: %s", url, exc)
        
        delay = random.uniform(args.delay_min, args.delay_max)
        time.sleep(delay)

    logger.info("Finished. Updated %d/%d files. %d files gained new sections.", 
                success_count, len(to_crawl), gained_sections)

if __name__ == "__main__":
    main()
