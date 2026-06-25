import os
import json
import time
import random
import logging
from pathlib import Path
import sys

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent))

from crawler.tbd_crawler import TBDTraditionalCrawler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("recrawl_traditional")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Recrawl traditional medicine records with empty/corrupted sections")
    parser.add_argument("--limit", type=int, default=20, help="Max files to recrawl in this run")
    parser.add_argument("--size-limit", type=int, default=600, help="File size threshold in bytes to consider empty/corrupted")
    parser.add_argument("--delay-min", type=float, default=1.0, help="Min delay between requests")
    parser.add_argument("--delay-max", type=float, default=2.5, help="Max delay between requests")
    args = parser.parse_args()

    dir_path = Path("data/raw/traditional")
    if not dir_path.exists():
        logger.error("Directory %s does not exist!", dir_path)
        return

    # Find candidates
    candidates = []
    for f in dir_path.glob("*.json"):
        try:
            if os.path.getsize(f) < args.size_limit:
                # Open and read source_url
                with open(f, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    url = data.get("source_url")
                    if url:
                        candidates.append((f, url))
        except Exception as exc:
            logger.warning("Error inspecting file %s: %s", f, exc)

    logger.info("Found %d candidate files for recrawling.", len(candidates))
    if not candidates:
        logger.info("No candidates found. All files are healthy!")
        return

    # Limit the run
    to_crawl = candidates[:args.limit]
    logger.info("Will process %d files in this run.", len(to_crawl))

    crawler = TBDTraditionalCrawler()
    success_count = 0

    for idx, (file_path, url) in enumerate(to_crawl, 1):
        logger.info("[%d/%d] Recrawling %s from %s", idx, len(to_crawl), file_path.name, url)
        try:
            drug = crawler.scrape_drug_detail(url)
            if drug:
                # Check if we got sections
                has_sections = bool(drug.get("sections"))
                logger.info("  -> Got sections: %s, ingredients: %d", list(drug["sections"].keys()), len(drug["herbal_ingredients"]))
                
                # Overwrite the old file
                with open(file_path, "w", encoding="utf-8") as f_out:
                    json.dump(drug, f_out, ensure_ascii=False, indent=2)
                success_count += 1
            else:
                logger.warning("  -> Failed to scrape detail for %s", url)
        except Exception as exc:
            logger.error("  -> Error processing %s: %s", url, exc)
        
        # Sleep to avoid block
        delay = random.uniform(args.delay_min, args.delay_max)
        time.sleep(delay)

    logger.info("Finished. Successfully updated %d/%d files.", success_count, len(to_crawl))

if __name__ == "__main__":
    main()
