"""
TBD (thuocbietduoc.com.vn) Crawler — Western Medicine (N06)
Scrapes Category N06: Thuốc trị ký sinh trùng, chống nhiễm khuẩn, kháng virus, kháng nấm

Based on real HTML investigation (2026-06-17):
- Category URL: https://thuocbietduoc.com.vn/nhom-thuoc-6-0/thuoc-tri-ky-sinh-trung-chong-nhiem-khuan-khang-virus-khang-nam.aspx
- Pagination:   ?page=N (starting from page 2)
- Drug detail:  https://thuocbietduoc.com.vn/thuoc-{ID}/{SLUG}.aspx
- Active ingredient table: div#thanh-phan-hoat-chat
- Clinical sections: h2#section-N → content until next h2
"""

import re
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TBDWesternCrawler:
    """
    Crawler for thuocbietduoc.com.vn Category N06 (Western/anti-infective drugs).
    Produces structured drug records matching the canonical Western Medicine schema.
    """

    BASE_URL = "https://thuocbietduoc.com.vn"
    CATEGORY_URL = (
        "https://thuocbietduoc.com.vn/nhom-thuoc-6-0/"
        "thuoc-tri-ky-sinh-trung-chong-nhiem-khuan-khang-virus-khang-nam.aspx"
    )
    SECTION_MAP = {
        "section-1": "indication",
        "section-2": "contraindication",
        "section-3": "dosage",
        "section-4": "side_effects",
        "section-5": "interactions",
        "section-6": "warnings",
        "section-7": "pharmacology",
        "section-8": "pharmacokinetics",
    }

    def __init__(self, delay: float = 1.2, output_dir: Optional[str] = None):
        self.delay = delay
        self.output_dir = Path(output_dir) if output_dir else Path("data/raw/western")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch URL and return BeautifulSoup, or None on error."""
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.content, "html.parser")
        except Exception as exc:
            logger.warning("[TBD-N06] Failed to fetch %s: %s", url, exc)
            return None

    def _text(self, el) -> str:
        """Return stripped text of a BS4 element, or empty string."""
        return el.get_text(separator=" ", strip=True) if el else ""

    # ------------------------------------------------------------------
    # Step 1: collect drug links from category list pages
    # ------------------------------------------------------------------

    def scrape_drug_links(self, page_num: int) -> List[str]:
        """
        Fetch a category list page and return all drug detail page URLs.
        """
        url = self.CATEGORY_URL if page_num == 1 else f"{self.CATEGORY_URL}?page={page_num}"
        soup = self._get(url)
        if not soup:
            return []

        links = []
        pattern = re.compile(r"^/thuoc-\d+/[^/]+\.aspx$")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            path = href.replace(self.BASE_URL, "") if href.startswith("http") else href
            if pattern.match(path):
                full = f"{self.BASE_URL}{path}"
                if full not in links:
                    links.append(full)

        logger.info("[TBD-N06] Page %d → %d links", page_num, len(links))
        return links

    # ------------------------------------------------------------------
    # Step 2: parse an individual drug detail page
    # ------------------------------------------------------------------

    def scrape_drug_detail(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Parse a Western medicine drug detail page into a canonical dict:
        {
            source_url, drug_type,
            name, registration_number, dosage_form, strength,
            manufacturer, manufacturer_country,
            active_ingredients: [{name, amount}],
            sections: {indication, contraindication, dosage, side_effects,
                       interactions, warnings, pharmacology, pharmacokinetics}
        }
        """
        soup = self._get(url)
        if not soup:
            return None

        drug: Dict[str, Any] = {
            "source_url": url,
            "drug_type": "WESTERN_MEDICINE",
        }

        # ── Drug name ──────────────────────────────────────────────────
        h1 = soup.find("h1")
        drug["name"] = self._text(h1) if h1 else "Chưa rõ"

        # ── Metadata fields ────────────────────────────────────────────
        drug["registration_number"] = None
        drug["dosage_form"] = None
        drug["strength"] = None

        for div in soup.find_all("div"):
            txt = div.get_text(strip=True).lower()
            if "số đăng ký" in txt and div.name == "div":
                sib = div.find_next_sibling("div")
                if sib and not drug["registration_number"]:
                    drug["registration_number"] = self._text(sib)
            elif "dạng bào chế" in txt and div.name == "div":
                sib = div.find_next_sibling("div")
                if sib and not drug["dosage_form"]:
                    drug["dosage_form"] = self._text(sib)
            elif "hàm lượng" in txt and div.name == "div":
                sib = div.find_next_sibling("div")
                if sib and not drug["strength"]:
                    drug["strength"] = self._text(sib)

        # Fallback: extract SDK from meta or page text
        if not drug["registration_number"]:
            for pattern in [
                r"(VD-\d{4,5}-\d{2})",
                r"(VN-\d{4,5}-\d{2})",
                r"(VNA-\d{4,5}-\d{2})",
            ]:
                m = re.search(pattern, soup.get_text(), re.IGNORECASE)
                if m:
                    drug["registration_number"] = m.group(1)
                    break

        # ── Manufacturer ───────────────────────────────────────────────
        mfr_link = soup.find("a", href=re.compile(r"^https://thuocbietduoc\.com\.vn/nha-san-xuat/"))
        if mfr_link:
            country_span = mfr_link.find("span")
            country = self._text(country_span).strip("- ") if country_span else None
            if country_span:
                country_span.extract()
            drug["manufacturer"] = self._text(mfr_link).strip()
            drug["manufacturer_country"] = country
        else:
            drug["manufacturer"] = None
            drug["manufacturer_country"] = None

        # ── Active ingredients ─────────────────────────────────────────
        # Same DOM structure as traditional: div#thanh-phan-hoat-chat → table rows
        active_ingredients = []
        ingr_div = soup.find("div", id="thanh-phan-hoat-chat")
        if ingr_div:
            for row in ingr_div.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 2:
                    name = self._text(cols[0])
                    amount = self._text(cols[1])
                    if name:
                        active_ingredients.append({
                            "name": name.strip(),
                            "amount": amount.strip() or None,
                            "is_main_active_ingredient": True,
                        })
        drug["active_ingredients"] = active_ingredients

        # ── Clinical sections ──────────────────────────────────────────
        sections: Dict[str, str] = {}
        for h2 in soup.find_all("h2", id=re.compile(r"^section-\d+$")):
            section_id = h2.get("id", "")
            key = self.SECTION_MAP.get(section_id)
            if not key:
                continue
            content_parts = []
            for sib in h2.next_siblings:
                if sib.name == "h2":
                    break
                text = self._text(sib)
                if text:
                    content_parts.append(text)
            if content_parts:
                sections[key] = " ".join(content_parts)
        drug["sections"] = sections

        return drug

    # ------------------------------------------------------------------
    # Step 3: orchestrate crawl + save
    # ------------------------------------------------------------------

    def run(self, max_pages: int = 5, save: bool = True) -> List[Dict[str, Any]]:
        """
        Crawl up to `max_pages` category pages, then scrape each drug detail.
        If `save=True`, writes each drug to JSON in self.output_dir.
        """
        results: List[Dict[str, Any]] = []
        logger.info("[TBD-N06] Starting crawl for up to %d pages.", max_pages)

        for page in range(1, max_pages + 1):
            links = self.scrape_drug_links(page)
            if not links:
                logger.info("[TBD-N06] No links on page %d, stopping.", page)
                break

            for link in links:
                logger.info("[TBD-N06] Scraping: %s", link)
                drug = self.scrape_drug_detail(link)
                if drug:
                    results.append(drug)
                    if save:
                        self._save(drug)
                time.sleep(self.delay)

        logger.info("[TBD-N06] Done. Collected %d drugs.", len(results))
        return results

    def _save(self, drug: Dict[str, Any]):
        """Persist a single drug dict as a JSON file."""
        slug = re.sub(r"[^\w\-]", "_", drug.get("name", "unknown"))[:60]
        reg = re.sub(r"[/\\]", "-", drug.get("registration_number") or "no-sdk")
        filename = self.output_dir / f"{reg}_{slug}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(drug, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    crawler = TBDWesternCrawler()
    data = crawler.run(max_pages=1, save=False)
    if data:
        first = data[0]
        print(f"Drug        : {first['name']}")
        print(f"SDK         : {first['registration_number']}")
        print(f"Manufacturer: {first['manufacturer']}")
        print(f"Ingredients : {len(first['active_ingredients'])} hoạt chất")
        print(f"Sections    : {list(first['sections'].keys())}")
    else:
        print("No drugs scraped — check URL or network.")
