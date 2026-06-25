"""
TBD (thuocbietduoc.com.vn) Crawler — Traditional Medicine (N29)
Scrapes Category N29: Thuốc có nguồn gốc thảo dược, động vật

Based on real HTML investigation (2026-06-17):
- Category URL: https://thuocbietduoc.com.vn/nhom-thuoc-29-0/thuoc-co-nguon-goc-thao-duoc-dong-vat.aspx
- Pagination:   ?page=N (starting from page 2)
- Drug detail:  https://thuocbietduoc.com.vn/thuoc-{ID}/{SLUG}.aspx
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


class TBDTraditionalCrawler:
    """
    Crawler for thuocbietduoc.com.vn Category N29 (Thuốc có nguồn gốc thảo dược, động vật).
    Produces structured drug records matching the canonical schema for Traditional Medicine.
    """

    BASE_URL = "https://thuocbietduoc.com.vn"
    CATEGORY_URL = "https://thuocbietduoc.com.vn/nhom-thuoc-29-0/thuoc-co-nguon-goc-thao-duoc-dong-vat.aspx"
    # Section id→canonical key mapping (h2 with id="section-N")
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
        self.output_dir = Path(output_dir) if output_dir else Path("data/raw/traditional")
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
            logger.warning("[TBD-N29] Failed to fetch %s: %s", url, exc)
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
        Page 1 has no ?page= param; pages ≥2 use ?page=N.
        """
        url = self.CATEGORY_URL if page_num == 1 else f"{self.CATEGORY_URL}?page={page_num}"
        soup = self._get(url)
        if not soup:
            return []

        links = []
        # Drug detail hrefs follow: /thuoc-{ID}/{SLUG}.aspx
        pattern = re.compile(r"^/thuoc-\d+/[^/]+\.aspx$")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Handle both relative and absolute hrefs
            if href.startswith("http"):
                path = href.replace(self.BASE_URL, "")
            else:
                path = href
            if pattern.match(path):
                full = f"{self.BASE_URL}{path}"
                if full not in links:
                    links.append(full)

        logger.info("[TBD-N29] Page %d → %d links", page_num, len(links))
        return links

    # ------------------------------------------------------------------
    # Step 2: parse an individual drug detail page
    # ------------------------------------------------------------------

    def scrape_drug_detail(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Parse a drug detail page into a canonical dict:
        {
            source_url, drug_type,
            name, registration_number, dosage_form, manufacturer,
            herbal_ingredients: [{name, amount, link}],
            sections: {indication, contraindication, dosage, side_effects,
                       interactions, warnings, pharmacology, pharmacokinetics}
        }
        """
        soup = self._get(url)
        if not soup:
            return None

        drug: Dict[str, Any] = {
            "source_url": url,
            "drug_type": "TRADITIONAL_MEDICINE",
        }

        # ── Drug name ──────────────────────────────────────────────────
        h1 = soup.find("h1")
        drug["name"] = self._text(h1) if h1 else "Chưa rõ"

        # ── Metadata quick-info cards ──────────────────────────────────
        # The site renders label/value pairs as two adjacent divs where the
        # label div contains a small <p> or <span> with the field name,
        # and the value is in a sibling div with class "font-semibold".
        drug["registration_number"] = None
        drug["dosage_form"] = None

        for label_div in soup.find_all("div"):
            text = label_div.get_text(strip=True).lower()
            if "số đăng ký" in text and label_div.name == "div":
                # Value is usually the next sibling or inside a nearby span/div
                sib = label_div.find_next_sibling("div")
                if sib:
                    drug["registration_number"] = self._text(sib)
            elif "dạng bào chế" in text and label_div.name == "div":
                sib = label_div.find_next_sibling("div")
                if sib:
                    drug["dosage_form"] = self._text(sib)

        # Fallback: scan meta description or title for registration number
        if not drug["registration_number"]:
            meta_desc = soup.find("meta", {"name": "description"})
            if meta_desc:
                m = re.search(r"(V\d+-H\d+-\d{2}|VNB-\d+-\d{2}|VND-\d+-\d{2}|VCT-\d+-\d{2}|TCT-\d+-\d{2})", meta_desc.get("content", ""), re.IGNORECASE)
                if m:
                    drug["registration_number"] = m.group(1)

        # ── Manufacturer ───────────────────────────────────────────────
        mfr_link = soup.find("a", href=re.compile(r"^https://thuocbietduoc\.com\.vn/nha-san-xuat/"))
        if mfr_link:
            # Country is in an inner <span>; remove it to get the company name
            country_span = mfr_link.find("span")
            country = self._text(country_span).strip("- ") if country_span else None
            if country_span:
                country_span.extract()
            drug["manufacturer"] = self._text(mfr_link).strip()
            drug["manufacturer_country"] = country
        else:
            drug["manufacturer"] = None
            drug["manufacturer_country"] = None

        # ── Herbal ingredients ─────────────────────────────────────────
        # Located in div#thanh-phan-hoat-chat → table → tbody → tr
        ingredients = []
        ingr_div = soup.find("div", id="thanh-phan-hoat-chat")
        if ingr_div:
            for row in ingr_div.find_all("tr"):
                cols = row.find_all("td")
                if len(cols) >= 2:
                    ingr_link = cols[0].find("a", href=re.compile(r"/thuoc-goc-"))
                    name = self._text(ingr_link) if ingr_link else self._text(cols[0])
                    amount = self._text(cols[1])
                    herb_url = ingr_link["href"] if ingr_link and ingr_link.get("href") else None
                    if herb_url and not herb_url.startswith("http"):
                        herb_url = f"{self.BASE_URL}{herb_url}"
                    if name:
                        ingredients.append({
                            "name": name.strip(),
                            "amount": amount.strip() or None,
                            "role": "Thành phần",
                            "herb_url": herb_url,
                        })

        # Fallback: scan class "ingredient-content" if no structured table ingredient is found
        if not ingredients:
            ingr_content_div = soup.find("div", class_="ingredient-content")
            if ingr_content_div:
                text_content = ingr_content_div.get_text(separator="\n", strip=True)
                for line in text_content.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    # Remove starting "-", "*", "•", "1.", etc.
                    line_clean = re.sub(r"^[-\*\•\s\d\.\)]+", "", line).strip()
                    if not line_clean:
                        continue
                    # Split by dot sequence (e.g. ................) or colon/hyphen
                    parts = re.split(r"\.{2,}|:| - | – ", line_clean)
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        amount = parts[1].strip()
                    else:
                        name = line_clean
                        amount = None
                    if name:
                        # Clean name of trailing dots/spaces
                        name = name.rstrip(".").strip()
                        if amount:
                            amount = amount.lstrip(".").strip()
                        ingredients.append({
                            "name": name,
                            "amount": amount,
                            "role": "Thành phần",
                            "herb_url": None,
                        })
        drug["herbal_ingredients"] = ingredients

        # ── Clinical sections ──────────────────────────────────────────
        # Map of div ID to canonical section key
        DIV_ID_MAP = {
            "cong-dung-thuoc": "indication",
            "cong-dung": "indication",
            "chi-dinh": "indication",
            "doi-tuong-su-dung": "indication",
            "doi-tuong-dung": "indication",
            "chong-chi-dinh": "contraindication",
            "khong-dung-cho": "contraindication",
            "lieu-luong-cach-dung": "dosage",
            "lieu-luong": "dosage",
            "lieu-dung": "dosage",
            "cach-dung": "dosage",
            "cach-su-dung": "dosage",
            "huong-dan-su-dung": "dosage",
            "tac-dung-phu": "side_effects",
            "tac-dung-khong-mong-muon": "side_effects",
            "tuong-tac-thuoc": "interactions",
            "tuong-tac": "interactions",
            "than-trong": "warnings",
            "than-trong-luc-dung": "warnings",
            "than-trong-khi-dung": "warnings",
            "canh-bao": "warnings",
            "luu-y": "warnings",
            "chu-y": "warnings",
            "luu-y-khi-su-dung": "warnings",
            "duoc-luc": "pharmacology",
            "duoc-luc-hoc": "pharmacology",
            "tac-dung": "pharmacology",
            "tinh-vi": "pharmacology",
            "quy-kinh": "pharmacology",
            "duoc-dong-hoc": "pharmacokinetics",
            "hap-thu": "pharmacokinetics",
        }

        def classify_section_by_text(title: str) -> Optional[str]:
            t_lower = title.lower()
            if any(k in t_lower for k in ["chỉ định", "chi dinh", "công dụng", "cong dung", "đối tượng sử dụng", "doi tuong su dung", "đối tượng dùng", "khuyên dùng"]):
                return "indication"
            if any(k in t_lower for k in ["chống chỉ định", "chong chi dinh", "không dùng", "khong dung"]):
                return "contraindication"
            if any(k in t_lower for k in ["liều dùng", "lieu dung", "liều lượng", "lieu luong", "cách dùng", "cach dung", "cách sử dụng", "cach su dung", "hướng dẫn sử dụng"]):
                return "dosage"
            if any(k in t_lower for k in ["tác dụng phụ", "tac dung phu", "tác dụng không mong muốn", "khong mong muon", "tác dụng ngoại ý"]):
                return "side_effects"
            if any(k in t_lower for k in ["tương tác", "tuong tac"]):
                return "interactions"
            if any(k in t_lower for k in ["cảnh báo", "canh bao", "thận trọng", "than trong", "lưu ý", "luu y", "chú ý", "chu y"]):
                return "warnings"
            if any(k in t_lower for k in ["dược lực", "duoc luc", "tác dụng", "tac dung", "tính vị", "quy kinh"]):
                return "pharmacology"
            if any(k in t_lower for k in ["dược động", "duoc dong"]):
                return "pharmacokinetics"
            return None

        # Identify all potential section elements in the document
        clinical_elements = []
        
        # 1. Divs with specific IDs
        for div_id in DIV_ID_MAP.keys():
            for div in soup.find_all("div", id=div_id):
                if div not in clinical_elements:
                    clinical_elements.append(div)
                    
        # 2. Elements with id="section-N"
        for el in soup.find_all(id=re.compile(r"^section-\d+$")):
            if el not in clinical_elements:
                clinical_elements.append(el)
                
        # 3. Heading tags matching clinical keywords
        for tag_name in ["h2", "h3"]:
            for el in soup.find_all(tag_name):
                text = el.get_text(strip=True)
                if classify_section_by_text(text) and el not in clinical_elements:
                    clinical_elements.append(el)

        # Get DOM positions to sort them and partition drug vs ingredient info
        all_tags = soup.find_all(True)
        tag_positions = {tag: idx for idx, tag in enumerate(all_tags)}
        
        hoat_chat_div = soup.find(id="thong-tin-hoat-chat")
        hoat_chat_pos = tag_positions.get(hoat_chat_div, 999999) if hoat_chat_div else 999999
        
        # Determine main active ingredient name for labeling
        ingredient_name = ""
        if hoat_chat_div:
            h2_hc = hoat_chat_div.find("h2")
            if h2_hc:
                hc_text = h2_hc.get_text(strip=True)
                if "hoạt chất:" in hc_text.lower():
                    ingredient_name = hc_text.split("hoạt chất:")[-1].strip()
                    
        # Sort by DOM order (filter out items that might have been detached)
        clinical_elements = sorted([el for el in clinical_elements if el in tag_positions], key=lambda x: tag_positions[x])
        
        drug_clinical: Dict[str, List[str]] = {k: [] for k in self.SECTION_MAP.values()}
        ingredient_clinical: Dict[str, List[str]] = {k: [] for k in self.SECTION_MAP.values()}
        
        for el in clinical_elements:
            el_id = el.get("id", "")
            title_text = el.get_text(strip=True) if el.name in ["h1", "h2", "h3", "h4", "strong"] else ""
            
            # Determine canonical key
            key = None
            if el_id in DIV_ID_MAP:
                key = DIV_ID_MAP[el_id]
            else:
                key = classify_section_by_text(title_text)
                if not key and el_id.startswith("section-"):
                    key = self.SECTION_MAP.get(el_id)
            
            if not key:
                continue
                
            # Extract content
            content = ""
            if el.name == "div" and el_id in DIV_ID_MAP:
                prose = el.find("div", class_="prose")
                if prose:
                    content = self._text(prose)
                else:
                    title_el = el.find(["h1", "h2", "h3", "h4"])
                    t_text = self._text(title_el)
                    full_text = self._text(el)
                    if t_text and full_text.startswith(t_text):
                        content = full_text[len(t_text):].strip()
                    else:
                        content = full_text
            else:
                content_parts = []
                for sib in el.next_siblings:
                    if sib in clinical_elements:
                        break
                    if sib.name in ["h1", "h2", "h3", "h4"] or (sib.name == "div" and sib.get("id") in DIV_ID_MAP):
                        break
                    text = self._text(sib)
                    if text:
                        content_parts.append(text)
                content = " ".join(content_parts)
                
            content = content.strip()
            if not content:
                continue
                
            # Partition by position relative to ingredient boundary
            pos = tag_positions.get(el, 0)
            if pos < hoat_chat_pos:
                drug_clinical[key].append(content)
            else:
                ingredient_clinical[key].append(content)
                
        # Merge sections
        sections: Dict[str, str] = {}
        for key in self.SECTION_MAP.values():
            parts = []
            if drug_clinical[key]:
                parts.append(" ".join(drug_clinical[key]))
            if ingredient_clinical[key]:
                ingr_label = f"[Thông tin hoạt chất {ingredient_name}]" if ingredient_name else "[Thông tin hoạt chất]"
                parts.append(f"{ingr_label} " + " ".join(ingredient_clinical[key]))
            if parts:
                sections[key] = " \n".join(parts)
                
        drug["sections"] = sections

        return drug

    # ------------------------------------------------------------------
    # Step 3: orchestrate crawl + save
    # ------------------------------------------------------------------

    def run(self, max_pages: int = 5, save: bool = True) -> List[Dict[str, Any]]:
        """
        Crawl up to `max_pages` category pages, then scrape each drug detail.
        If `save=True`, writes each drug to JSON in self.output_dir.

        Returns the list of all scraped drug dicts.
        """
        results: List[Dict[str, Any]] = []
        logger.info("[TBD-N29] Starting crawl for up to %d pages.", max_pages)

        for page in range(1, max_pages + 1):
            links = self.scrape_drug_links(page)
            if not links:
                logger.info("[TBD-N29] No links on page %d, stopping.", page)
                break

            for link in links:
                logger.info("[TBD-N29] Scraping: %s", link)
                drug = self.scrape_drug_detail(link)
                if drug:
                    results.append(drug)
                    if save:
                        self._save(drug)
                time.sleep(self.delay)

        logger.info("[TBD-N29] Done. Collected %d drugs.", len(results))
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
    crawler = TBDTraditionalCrawler()
    data = crawler.run(max_pages=1, save=False)
    if data:
        first = data[0]
        print(f"Drug  : {first['name']}")
        print(f"SDK   : {first['registration_number']}")
        print(f"Mfr   : {first['manufacturer']}")
        print(f"Herbs : {len(first['herbal_ingredients'])} vị thuốc")
        print(f"Sections: {list(first['sections'].keys())}")
    else:
        print("No drugs scraped — check URL or network.")
