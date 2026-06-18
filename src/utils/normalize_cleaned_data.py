import json
import os
import re
import sys
import glob
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure we can import from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.drug import (
    Drug, DrugMetadata, DrugSections,
    ActiveIngredient, Manufacturer, Packaging
)

def clean_name(name: str) -> str:
    # Remove things like "dưới dạng ...", "dạng ...", "(dưới dạng ...)"
    name = re.sub(r'\s*\(\s*dưới dạng.*?\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*dưới dạng.*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(\s*as.*?\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(\s*equivalent to.*?\)', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s*\(\s*tương đương.*?\)', '', name, flags=re.IGNORECASE)
    
    # Remove any empty parentheses
    name = re.sub(r'\(\s*\)', '', name)
    # Remove leading/trailing punctuation and spaces
    name = name.strip(' ;,.*&()[]{}')
    return name

def parse_single_ingredient(text: str) -> Tuple[str, str]:
    text = text.strip()
    if not text or text in ['-', '_', '.', ',', ';', 'None', 'none']:
        return "", ""
        
    # Standardize brackets
    text = text.replace('[', '(').replace(']', ')')
    
    # Regex to extract strength
    # Match patterns like: 500mg, 500 mg, 200mg/5ml, 750000IU, 0.5g, 10%, 1 MIU, etc.
    strength_pattern = r'(\d+(?:\.\d+)?\s*(?:mg|g|mcg|ug|IU|UI|MIU|ml|%)(?:\s*/\s*\d*\s*(?:ml|g|giọt|chai|lọ|vỉ|gói|viên))?)'
    
    # Find all strengths in the string
    matches = list(re.finditer(strength_pattern, text, re.IGNORECASE))
    
    strength = ""
    name_part = text
    
    if matches:
        # Take the last strength match
        match = matches[-1]
        strength = match.group(1).strip()
        # Remove the strength from the name part
        start, end = match.span()
        name_part = text[:start] + text[end:]
        
    name_part = name_part.strip()
    
    # Extract nested names/synonyms in parentheses
    paren_match = re.search(r'^([^(]+)\s*\(([^)]+)\)$', name_part)
    if paren_match:
        name_a = paren_match.group(1).strip()
        name_b = paren_match.group(2).strip()
        
        # Check if name_b is just salt info
        is_salt = any(indicator in name_b.lower() for indicator in ["dưới dạng", "dạng", "as", "equivalent", "tương đương"])
        
        if is_salt:
            name = clean_name(name_a)
        else:
            cleaned_a = clean_name(name_a)
            cleaned_b = clean_name(name_b)
            # Use the longer/more specific name if it contains the other
            if cleaned_a.lower() in cleaned_b.lower():
                name = cleaned_b
            elif cleaned_b.lower() in cleaned_a.lower():
                name = cleaned_a
            else:
                name = cleaned_b
    else:
        name = clean_name(name_part)
        
    # Clean strength formatting
    if strength:
        strength = re.sub(r'\s+', '', strength).lower()
        
    # Final check: if the name is just a dash or empty, discard
    if name in ['-', '', '_', '.', ',', ';']:
        return "", ""
        
    return name, strength

def split_ingredients(text: str) -> List[str]:
    if not text:
        return []
    parts = []
    current_part = []
    paren_depth = 0
    
    for char in text:
        if char in '([{':
            paren_depth += 1
        elif char in ')]}':
            paren_depth -= 1
        
        if (char in ';' or (char == ',' and ';' not in text)) and paren_depth == 0:
            parts.append("".join(current_part))
            current_part = []
        else:
            current_part.append(char)
            
    if current_part:
        parts.append("".join(current_part))
        
    cleaned_parts = []
    for p in parts:
        p = p.strip()
        if p:
            if re.search(r'\s+(?:và|&)\s+', p, re.IGNORECASE) and '(' not in p:
                subparts = re.split(r'\s+(?:và|&)\s+', p, flags=re.IGNORECASE)
                for sp in subparts:
                    sp = sp.strip()
                    if sp:
                        cleaned_parts.append(sp)
            else:
                cleaned_parts.append(p)
            
    return cleaned_parts

def choose_better_strength(s1: str, s2: str) -> str:
    if not s1:
        return s2
    if not s2:
        return s1
    if '/' in s1 and '/' not in s2:
        return s1
    if '/' in s2 and '/' not in s1:
        return s2
    return s1 if len(s1) >= len(s2) else s2

def parse_ingredients_from_fields(ingredients_str: str, active_ingredients_str: str) -> List[Dict[str, str]]:
    results = {}
    
    # Process Ingredients string
    parts_ing = split_ingredients(ingredients_str)
    for p in parts_ing:
        name, strength = parse_single_ingredient(p)
        if name:
            results[name.lower()] = {"name": name, "strength": strength}
            
    # Process Active Ingredients string
    parts_act = split_ingredients(active_ingredients_str)
    for p in parts_act:
        name, strength = parse_single_ingredient(p)
        if name:
            key = name.lower()
            if key in results:
                # Merge strengths choosing the better one
                results[key]["strength"] = choose_better_strength(results[key]["strength"], strength)
            else:
                # Synonym check
                found = False
                for existing_key, val in results.items():
                    if (key in existing_key) or (existing_key in key) or \
                       (key.replace('acid ', '') in existing_key) or (existing_key.replace('acid ', '') in key):
                        found = True
                        val["strength"] = choose_better_strength(val["strength"], strength)
                        break
                if not found:
                    results[key] = {"name": name, "strength": strength}
                
    return list(results.values())

def normalize_teammate_drug(raw: Dict[str, Any]) -> Dict[str, Any]:
    # 1. Parse active ingredients
    ingredients_str = raw.get("Ingredients", "") or ""
    active_ingredients_str = raw.get("Active Ingredients", "") or ""
    
    parsed_ingredients = parse_ingredients_from_fields(ingredients_str, active_ingredients_str)
    
    active_ingredients = [
        ActiveIngredient(name=item["name"], is_main_active_ingredient=True)
        for item in parsed_ingredients
    ]
    
    # Combined strength string
    strength_str = "/".join(item["strength"] for item in parsed_ingredients if item["strength"])
    
    # 2. Parse Manufacturer name & country
    m_str = raw.get("Manufacturer", "") or ""
    m_name = m_str
    m_country = None
    if " - " in m_str:
        parts = m_str.rsplit(" - ", 1)
        m_name = parts[0].strip()
        m_country = parts[1].strip()
        
    # 3. Packaging
    packagings = []
    p_str = raw.get("Packaging", "") or ""
    if p_str:
        packagings.append(Packaging(unit_name=p_str))
        
    # 4. Drug Pydantic Model
    drug = Drug(
        metadata=DrugMetadata(
            name=raw.get("Drug Name", "UNKNOWN"),
            registration_number=raw.get("Registration Number") or "UNKNOWN",
            drug_type="WESTERN_MEDICINE",
            active_ingredient_list=active_ingredients,
            strength=strength_str or None,
            manufacturer=Manufacturer(name=m_name or "Chưa rõ", country=m_country),
            packagings=packagings,
            registrant=raw.get("Registrant") or None,
        ),
        sections=DrugSections(
            indication=raw.get("Indications") or None,
            contraindication=raw.get("Contraindications") or None,
            dosage=raw.get("Dosage and Administration") or None,
            side_effects=raw.get("Side Effects") or None,
            interactions=raw.get("Drug Interactions") or None,
        )
    )
    # Return as dict
    return drug.model_dump()

def main():
    dataset_dir = Path("d:/Study/University/SE365/PROJECT/SE365-RAG-PHARMA/src/database/Datasets/thuoc/cleaned_json")
    json_files = list(dataset_dir.glob("*.json"))
    
    print(f"Normalizing {len(json_files)} teammate JSON files...")
    
    success_count = 0
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                drugs = json.load(f)
                
            normalized_list = []
            for d in drugs:
                normalized_list.append(normalize_teammate_drug(d))
                
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(normalized_list, f, indent=4, ensure_ascii=False)
                
            success_count += 1
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")
            
    print(f"Successfully normalized {success_count}/{len(json_files)} files!")

if __name__ == "__main__":
    main()
