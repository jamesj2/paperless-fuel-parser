#!/usr/bin/env python3
"""
Gas Station Receipt Parser for Paperless-ngx

Extracts Gallons, Total, and Address from gas station receipts
and updates custom fields in Paperless-ngx.

Supports:
  - LLM extraction via Gemini or Claude (configurable)
  - Regex fallback when LLM is unavailable or fails
  - Batch mode: process all receipts with empty fields
  - Single mode: process one document by ID (for post-consumption hooks)

Usage:
  Batch mode:   python gas_receipt_parser.py
  Single doc:   python gas_receipt_parser.py --doc-id 123
  Dry run:      python gas_receipt_parser.py --dry-run
"""

import argparse
import atexit
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path

# Add local vendor directory to path (for containerized installs)
_vendor_dir = Path(__file__).parent / "vendor"
if _vendor_dir.is_dir():
    sys.path.insert(0, str(_vendor_dir))

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).parent / ".env")

PAPERLESS_URL = os.getenv("PAPERLESS_URL", "http://localhost:8000").rstrip("/")
PAPERLESS_TOKEN = os.getenv("PAPERLESS_API_TOKEN", "")
DOCTYPE_NAME = os.getenv("PAPERLESS_DOCTYPE_NAME", "Car: Fuel Receipt")

FIELD_GALLONS = os.getenv("FIELD_GALLONS", "Gallons")
FIELD_TOTAL = os.getenv("FIELD_TOTAL", "Total")
FIELD_ADDRESS = os.getenv("FIELD_ADDRESS", "Address")
FIELD_PPG = os.getenv("FIELD_PPG", "Price Per Gallon")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6-20250514")

REGEX_ONLY = os.getenv("REGEX_ONLY", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Retry settings for rate-limited (429) LLM requests
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "10"))  # seconds

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("gas_receipt_parser")

# ---------------------------------------------------------------------------
# Paperless-ngx API helpers
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Token {PAPERLESS_TOKEN}"})


def api(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{PAPERLESS_URL}/api/{path.lstrip('/')}"
    resp = SESSION.request(method, url, **kwargs)
    resp.raise_for_status()
    return resp


def get_all_pages(path: str, params: dict | None = None) -> list[dict]:
    """Fetch all pages from a paginated Paperless API endpoint."""
    results = []
    params = dict(params or {})
    params.setdefault("page_size", 100)
    while True:
        data = api("GET", path, params=params).json()
        results.extend(data.get("results", []))
        if not data.get("next"):
            break
        params["page"] = params.get("page", 1) + 1
    return results


def get_document_type_id(name: str) -> int | None:
    """Look up a document type ID by name."""
    types = get_all_pages("document_types/")
    for dt in types:
        if dt["name"].lower() == name.lower():
            return dt["id"]
    return None


def get_custom_fields() -> dict[str, dict]:
    """Return a dict mapping field name -> {id, data_type}."""
    fields = get_all_pages("custom_fields/")
    return {f["name"]: {"id": f["id"], "data_type": f["data_type"]} for f in fields}


def get_documents_by_type(doc_type_id: int) -> list[dict]:
    """Fetch all documents with the given document type."""
    return get_all_pages("documents/", params={"document_type__id": doc_type_id})


def get_document(doc_id: int) -> dict:
    """Fetch a single document by ID."""
    return api("GET", f"documents/{doc_id}/").json()


def get_document_content(doc_id: int) -> str:
    """Get the OCR text content of a document."""
    doc = get_document(doc_id)
    return doc.get("content", "")


def update_custom_fields(doc_id: int, field_updates: list[dict]) -> None:
    """
    Update custom fields on a document.
    field_updates: list of {"field": <field_id>, "value": <value>}
    """
    doc = get_document(doc_id)
    existing = doc.get("custom_fields", [])

    existing_by_id = {cf["field"]: cf for cf in existing}
    for update in field_updates:
        existing_by_id[update["field"]] = update

    api(
        "PATCH",
        f"documents/{doc_id}/",
        json={"custom_fields": list(existing_by_id.values())},
    )


def get_correspondents() -> dict[str, int]:
    """Return a dict mapping lowercase correspondent name -> id."""
    corrs = get_all_pages("correspondents/")
    return {c["name"].lower(): c["id"] for c in corrs}


def _normalize(s: str) -> str:
    """Normalize a string for fuzzy comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s#]", "", s)  # keep alphanumeric, spaces, #
    s = re.sub(r"\s+", " ", s)            # collapse whitespace
    return s


def _similarity(a: str, b: str) -> float:
    """Simple similarity ratio between two strings (0.0 - 1.0)."""
    a, b = _normalize(a), _normalize(b)
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Check if one contains the other
    if a in b or b in a:
        return 0.9
    # Character-level similarity (Sørensen–Dice coefficient on bigrams)
    def bigrams(s):
        return {s[i : i + 2] for i in range(len(s) - 1)}
    bg_a, bg_b = bigrams(a), bigrams(b)
    if not bg_a or not bg_b:
        return 0.0
    return 2 * len(bg_a & bg_b) / (len(bg_a) + len(bg_b))


def find_correspondent(name: str, corr_map: dict[str, int]) -> int | None:
    """
    Find a correspondent by name with fuzzy matching.
    Returns the ID of the best match, or None if no good match found.
    """
    # Exact match first (case-insensitive)
    normalized = name.strip().lower()
    if normalized in corr_map:
        return corr_map[normalized]

    # Fuzzy match — find the best candidate above threshold
    best_score = 0.0
    best_id = None
    for corr_name, corr_id in corr_map.items():
        score = _similarity(name, corr_name)
        if score > best_score:
            best_score = score
            best_id = corr_id

    if best_score >= 0.7:
        log.info("  Fuzzy matched correspondent '%s' (score=%.2f)", name, best_score)
        return best_id

    return None


def create_correspondent(name: str) -> int:
    """Create a new correspondent and return its ID."""
    resp = api("POST", "correspondents/", json={"name": name})
    return resp.json()["id"]


def set_document_correspondent(doc_id: int, correspondent_id: int) -> None:
    """Set the correspondent on a document."""
    api("PATCH", f"documents/{doc_id}/", json={"correspondent": correspondent_id})


# ---------------------------------------------------------------------------
# Regex extraction (fallback)
# ---------------------------------------------------------------------------

def extract_gallons_regex(text: str) -> str | None:
    """Extract gallon amount from receipt text."""
    patterns = [
        # "12.345 GAL" or "12.345 GALLONS"
        r"(\d{1,3}\.\d{2,4})\s*(?:GAL(?:LONS?)?)",
        # "GALLONS: 12.345" or "GAL: 12.345"
        r"(?:GAL(?:LONS?)?)\s*[:\s]\s*(\d{1,3}\.\d{2,4})",
        # "VOLUME: 12.345"
        r"VOLUME\s*[:\s]\s*(\d{1,3}\.\d{2,4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_total_regex(text: str) -> str | None:
    """Extract total dollar amount from receipt text."""
    patterns = [
        # "TOTAL $ 45.67" or "TOTAL: $45.67"
        r"TOTAL\s*[:\s]*\$?\s*(\d{1,4}\.\d{2})",
        # "FUEL SALE $ 45.67"
        r"FUEL\s+SALE\s*[:\s]*\$?\s*(\d{1,4}\.\d{2})",
        # "AMOUNT: $45.67"
        r"AMOUNT\s*[:\s]*\$?\s*(\d{1,4}\.\d{2})",
        # "SALE $ 45.67"
        r"SALE\s*[:\s]*\$?\s*(\d{1,4}\.\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_ppg_regex(text: str) -> str | None:
    """Extract price per gallon from receipt text."""
    patterns = [
        # "PRICE/G: $3.199" or "PRICE/G $3.199" (common on FAS-TRIP)
        # Also handles OCR space in "$3 .299" -> "$3.299"
        r"PRICE\s*/\s*[GQ]\s*[:\s]*\$?\s*(\d{1,2})\s*\.(\d{2,4})",
        # "PRICE/GAL: $3.199"
        r"PRICE\s*/\s*GAL(?:LON)?\s*[:\s]*\$?\s*(\d{1,2})\s*\.(\d{2,4})",
        # "PPG: $3.199" or "PPG $3.199"
        r"PPG\s*[:\s]*\$?\s*(\d{1,2})\s*\.(\d{2,4})",
        # "@ $3.199/G" or "@ 3.199/GAL"
        r"@\s*\$?\s*(\d{1,2})\s*\.(\d{2,4})\s*/\s*G(?:AL)?",
        # "$3.199/GAL"
        r"\$(\d{1,2})\s*\.(\d{2,4})\s*/\s*GAL(?:LON)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}.{match.group(2)}"

    return None


def extract_station_regex(text: str) -> str | None:
    """
    Extract the gas station name from receipt text.
    Station names typically appear in the first few non-empty lines,
    before the address and date.
    """
    lines = text.strip().split("\n")

    # Skip common noise lines and look for a plausible station name
    skip_patterns = re.compile(
        r"^("
        r"welcome\s+to|"       # "Welcome to Shell" -> we want "Shell"
        r"date\b|time\b|"      # date/time lines
        r"pump\s*#|"           # pump number
        r"service\s+level|"    # service level
        r"product\s*:|"        # product type
        r"gallons|price|"      # data fields
        r"total|sale|amount|"  # totals
        r"visa|mastercard|"    # payment
        r"credit|debit|"       # payment
        r"auth\s*#|"           # auth codes
        r"\d{5}|"              # zip codes
        r"\(\d{3}\)\s*\d{3}|"  # phone numbers
        r"\d{1,2}/\d{1,2}/\d"  # dates
        r")",
        re.IGNORECASE,
    )

    # "Welcome to Shell" -> extract "Shell"
    welcome_pattern = re.compile(r"welcome\s+to\s+(.+)", re.IGNORECASE)

    # Address line (digits at start = likely an address, not a name)
    address_pattern = re.compile(r"^\d{2,6}\s+")

    for line in lines[:10]:
        line = line.strip()
        if not line or len(line) < 3:
            continue

        # Check for "Welcome to <name>"
        wm = welcome_pattern.match(line)
        if wm:
            return wm.group(1).strip()

        # Skip lines that look like addresses, dates, fields, etc.
        if address_pattern.match(line):
            continue
        if skip_patterns.search(line):
            continue
        # Skip lines that are just numbers or codes
        if re.match(r"^[\d\s\-#]+$", line):
            continue

        # This is likely the station name
        return line.strip()

    return None


def _clean_ocr_line(line: str) -> str:
    """Remove common OCR artifacts from a line."""
    # Strip stray $, #, *, and trailing/leading whitespace
    return re.sub(r"[\$#\*]+", "", line).strip()


def extract_address_regex(text: str) -> str | None:
    """
    Extract a street address from the receipt text.
    Looks for common patterns like "123 Main St" near the top of the receipt.
    Handles addresses split across 2-3 lines (street / city+state / zip).
    """
    lines = text.strip().split("\n")
    # Addresses usually appear in the first ~15 lines of a receipt
    header_lines = [_clean_ocr_line(l) for l in lines[:15]]

    street_suffixes = (
        r"(?:ST|STREET|AVE|AVENUE|BLVD|BOULEVARD|DR|DRIVE|RD|ROAD"
        r"|LN|LANE|WAY|CT|COURT|PL|PLACE|HWY|HIGHWAY|PIKE|PKWY|PARKWAY)"
    )

    address_pattern = re.compile(
        r"(\d{1,6}\s+"                # street number
        r"(?:[NSEW]\.?\s+)?"          # optional N/S/E/W prefix
        r"[A-Za-z0-9\s\.]+?"          # street name
        + street_suffixes +
        r"\.?"
        r"(?:\s+[A-Za-z0-9]+)*"       # trailing tokens (e.g. "46", "STE 5")
        r")",
        re.IGNORECASE,
    )

    # City + State + optional Zip on the SAME line
    city_state_zip = re.compile(
        r"([A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?)", re.IGNORECASE
    )
    # City + State WITHOUT zip (zip may be on the next line)
    city_state_only = re.compile(
        r"^([A-Za-z][A-Za-z\s]*,?\s*[A-Z]{2})\s*$", re.IGNORECASE
    )
    # Standalone zip code
    zip_only = re.compile(r"^(\d{5}(?:-\d{4})?)\s*$")

    for i, line in enumerate(header_lines):
        match = address_pattern.search(line)
        if match:
            address = match.group(1).strip()

            # Look ahead for city/state/zip across the next 1-2 lines
            if i + 1 < len(header_lines):
                next_line = header_lines[i + 1]

                # Case 1: next line has city + state + zip all together
                csz = city_state_zip.search(next_line)
                if csz:
                    address += ", " + csz.group(1).strip()
                    return address

                # Case 2: next line has city + state, line after has zip
                cs = city_state_only.match(next_line)
                if cs:
                    address += ", " + cs.group(1).strip()
                    if i + 2 < len(header_lines):
                        zp = zip_only.match(header_lines[i + 2])
                        if zp:
                            address += " " + zp.group(1)
                    return address

            return address

    # Fallback: look for city/state/zip alone
    for line in header_lines:
        csz = city_state_zip.search(line)
        if csz:
            return csz.group(1).strip()

    return None


def extract_with_regex(text: str) -> dict:
    """Extract all fields using regex. Returns dict with found values."""
    return {
        "gallons": extract_gallons_regex(text),
        "total": extract_total_regex(text),
        "address": extract_address_regex(text),
        "ppg": extract_ppg_regex(text),
        "station": extract_station_regex(text),
    }


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

LLM_PROMPT = """\
You are extracting structured data from a gas station receipt.

From the receipt text below, extract:
1. **Gallons**: The number of gallons of fuel purchased (numeric, e.g. "12.345")
2. **Total**: The total dollar amount paid for fuel (numeric, e.g. "45.67")
3. **Address**: The street address of the gas station (e.g. "123 Main St, Springfield, IL 62704")
4. **Price Per Gallon**: The price per gallon of fuel (numeric, e.g. "3.199")
5. **Station**: The name of the gas station (e.g. "Shell", "QuikTrip #80656")

Rules:
- For Gallons, return ONLY the numeric value (no units).
- For Total, return ONLY the numeric value (no dollar sign).
- For Address, include street, city, state, and zip if available.
- For Price Per Gallon (ppg), return ONLY the numeric value (no dollar sign).
  It is often labeled PRICE/G, PRICE/GAL, or PPG on receipts.
  OCR may misread it (e.g. "fo." instead of "$0." or "/Q" instead of "/G").
- For Station, return the gas station brand/name, including store number if present.
  It usually appears at the very top of the receipt. "Welcome to X" means the station is "X".
- If a field cannot be determined, return null for that field.
- Return ONLY valid JSON, no other text.

Return your answer as JSON:
{
  "gallons": "12.345",
  "total": "45.67",
  "address": "123 Main St, Springfield, IL 62704",
  "ppg": "3.199",
  "station": "FAS-TRIP #107"
}

Receipt text:
---
%s
---"""


def extract_with_gemini(text: str) -> dict | None:
    """Extract fields using Google Gemini API with retry on rate limits."""
    from google import genai
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=LLM_PROMPT % text,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            return _parse_llm_json(response.text)
        except genai_errors.ClientError as e:
            if e.code == 429 and attempt < LLM_MAX_RETRIES:
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "Gemini rate limited (attempt %d/%d), retrying in %.0fs...",
                    attempt, LLM_MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                log.exception("Gemini extraction failed")
                return None
        except Exception:
            log.exception("Gemini extraction failed")
            return None
    return None


def extract_with_claude(text: str) -> dict | None:
    """Extract fields using Anthropic Claude API with retry on rate limits."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": LLM_PROMPT % text}],
            )
            return _parse_llm_json(message.content[0].text)
        except anthropic.RateLimitError:
            if attempt < LLM_MAX_RETRIES:
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                log.warning(
                    "Claude rate limited (attempt %d/%d), retrying in %.0fs...",
                    attempt, LLM_MAX_RETRIES, delay,
                )
                time.sleep(delay)
            else:
                log.error("Claude rate limited, all %d retries exhausted", LLM_MAX_RETRIES)
                return None
        except Exception:
            log.exception("Claude extraction failed")
            return None
    return None


def _parse_llm_json(raw: str) -> dict | None:
    """Parse JSON from LLM response, tolerating markdown fences."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return {
            "gallons": data.get("gallons"),
            "total": data.get("total"),
            "address": data.get("address"),
            "ppg": data.get("ppg"),
            "station": data.get("station"),
        }
    except json.JSONDecodeError:
        log.error("Failed to parse LLM JSON: %s", raw[:200])
        return None


def extract_with_llm(text: str) -> dict | None:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            log.warning("GEMINI_API_KEY not set, skipping LLM extraction")
            return None
        return extract_with_gemini(text)
    elif LLM_PROVIDER == "claude":
        if not ANTHROPIC_API_KEY:
            log.warning("ANTHROPIC_API_KEY not set, skipping LLM extraction")
            return None
        return extract_with_claude(text)
    else:
        log.warning("Unknown LLM_PROVIDER '%s', skipping", LLM_PROVIDER)
        return None


# ---------------------------------------------------------------------------
# Combined extraction with fallback
# ---------------------------------------------------------------------------


def extract_fields(text: str) -> dict:
    """
    Extract Gallons, Total, Address, and Price Per Gallon from receipt text.
    Tries LLM first (unless REGEX_ONLY), then fills gaps with regex.
    """
    result = {"gallons": None, "total": None, "address": None, "ppg": None, "station": None}

    # Try LLM first
    if not REGEX_ONLY:
        llm_result = extract_with_llm(text)
        if llm_result:
            log.debug("LLM result: %s", llm_result)
            for key in result:
                if llm_result.get(key):
                    result[key] = str(llm_result[key]).strip()

    # Fill in any gaps with regex
    missing = [k for k, v in result.items() if not v]
    if missing:
        log.info("Using regex fallback for: %s", ", ".join(missing))
        regex_result = extract_with_regex(text)
        for key in missing:
            if regex_result.get(key):
                result[key] = str(regex_result[key]).strip()

    return result


# ---------------------------------------------------------------------------
# Document processing
# ---------------------------------------------------------------------------


def fields_to_fill(doc: dict, field_map: dict[str, dict]) -> list[str]:
    """
    Determine which of our target fields are empty on this document.
    Returns list of field names that need filling.
    """
    target_names = [FIELD_GALLONS, FIELD_TOTAL, FIELD_ADDRESS, FIELD_PPG]
    existing = {cf["field"]: cf["value"] for cf in doc.get("custom_fields", [])}

    empty = []
    for name in target_names:
        field_info = field_map.get(name)
        if not field_info:
            continue
        val = existing.get(field_info["id"])
        if val is None or val == "" or val == 0:
            empty.append(name)

    return empty


def process_document(
    doc_id: int,
    field_map: dict[str, dict],
    corr_map: dict[str, int] | None = None,
    dry_run: bool = False,
) -> bool:
    """
    Process a single document: extract data, update custom fields,
    and set correspondent if missing.
    Returns True if any changes were made.
    """
    doc = get_document(doc_id)
    title = doc.get("title", f"doc #{doc_id}")
    log.info("Processing: %s (id=%d)", title, doc_id)

    empty_fields = fields_to_fill(doc, field_map)
    needs_correspondent = doc.get("correspondent") is None

    if not empty_fields and not needs_correspondent:
        log.info("  All fields already filled, skipping")
        return False

    if empty_fields:
        log.info("  Empty fields: %s", ", ".join(empty_fields))
    if needs_correspondent:
        log.info("  Missing correspondent")

    content = doc.get("content", "")
    if not content.strip():
        log.warning("  No OCR content available, skipping")
        return False

    extracted = extract_fields(content)
    log.info("  Extracted: %s", extracted)

    changed = False

    # --- Update custom fields ---
    name_to_key = {
        FIELD_GALLONS: "gallons",
        FIELD_TOTAL: "total",
        FIELD_ADDRESS: "address",
        FIELD_PPG: "ppg",
    }

    updates = []
    for field_name in empty_fields:
        key = name_to_key.get(field_name)
        value = extracted.get(key)
        if value is None:
            log.warning("  Could not extract '%s'", field_name)
            continue

        field_info = field_map[field_name]

        # Convert value to match the Paperless field data type
        if field_info["data_type"] == "monetary":
            # Monetary fields require "USD" prefix + exactly 2 decimal places
            value = f"USD{float(value):.2f}"
        elif field_info["data_type"] == "float":
            value = float(value)
        elif field_info["data_type"] == "integer":
            value = int(float(value))
        elif field_info["data_type"] == "string":
            value = str(value)

        updates.append({"field": field_info["id"], "value": value})
        log.info("  Will set %s = %s", field_name, value)

    if updates:
        if dry_run:
            log.info("  [DRY RUN] Would update %d field(s)", len(updates))
        else:
            update_custom_fields(doc_id, updates)
            log.info("  Updated %d field(s)", len(updates))
        changed = True

    # --- Set correspondent if missing ---
    if needs_correspondent and extracted.get("station"):
        station_name = extracted["station"]
        if corr_map is None:
            corr_map = get_correspondents()

        corr_id = find_correspondent(station_name, corr_map)

        if corr_id:
            log.info("  Will set correspondent to existing: '%s' (id=%d)", station_name, corr_id)
        else:
            log.info("  Correspondent '%s' not found, will create", station_name)

        if dry_run:
            log.info("  [DRY RUN] Would set correspondent")
        else:
            if not corr_id:
                corr_id = create_correspondent(station_name)
                corr_map[station_name.strip().lower()] = corr_id
                log.info("  Created correspondent '%s' (id=%d)", station_name, corr_id)
            set_document_correspondent(doc_id, corr_id)
            log.info("  Set correspondent (id=%d)", corr_id)
        changed = True
    elif needs_correspondent:
        log.warning("  Could not extract station name for correspondent")

    return changed


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def run_batch(dry_run: bool = False) -> None:
    """Process all receipt documents with empty custom fields."""
    field_map = get_custom_fields()

    # Validate that the expected fields exist
    for name in [FIELD_GALLONS, FIELD_TOTAL, FIELD_ADDRESS, FIELD_PPG]:
        if name not in field_map:
            log.error(
                "Custom field '%s' not found in Paperless-ngx. "
                "Please create it first.",
                name,
            )
            sys.exit(1)

    corr_map = get_correspondents()

    doc_type_id = get_document_type_id(DOCTYPE_NAME)
    if doc_type_id is None:
        log.error("Document type '%s' not found", DOCTYPE_NAME)
        sys.exit(1)

    log.info("Fetching documents with type '%s' (id=%d)...", DOCTYPE_NAME, doc_type_id)
    documents = get_documents_by_type(doc_type_id)
    log.info("Found %d document(s)", len(documents))

    updated = 0
    for doc in documents:
        if process_document(doc["id"], field_map, corr_map=corr_map, dry_run=dry_run):
            updated += 1

    log.info("Done. Updated %d / %d document(s).", updated, len(documents))


def run_single(doc_id: int, dry_run: bool = False) -> None:
    """Process a single document by ID."""
    field_map = get_custom_fields()

    for name in [FIELD_GALLONS, FIELD_TOTAL, FIELD_ADDRESS, FIELD_PPG]:
        if name not in field_map:
            log.error("Custom field '%s' not found in Paperless-ngx.", name)
            sys.exit(1)

    corr_map = get_correspondents()
    process_document(doc_id, field_map, corr_map=corr_map, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Lock file — prevent concurrent runs
# ---------------------------------------------------------------------------

LOCK_FILE = Path(__file__).parent / ".gas_receipt_parser.lock"


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
        return True
    except OSError:
        return False


def acquire_lock() -> None:
    """Acquire a PID-based lock file. Exit if another instance is running."""
    if LOCK_FILE.exists():
        try:
            stale_pid = int(LOCK_FILE.read_text().strip())
            if _is_pid_alive(stale_pid):
                log.error(
                    "Another instance is already running (PID %d). Exiting.",
                    stale_pid,
                )
                sys.exit(0)
            else:
                log.info("Removing stale lock file (PID %d no longer running)", stale_pid)
        except (ValueError, OSError):
            log.info("Removing invalid lock file")

    LOCK_FILE.write_text(str(os.getpid()))
    atexit.register(release_lock)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # triggers atexit


def release_lock() -> None:
    """Remove the lock file."""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Extract gas receipt data and update Paperless-ngx custom fields",
    )
    parser.add_argument(
        "--doc-id",
        type=int,
        default=None,
        help="Process a single document by ID (for post-consumption hooks)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes",
    )
    args = parser.parse_args()

    if not PAPERLESS_TOKEN:
        log.error("PAPERLESS_API_TOKEN not set. Check your .env file.")
        sys.exit(1)

    acquire_lock()

    if args.doc_id:
        run_single(args.doc_id, dry_run=args.dry_run)
    else:
        run_batch(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
