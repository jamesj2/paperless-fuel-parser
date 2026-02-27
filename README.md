# Gas Receipt Parser for Paperless-ngx

Automatically extracts **Gallons**, **Total**, and **Address** from gas station receipts in Paperless-ngx and fills in custom fields.

## Prerequisites

- Paperless-ngx v2.x with API access
- Python 3.10+
- Custom fields created in Paperless-ngx: `Gallons`, `Total`, `Address`
- A document type (e.g. `Receipt`) assigned to gas station receipts
- An API key for Gemini or Anthropic (optional — regex-only mode available)

## Setup

```bash
# Clone / navigate to this directory
cd paperless

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit the config
cp .env.example .env
# Edit .env with your Paperless URL, API token, and LLM API key
```

## Usage

### Batch mode — process all receipts with empty fields

```bash
python gas_receipt_parser.py
```

### Dry run — see what would be updated without making changes

```bash
python gas_receipt_parser.py --dry-run
```

### Single document — process one document by ID

```bash
python gas_receipt_parser.py --doc-id 123
```

### Regex-only mode — skip LLM, use pattern matching only

Set `REGEX_ONLY=true` in your `.env` file.

## Post-Consumption Hook

To auto-process receipts as they're consumed by Paperless-ngx:

1. Mount the scripts into your Paperless container (docker-compose example):
   ```yaml
   services:
     paperless:
       volumes:
         - ./:/usr/src/paperless/scripts/gas-parser
   ```

2. Set the post-consume script environment variable:
   ```yaml
   environment:
     PAPERLESS_POST_CONSUME_SCRIPT: /usr/src/paperless/scripts/gas-parser/post_consume.sh
   ```

3. Install Python dependencies inside the container:
   ```bash
   docker exec -it paperless pip install -r /usr/src/paperless/scripts/gas-parser/requirements.txt
   ```

4. Ensure the `.env` file is present in the mounted directory.

**Note:** The post-consume script runs for *every* consumed document. The parser checks if the document has the correct type and empty fields before doing any extraction, so non-receipt documents are skipped quickly.

## Custom Field Types

The script handles these Paperless custom field data types:

| Field     | Recommended Type | Notes                          |
|-----------|-----------------|--------------------------------|
| Gallons   | Float           | e.g. `12.345`                  |
| Total     | Monetary        | e.g. `45.67`                   |
| Address   | String          | e.g. `123 Main St, City, ST`   |

## How Extraction Works

1. **LLM (primary):** Sends the receipt OCR text to Gemini or Claude with a structured prompt. The LLM returns JSON with the extracted fields. Most accurate for varied receipt formats.

2. **Regex (fallback):** Pattern matching for common receipt layouts. Handles standard formats like `12.345 GAL`, `TOTAL $45.67`, and street addresses near the top of the receipt.

If the LLM extracts some but not all fields, regex fills in the gaps.
