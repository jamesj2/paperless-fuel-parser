# paperless-fuel-parser

Automatically extracts **Gallons**, **Total**, **Address**, and **Price Per Gallon** from scanned gas station receipts in [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) and fills in custom fields via the API.

Uses **Gemini** or **Claude** for intelligent OCR text parsing with **regex fallback** when the LLM is unavailable.

## Features

- **LLM-powered extraction** — Gemini or Claude parses receipt OCR text with high accuracy, even handling mangled OCR
- **Regex fallback** — pattern matching fills in any fields the LLM misses or when running without an API key
- **Batch & single-doc modes** — process all receipts at once or one document at a time
- **Post-consumption hook** — automatically parse receipts as they're scanned into Paperless-ngx
- **Rate-limit retry** — exponential backoff on 429 errors with configurable attempts
- **Concurrent run protection** — PID-based lock file prevents overlapping executions
- **Dry-run mode** — preview what would be updated without writing changes

## Prerequisites

- Paperless-ngx v2.x with API access
- Python 3.10+
- Custom fields created in Paperless-ngx: `Gallons`, `Total`, `Address`, `Price Per Gallon`
- A document type (e.g. `Car: Fuel Receipt`) assigned to gas station receipts
- An API key for Gemini or Anthropic (optional — regex-only mode available)

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/paperless-fuel-parser.git
cd paperless-fuel-parser

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

### Dry run — preview changes without writing

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

**Note:** The post-consume script runs for *every* consumed document. The parser checks the document type and whether fields are already filled before doing any extraction, so non-receipt documents are skipped quickly.

## Custom Field Types

The script handles these Paperless custom field data types automatically:

| Field           | Recommended Type | Example          |
|-----------------|-----------------|------------------|
| Gallons         | Float           | `12.345`         |
| Total           | Monetary        | `USD45.67`       |
| Address         | String          | `123 Main St, City, ST 12345` |
| Price Per Gallon| Float           | `3.199`          |

> **Monetary fields** are formatted as `USD` + 2 decimal places per Paperless-ngx requirements. Use Float for fields that need more precision (e.g. 3 decimal gas prices).

## How Extraction Works

1. **LLM (primary):** Sends the receipt OCR text to Gemini or Claude with a structured JSON prompt. The LLM handles varied receipt layouts, OCR artifacts, and multi-line addresses intelligently.

2. **Regex (fallback):** Pattern matching for common receipt formats. Handles `GALLONS: 12.345`, `PRICE/G: $3.199`, `TOTAL $45.67`, addresses split across multiple lines, and OCR quirks like stray spaces in prices.

If the LLM extracts some but not all fields, regex fills in the gaps.

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `PAPERLESS_URL` | Paperless-ngx base URL | `http://localhost:8000` |
| `PAPERLESS_API_TOKEN` | API authentication token | — |
| `PAPERLESS_DOCTYPE_NAME` | Document type to filter by | `Car: Fuel Receipt` |
| `LLM_PROVIDER` | `gemini` or `claude` | `gemini` |
| `REGEX_ONLY` | Skip LLM, regex only | `false` |
| `LLM_MAX_RETRIES` | Retry attempts on rate limit | `3` |
| `LLM_RETRY_BASE_DELAY` | Base delay in seconds (doubles each retry) | `10` |

## License

MIT
