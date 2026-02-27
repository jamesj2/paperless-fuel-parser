#!/usr/bin/env bash
# Paperless-ngx post-consumption hook for gas receipt parsing.
#
# Paperless passes the document ID as $DOCUMENT_ID.
# Set POST_CONSUME_SCRIPT_PATH in your Paperless-ngx config to point here.
#
# Example (docker-compose environment):
#   PAPERLESS_POST_CONSUME_SCRIPT=/usr/src/paperless/scripts/post_consume.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$DOCUMENT_ID" ]; then
    echo "Error: DOCUMENT_ID not set. This script should be called by Paperless-ngx."
    exit 1
fi

exec python3 "$SCRIPT_DIR/gas_receipt_parser.py" --doc-id "$DOCUMENT_ID"
