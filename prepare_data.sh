#!/usr/bin/env bash
set -euo pipefail
: "${KIND:?Set KIND to ruler, ruler_qa1, ruler_qa2, musique, or 2wiki}"
: "${INPUT:?Set INPUT to the source data file}"
: "${OUTPUT:?Set OUTPUT to the destination JSONL file}"
: "${TOKENIZER:?Set TOKENIZER to a local tokenizer path or model ID}"
: "${TARGET_TOKENS:?Set TARGET_TOKENS explicitly; use 0 to disable target-length padding}"
: "${COUNT:?Set COUNT explicitly; use 0 to process all rows}"
: "${SEED:?Set SEED explicitly}"
: "${INSTRUCTION:?Set INSTRUCTION explicitly}"
: "${GOLD_DOCUMENT_COUNT:?Set GOLD_DOCUMENT_COUNT explicitly; use 0 to accept any count}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/prepare_data.py" \
  --kind "$KIND" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --tokenizer "$TOKENIZER" \
  --target-tokens "$TARGET_TOKENS" \
  --count "$COUNT" \
  --seed "$SEED" \
  --instruction "$INSTRUCTION" \
  --gold-document-count "$GOLD_DOCUMENT_COUNT"
