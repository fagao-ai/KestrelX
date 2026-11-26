#!/usr/bin/env bash
set -euo pipefail

status=0
while IFS= read -r -d '' file; do
  lines=$(wc -l < "$file")
  if (( lines > 1200 )); then
    echo "$file has $lines lines (limit: 1200)" >&2
    status=1
  fi
done < <(find src kestrel-contract/src kestrel-ffi/src kestrel-node/src \
  kestrel-store-postgres/src kestrel-store-postgres/tests \
  kestrel-store-sqlite/src kestrel-store-sqlite/tests tests kestrel-ffi/tests examples \
  -type f -name '*.rs' -print0)

exit "$status"
