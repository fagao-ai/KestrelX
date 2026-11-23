#!/usr/bin/env bash
set -euo pipefail

status=0
while IFS= read -r -d '' file; do
  lines=$(wc -l < "$file")
  if (( lines > 1200 )); then
    echo "$file has $lines lines (limit: 1200)" >&2
    status=1
  fi
done < <(find src kestrel-ffi/src tests kestrel-ffi/tests examples \
  -type f -name '*.rs' -print0)

exit "$status"
