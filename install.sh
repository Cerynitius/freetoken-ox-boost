#!/usr/bin/env bash
# freetoken-ox-boost installer
# Usage:
#   ./install.sh /path/to/FreeToken          # apply patches + copy overlay (needs a clean v0.1.2 tree)
#   ./install.sh --check /path/to/FreeToken  # dry-run only, writes nothing
set -euo pipefail

CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
TARGET="${1:?Usage: ./install.sh [--check] /path/to/FreeToken (repo root containing python/freetoken)}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -f "$TARGET/python/freetoken/version.py" ] || { echo "error: $TARGET is not a FreeToken repo root"; exit 1; }
VER=$(grep -oE '"[0-9.]+"' "$TARGET/python/freetoken/version.py" | tr -d '"')
if [ "$VER" != "0.1.2" ]; then
  echo "warning: target version $VER != 0.1.2 (patch baseline); continuing, hunks may drift"
fi

echo "== dry-run: checking patches =="
FAIL=0
for p in "$HERE"/patches/*.patch; do
  if git apply --check -p1 --directory="$TARGET" "$p" 2>/dev/null; then
    echo "  ok   $(basename "$p")"
  else
    echo "  FAIL $(basename "$p")  (already applied, or baseline mismatch)"
    FAIL=1
  fi
done
[ "$FAIL" = 1 ] && { echo "some patches do not apply; aborting (do not install twice on the same tree)"; exit 1; }
[ "$CHECK" = 1 ] && { echo "check passed; nothing written"; exit 0; }

echo "== applying patches =="
for p in "$HERE"/patches/*.patch; do
  git apply -p1 --directory="$TARGET" "$p"
done

echo "== copying overlay files =="
cp -R "$HERE"/overlay/freetoken/models/glm5_next "$TARGET"/python/freetoken/models/
cp "$HERE"/overlay/freetoken/kernel/triton/kda_gate.py \
   "$HERE"/overlay/freetoken/kernel/triton/fused_route.py \
   "$TARGET"/python/freetoken/kernel/triton/
cp "$HERE"/overlay/freetoken/kernel/triton/dsv4/hc_norm.py "$TARGET"/python/freetoken/kernel/triton/dsv4/
cp "$HERE"/overlay/freetoken/moe/spec_prefetch.py \
   "$HERE"/overlay/freetoken/moe/lfu_ensure.py \
   "$TARGET"/python/freetoken/moe/
cp "$HERE"/overlay/freetoken/gpu_select.py "$TARGET"/python/freetoken/

python3 -m compileall -q "$TARGET"/python/freetoken && echo "== done: syntax check passed =="
echo "launch example: examples/serve_full.sh; switches: MANIFEST.md"
