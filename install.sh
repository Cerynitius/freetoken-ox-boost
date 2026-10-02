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
  if git apply --check --unsafe-paths -p1 --directory="$TARGET" "$p" 2>/dev/null; then
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
  git apply --unsafe-paths -p1 --directory="$TARGET" "$p"
done

echo "== copying overlay files =="
# the whole overlay tree (new files only; mirrors python/freetoken/)
cp -R "$HERE"/overlay/freetoken/. "$TARGET"/python/freetoken/

python3 -m compileall -q "$TARGET"/python/freetoken && echo "== done: syntax check passed =="
echo "launch example: examples/serve_full.sh; switches: MANIFEST.md"
echo "vision support needs pillow: pip install pillow (FREETOKEN_GLM5_VISION=1)"
