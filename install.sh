#!/usr/bin/env bash
# freetoken-glm5-boost 安装器
# 用法:
#   ./install.sh /path/to/FreeToken        # 应用补丁 + 拷贝 overlay(需 v0.1.2 干净树)
#   ./install.sh --check /path/to/FreeToken  # 只做 dry-run,不落盘
set -euo pipefail

CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
TARGET="${1:?用法: ./install.sh [--check] /path/to/FreeToken (仓库根,含 python/freetoken)}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -f "$TARGET/python/freetoken/version.py" ] || { echo "错误: $TARGET 不是 FreeToken 仓库根"; exit 1; }
VER=$(grep -oE '"[0-9.]+"' "$TARGET/python/freetoken/version.py" | tr -d '"')
if [ "$VER" != "0.1.2" ]; then
  echo "警告: 目标版本 $VER != 0.1.2(补丁基线),继续但可能有 hunk 偏移"
fi

echo "== dry-run 检查补丁 =="
FAIL=0
for p in "$HERE"/patches/*.patch; do
  if git apply --check -p1 --directory="$TARGET" "$p" 2>/dev/null; then
    echo "  ok   $(basename "$p")"
  else
    echo "  FAIL $(basename "$p")  (可能已应用过或基线不符)"
    FAIL=1
  fi
done
[ "$FAIL" = 1 ] && { echo "存在无法应用的补丁,中止(已应用过的树请勿重复安装)"; exit 1; }
[ "$CHECK" = 1 ] && { echo "check 通过,未落盘"; exit 0; }

echo "== 应用补丁 =="
for p in "$HERE"/patches/*.patch; do
  git apply -p1 --directory="$TARGET" "$p"
done

echo "== 拷贝 overlay 新文件 =="
cp -R "$HERE"/overlay/freetoken/models/glm5_next "$TARGET"/python/freetoken/models/
cp "$HERE"/overlay/freetoken/kernel/triton/kda_gate.py \
   "$HERE"/overlay/freetoken/kernel/triton/fused_route.py \
   "$TARGET"/python/freetoken/kernel/triton/
cp "$HERE"/overlay/freetoken/kernel/triton/dsv4/hc_norm.py "$TARGET"/python/freetoken/kernel/triton/dsv4/
cp "$HERE"/overlay/freetoken/moe/spec_prefetch.py \
   "$HERE"/overlay/freetoken/moe/lfu_ensure.py \
   "$TARGET"/python/freetoken/moe/
cp "$HERE"/overlay/freetoken/gpu_select.py "$TARGET"/python/freetoken/

python3 -m compileall -q "$TARGET"/python/freetoken && echo "== 完成:语法检查通过 =="
echo "启动示例见 examples/serve_full.sh;各开关见 MANIFEST.md"
