#!/usr/bin/env bash
# Irodori TTS Phase 1 uninstall — install.sh の逆操作。
#
# 使い方:
#   sudo ./deploy/uninstall.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "error: root 必要。`sudo ./deploy/uninstall.sh` で実行してください" >&2
  exit 1
fi

LIB_DIR=/usr/local/lib/ai-pyramid-tts
BIN_DIR=/usr/local/bin
UNIT_DIR=/etc/systemd/system
TMP_DIR=/tmp/ai-pyramid-tts

# 動作中なら停止（active な oneshot が無ければ no-op）
/bin/systemctl stop ai-pyramid-tts.service 2>/dev/null || true
/bin/systemctl disable ai-pyramid-tts.service 2>/dev/null || true

rm -f "$UNIT_DIR/ai-pyramid-tts.service"
rm -f "$BIN_DIR/irodori-tts"
rm -rf "$LIB_DIR"
rm -rf "$TMP_DIR"

/bin/systemctl daemon-reload

echo "[uninstall] done. ファイル削除済。"
echo "  /usr/local/bin/irodori-tts"
echo "  /etc/systemd/system/ai-pyramid-tts.service"
echo "  /usr/local/lib/ai-pyramid-tts/"
echo "  /tmp/ai-pyramid-tts/"
