#!/usr/bin/env bash
# Irodori TTS Phase 1 install — systemd 化 ai-pyramid-tts.service + 配置スクリプト群。
# 冪等。再実行 OK。
#
# 使い方:
#   sudo ./deploy/install.sh
#
# 削除:
#   sudo ./deploy/uninstall.sh
#
# install 後の動作:
#   irodori-tts "こんにちは"
#   irodori-tts "こんにちは" --voice male
#   （内部: sudo -n systemctl start ai-pyramid-tts.service を呼ぶ。NOPASSWD ai-pyramid* で許可済）
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "error: root 必要。`sudo ./deploy/install.sh` で実行してください" >&2
  exit 1
fi

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DEPLOY_DIR")"
# user 環境（呼出元の HOME を pickup。sudo 越しでも SUDO_USER の home を取る）
SUDO_HOME="${SUDO_USER:+/home/$SUDO_USER}"
[ -d "${SUDO_HOME:-}" ] || SUDO_HOME="$HOME"
PYSITE_DEFAULT="$SUDO_HOME/.local/lib/python3.10/site-packages"
PYSITE="${IRODORI_PYSITE:-$PYSITE_DEFAULT}"

LIB_DIR=/usr/local/lib/ai-pyramid-tts
BIN_DIR=/usr/local/bin
UNIT_DIR=/etc/systemd/system
TMP_DIR=/tmp/ai-pyramid-tts

echo "[install] REPO=$REPO_ROOT"
echo "[install] PYSITE=$PYSITE"
[ -d "$PYSITE" ] || echo "[warn] PYSITE が存在しない: $PYSITE （axengine 等を持つ user の site-packages が必要）" >&2
[ -d "$REPO_ROOT/build" ] || echo "[warn] $REPO_ROOT/build が無い。axmodel 未転送かも" >&2

# 1. lib スクリプト
install -d -m 755 "$LIB_DIR"
install -m 755 "$DEPLOY_DIR/lib/stop-npu.sh"    "$LIB_DIR/"
install -m 755 "$DEPLOY_DIR/lib/synth.sh"        "$LIB_DIR/"
install -m 755 "$DEPLOY_DIR/lib/restart-npu.sh"  "$LIB_DIR/"

# 2. voices.json (catalog)
install -m 644 "$DEPLOY_DIR/voices.json" "$LIB_DIR/voices.json"

# 3. unit ファイル (@REPO@ / @PYSITE@ 置換)
TMPUNIT="$(mktemp)"
sed -e "s|@REPO@|$REPO_ROOT|g" -e "s|@PYSITE@|$PYSITE|g" \
  "$DEPLOY_DIR/ai-pyramid-tts.service" > "$TMPUNIT"
install -m 644 "$TMPUNIT" "$UNIT_DIR/ai-pyramid-tts.service"
rm "$TMPUNIT"

# 4. wrapper CLI
install -m 755 "$DEPLOY_DIR/irodori-tts" "$BIN_DIR/irodori-tts"

# 5. 共有 tmp ディレクトリ
#   - sticky 無し (0755) にし、user 所有にする → wrapper が後始末で root-owned result.wav を
#     削除可能（dir owner は file owner 関係なく削除できる、sticky 無なら他人 file も rm 可）
#   - user が request.env/text.in を書き、root unit (synth.sh) が読み、root が result.wav を書き、
#     最後に wrapper(user) が trap EXIT で 3 ファイルすべて rm → 真の zero footprint
SUDO_USER_NAME="${SUDO_USER:-admin-user}"
install -d "$TMP_DIR"
chmod 0755 "$TMP_DIR"
chown "$SUDO_USER_NAME:$SUDO_USER_NAME" "$TMP_DIR"

# 6. unit を systemd に認識させる
/bin/systemctl daemon-reload

echo "[install] done."
echo "  unit:   $UNIT_DIR/ai-pyramid-tts.service"
echo "  lib:    $LIB_DIR/"
echo "  cli:    $BIN_DIR/irodori-tts"
echo "  tmp:    $TMP_DIR (1777)"
echo
echo "テスト: irodori-tts 'こんにちは'"
