#!/usr/bin/env bash
# Voice scout — seed 探索の対話ラッパー。NPU 排他のためのサービス停止＋復帰だけ担当し、
# 中身は e2e_demo/voice_scout.py（axmodel persistent + 背景プリフェッチ）に委譲する。
#
# 使い方:
#   deploy/voice-scout.sh                       # 既定 0-99 / "おはようございます。"
#   deploy/voice-scout.sh --seeds 0-49 --text "テスト"
#   deploy/voice-scout.sh --skip-existing       # ratings.csv から再開
#
# 環境変数（deploy/tts.sh と同じ既定）:
#   IRODORI_PY, IRODORI_PYSITE, IRODORI_REPO, IRODORI_AXLLM_UNIT, IRODORI_SERVICES, IRODORI_RESTART
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${IRODORI_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${IRODORI_PY:-/usr/bin/python3.10}"
PYSITE="${IRODORI_PYSITE:-$HOME/.local/lib/python3.10/site-packages}"
AXLLM="${IRODORI_AXLLM_UNIT:-$(systemctl list-units 'axllm-serve@*.service' --all --no-legend 2>/dev/null | awk '{print $1}' | head -n1)}"
SERVICES="${IRODORI_SERVICES:-pet-album ax-yolo-daemon ${AXLLM}}"
RESTART="${IRODORI_RESTART:-${AXLLM} ax-yolo-daemon pet-album}"

cleanup() {
  echo "[svc] restart: $RESTART" >&2
  for s in $RESTART; do sudo -n systemctl start "$s" 2>/dev/null || true; done
  return 0
}
trap cleanup EXIT

echo "[svc] stop: $SERVICES" >&2
for s in $SERVICES; do sudo -n systemctl stop "$s" 2>/dev/null || true; done

cd "$REPO"  # voice_scout.py が build/* を相対参照
# 進捗ログは stderr（python 内で sys.stderr に寄せ済）。stdin/stdout は対話 prompt 用に温存。
sudo -n "PYTHONPATH=$REPO/e2e_demo:$PYSITE" "$PY" "$REPO/e2e_demo/voice_scout.py" "$@"
