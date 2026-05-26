#!/usr/bin/env bash
# Irodori-TTS 全段NPU ワンショット text→wav（実機 AX8850 / AI Pyramid Pro）。
#
# `e2e_demo/run_npu_full.py`（cond① + DiT + dacvae, torch/safetensors 不要）を
# NPU排他のためのサービス停止→TTS→サービス復帰でラップしたデプロイ用CLI。
# cold ~23s（毎回 axmodel ロード）。常駐ではない＝kokoro置換ではなく単発合成用途。
#
# 使い方:
#   deploy/tts.sh "今日はとても良い天気ですね。" -o /tmp/out.wav --play
#   deploy/tts.sh "テキスト" --seed 3 --steps 16
#   deploy/tts.sh "テキスト" -o - | aplay        # wav を stdout に流して直接パイプ
#   deploy/tts.sh "テキスト" -o - | ssh host aplay
#
# 必須環境変数（デバイス固有・publicリポジトリにハードコードしない）:
#   IRODORI_TTS_HOME   irodori_tts パッケージのパス（PYTHONPATH に入る）
#                      例: export IRODORI_TTS_HOME=$HOME/github/Irodori-TTS
# 任意環境変数（既定はこのデバイス向け）:
#   IRODORI_PY         python 実行体            (既定 /usr/bin/python3.10)
#   IRODORI_PYSITE     axengine 等の site-packages（root から見えないため明示）
#                      (既定 $HOME/.local/lib/python3.10/site-packages)
#   IRODORI_REPO       本リポジトリの場所       (既定: このスクリプトの2つ上)
#   IRODORI_AXDIR      axmodel ディレクトリ     (既定: $IRODORI_REPO/build)
#   IRODORI_AXLLM_UNIT axllm の systemd ユニット名（templated: axllm-serve@<model>.service）。
#                      未設定なら実行中インスタンスを動的解決（モデル名はデバイス固有）。
#   IRODORI_SERVICES   停止するNPU排他サービス  (既定 "pet-album ax-yolo-daemon <axllm>")
#   IRODORI_RESTART    復帰順                   (既定 "<axllm> ax-yolo-daemon pet-album")
#                      ※yoloのExecStartPreがaxllm:8000待ち→axllm先。
set -euo pipefail

# ── config ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${IRODORI_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${IRODORI_PY:-/usr/bin/python3.10}"
PYSITE="${IRODORI_PYSITE:-$HOME/.local/lib/python3.10/site-packages}"
AXDIR="${IRODORI_AXDIR:-$REPO/build}"
# axllm は templated unit (axllm-serve@<model>.service)。実行中インスタンスを動的解決
# （モデル名がデバイス固有のためハードコードしない）。停止しないと NPU 排他違反で SEGV。
AXLLM="${IRODORI_AXLLM_UNIT:-$(systemctl list-units 'axllm-serve@*.service' --all --no-legend 2>/dev/null | awk '{print $1}' | head -n1)}"
SERVICES="${IRODORI_SERVICES:-pet-album ax-yolo-daemon ${AXLLM}}"
RESTART="${IRODORI_RESTART:-${AXLLM} ax-yolo-daemon pet-album}"

COND="$AXDIR/axmodel_cond_textkv_dur/compiled.axmodel"
CONST="$AXDIR/cond_constants.npz"
DIT="$AXDIR/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel"
DACVAE="$AXDIR/axmodel_dacvae_T201/compiled.axmodel"

# ── args ────────────────────────────────────────────────────────────────
TEXT=""; OUT="/tmp/tts.wav"; SEED=0; STEPS=16; TVALID=0; DSCALE=1.0
PLAY=0; KEEP_SERVICES=0
usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out)      OUT="$2"; shift 2 ;;
    --seed)        SEED="$2"; shift 2 ;;
    --steps)       STEPS="$2"; shift 2 ;;
    --t-valid)     TVALID="$2"; shift 2 ;;   # 0=A3自動(既定), >0=手動フレーム数
    --duration-scale) DSCALE="$2"; shift 2 ;;
    --play)        PLAY=1; shift ;;
    --keep-services) KEEP_SERVICES=1; shift ;;  # サービスを止めない（NPU空き前提）
    -h|--help)     usage 0 ;;
    -*)            echo "unknown option: $1" >&2; usage 1 ;;
    *)             TEXT="$1"; shift ;;
  esac
done
[[ -z "$TEXT" ]] && { echo "error: テキストが空です" >&2; usage 1; }
[[ -z "${IRODORI_TTS_HOME:-}" ]] && {
  echo "error: IRODORI_TTS_HOME 未設定（irodori_tts のパスを指す）" >&2
  echo "  例: export IRODORI_TTS_HOME=\$HOME/github/Irodori-TTS" >&2; exit 1; }

# `-o -` なら wav を stdout へ（パイプ用）: temp に書いて最後に cat。ログは全て stderr。
# ※ /tmp は sticky+world-writable なので fs.protected_regular が「user所有tempへのroot書込」を
#   拒否する。user所有の一時ディレクトリ(mktemp -d, sticky無)を作り、その中に root が書く。
STREAM=0; WAVPATH="$OUT"; WAVDIR=""
if [[ "$OUT" == "-" ]]; then
  STREAM=1; PLAY=0; WAVDIR="$(mktemp -d)"; WAVPATH="$WAVDIR/out.wav"
fi

# ── preflight: axmodel の存在確認 ──────────────────────────────────────
miss=0
for f in "$COND" "$CONST" "$DIT" "$DACVAE"; do
  [[ -e "$f" ]] || { echo "missing: $f" >&2; miss=1; }
done
[[ $miss -eq 1 ]] && {
  echo "→ axmodel(gitignore済バイナリ)が未転送。build host から転送してください。" >&2
  echo "  cond/dit/dacvae の入手元は docs/deploy_ai_pyramid_pro.md 参照。" >&2; exit 1; }

# ── 後始末（trap）: サービス復帰 + stream の temp 削除。異常終了でも必ず実行 ──
cleanup() {
  [[ $KEEP_SERVICES -eq 0 ]] && {
    echo "[svc] restart: $RESTART" >&2
    for s in $RESTART; do sudo -n systemctl start "$s" 2>/dev/null || true; done
  }
  [[ $STREAM -eq 1 && -n "$WAVDIR" ]] && rm -rf "$WAVDIR"
}
trap cleanup EXIT

# ── NPU排他: サービス停止 ──────────────────────────────────────────────
if [[ $KEEP_SERVICES -eq 0 ]]; then
  echo "[svc] stop: $SERVICES" >&2
  for s in $SERVICES; do sudo -n systemctl stop "$s" 2>/dev/null || true; done
fi

# ── 合成（root NPU 実行）──────────────────────────────────────────────
echo "[tts] text=\"$TEXT\" seed=$SEED steps=$STEPS t-valid=$TVALID -> $OUT" >&2
[[ "$TVALID" -eq 0 ]] && echo "[warn] t-valid=0 (A3自動). duration head は過大予測の既知欠陥があり" \
  "短文で末尾ノイズ/冒頭ゴミが出ることがある。気になる場合は --t-valid <frames>(25fps) を指定。" >&2
cd "$REPO"  # run_npu_full.py が build/model_introspection.json を相対パスで開くため
# python の進捗ログは stdout に出るので stderr へ寄せる（stdout は wav パイプ用に温存）
sudo -n "PYTHONPATH=$IRODORI_TTS_HOME:$PYSITE" "$PY" "$REPO/e2e_demo/run_npu_full.py" \
  --text "$TEXT" --out-wav "$WAVPATH" --seed "$SEED" --num-steps "$STEPS" \
  --t-valid "$TVALID" --duration-scale "$DSCALE" \
  --cond "$COND" --constants "$CONST" --dit "$DIT" --dacvae "$DACVAE" 1>&2

# ── 出力 ────────────────────────────────────────────────────────────────
if [[ $STREAM -eq 1 ]]; then
  cat "$WAVPATH"            # wav を stdout へ（| aplay 等）
  echo "[done] -> stdout" >&2
else
  [[ $PLAY -eq 1 ]] && { echo "[play] $OUT" >&2; aplay -D plughw:0,0 "$OUT" 2>/dev/null || aplay "$OUT" || true; }
  echo "[done] $OUT" >&2
fi
