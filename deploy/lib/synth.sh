#!/bin/bash
# ExecStart: 実際の TTS 合成本体。systemd unit 内で root として動作。
# 引数の受取り:
#   - TEXT は /tmp/ai-pyramid-tts/text.in から（raw、shell escape 不要）
#   - 数値パラメータ (SEED/STEPS/T_VALID/...) は unit の EnvironmentFile= 経由
# wav 出力先は /tmp/ai-pyramid-tts/result.wav 固定。wrapper 側が user 指定先へ copy する。
#
# 必須環境変数（unit の Environment= or EnvironmentFile= 由来）:
#   REPO        リポジトリの絶対パス（install.sh が @REPO@ を置換）
#   PYSITE      axengine/tokenizers のある site-packages（@PYSITE@ 置換）
#   SEED, STEPS, T_VALID, T_VALID_CAP_FRAMES, DURATION_SCALE
set -e

TEXT_FILE=/tmp/ai-pyramid-tts/text.in
RESULT=/tmp/ai-pyramid-tts/result.wav

[ -f "$TEXT_FILE" ] || { echo "[synth] $TEXT_FILE が無い (wrapper の書込忘れ?)" >&2; exit 1; }
TEXT=$(cat "$TEXT_FILE")
[ -z "$TEXT" ] && { echo "[synth] TEXT empty" >&2; exit 1; }

# 既定値（request.env が値を空にしてきた場合のフォールバック）
: "${SEED:=0}"
: "${STEPS:=16}"
: "${T_VALID:=0}"
: "${T_VALID_CAP_FRAMES:=0}"
: "${DURATION_SCALE:=1.0}"
[ -z "${REPO:-}" ] && { echo "[synth] REPO 未設定 (install 時の @REPO@ 置換ミス?)" >&2; exit 1; }

echo "[synth] text=${TEXT@Q} seed=${SEED} steps=${STEPS} t_valid=${T_VALID} cap=${T_VALID_CAP_FRAMES} dscale=${DURATION_SCALE}"

cd "$REPO"
exec env "PYTHONPATH=${PYSITE}" /usr/bin/python3.10 "$REPO/e2e_demo/run_npu_full.py" \
  --text "$TEXT" --out-wav "$RESULT" \
  --seed "$SEED" --num-steps "$STEPS" \
  --t-valid "$T_VALID" --t-valid-cap-frames "$T_VALID_CAP_FRAMES" --duration-scale "$DURATION_SCALE" \
  --cond "$REPO/build/axmodel_cond_textkv_dur/compiled.axmodel" \
  --constants "$REPO/build/cond_constants.npz" \
  --dit "$REPO/build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel" \
  --dacvae "$REPO/build/axmodel_dacvae_T201/compiled.axmodel"
