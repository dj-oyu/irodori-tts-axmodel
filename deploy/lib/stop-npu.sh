#!/bin/bash
# ExecStartPre: NPU 排他のため競合サービスを停止する。systemd unit 内で root 実行。
# axllm は templated unit `axllm-serve@<model>.service` で model 名がデバイス固有のため動的解決。
set -e

AXLLM=$(/bin/systemctl list-units 'axllm-serve@*.service' --all --no-legend 2>/dev/null \
  | awk '{print $1}' | head -n1)

echo "[stop-npu] axllm=${AXLLM:-<none>}"

for s in pet-album.service ax-yolo-daemon.service "${AXLLM}"; do
  [ -z "$s" ] && continue
  /bin/systemctl stop "$s" 2>/dev/null || true
done
