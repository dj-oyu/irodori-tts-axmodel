#!/bin/bash
# ExecStopPost: TTS 終了後（成功/失敗/SIGKILL すべて）に NPU 競合サービスを順序復帰。
# Type=oneshot + ExecStopPost の組合せで、SIGKILL でもサービス停止状態の放置を防ぐ
# （bash trap EXIT では取れない SIGKILL を systemd が確実に拾う）。
# 復帰順: axllm → yolo → pet-album（yolo の ExecStartPre が axllm:8000 を待つため）。

AXLLM=$(/bin/systemctl list-units 'axllm-serve@*.service' --all --no-legend 2>/dev/null \
  | awk '{print $1}' | head -n1)

echo "[restart-npu] axllm=${AXLLM:-<none>}"

for s in "${AXLLM}" ax-yolo-daemon.service pet-album.service; do
  [ -z "$s" ] && continue
  /bin/systemctl start "$s" 2>/dev/null || true
done
