#!/bin/bash
# ExecStopPost: TTS 終了後（成功/失敗/SIGKILL すべて）に NPU 競合サービスを復帰。
# Type=oneshot + ExecStopPost の組合せで、SIGKILL でもサービス停止状態の放置を防ぐ
# （bash trap EXIT では取れない SIGKILL を systemd が確実に拾う）。
#
# 注意: `systemctl start --no-block` で fire-and-forget。
#   default の systemctl start は service が active になるまで待つが、axllm は cold start
#   ~15s かかり、これを ExecStopPost で blocking 待ちすると stop-post timeout に当たって
#   TERMINATE される。--no-block で起動コマンドだけ投げて即終了し、systemd 側に並行起動を
#   任せる（順序は axllm.service の After=/Wants= で管理されている前提）。

AXLLM=$(/bin/systemctl list-units 'axllm-serve@*.service' --all --no-legend 2>/dev/null \
  | awk '{print $1}' | head -n1)

echo "[restart-npu] axllm=${AXLLM:-<none>} (--no-block, systemd 側で並行起動)"

for s in "${AXLLM}" ax-yolo-daemon.service pet-album.service; do
  [ -z "$s" ] && continue
  /bin/systemctl start --no-block "$s" 2>/dev/null || true
done
