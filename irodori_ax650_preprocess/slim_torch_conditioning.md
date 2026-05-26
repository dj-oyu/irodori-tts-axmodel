# Slim torch conditioning (Path B) — 共有メモ (2026-05-26)

> 実機(AX8850)で測定した「条件付け(Stage A)の最小フットプリント版」。build host と共有用。
> 関連: `NPU実用化_ギャップ分析.md`(A1/D2/D3), `実機検証_R123_axmodel説明.md`。

## 背景
条件付け(text→KVキャッシュ)は torch CPU fp32 で動く（NPU化されていない=A1）。素朴な
`a_build_cond.py` は `model.safetensors`(1.91GB)を**丸ごとロード**し、構築時に全paramを確保
(1.9GB anon)→ system RAM 1.9Gi に対し OOM/swap律速・load 84–110s。

## slim版がやること（`e2e_demo/slim_stageA.py`, `slim_cond_probe.py`）
- **meta-device 構築** (`with torch.device("meta")`) でゼロ確保。
- `encode_conditions` + `build_context_kv_cache` が触る param **だけ**を材質化:
  `text_encoder.* / text_norm / speaker_encoder.* / speaker_norm /
   blocks.N.attention.{wk_text,wv_text,wk_speaker,wv_speaker,k_norm}` = **281 param / 703MB**。
  残り 1.2GB(DiT本体 MLP/self-attn/AdaLN)は meta のまま＝メモリ0（NPUで動くので不要）。
- safetensors の **no-copy mmap** をそのまま `nn.Parameter` に割当（`.copy_` しない）。
- RoPE `_freqs_cis_cache` は `persistent=False`＝forwardで再計算なので meta で安全。

## 実機測定（services停止, main RAM）
| | 値 |
|---|---|
| 材質化重み | 703MB（text_enc 323 + speaker_enc 231 + block-ctxKV 150 + norm） |
| 起動時 RSS | ~286MB（mmap遅延faultのため） |
| 稼働時 RSS | ~942MB（**anon ~360MB / file ~609MB**）。file=再利用可ページキャッシュ |
| テキスト長依存 | **なし**（tok=5 と tok=29 で同一）＝重み支配 |
| 正当性 | フルモデル cond と **bitwise一致**(146/146 keys)。e2e wav も **bitwise一致**(同md5) |

→ **swap競合(anon)は ~2GB → ~360MB に削減**。fp32品質は完全維持（同ビット）。
mem=2048M で排他なら swap無し常駐可能。axllm 同時稼働も射程内（anon360+axllm~1G+OS~0.5≈1.86G）。

## 使い方
```bash
# slim Stage A: text -> 3-branch CFG cond (cond/text/spk) npz
PYTHONPATH=/path/to/Irodori-TTS python3 e2e_demo/slim_stageA.py \
  --weights /path/to/model.safetensors --text "…" --label plain --out-dir /tmp/slim_e2e \
  [--ref <full-model cond.npz>]   # --ref で bitwise一致を検証

# その後は既存の e2e_npu.py に --cond /tmp/slim_e2e/cond_plain.npz を渡すだけ
#   （DiT=allfcu16_npu3 / dacvae=b0(T119) or T201 はそのまま）

# フットプリント計測専用ツール（VmRSS/anon/file の3点 + bitwise check）
PYTHONPATH=/path/to/Irodori-TTS python3 e2e_demo/slim_cond_probe.py --weights … --ref …
```

## build host 側で効くポイント
- このslimは**条件付けを NPU化しなくても**（A1未着手でも）torch常駐の現実解になる ＝ Path B。
- もし条件付けNPU化(Path A)を進めるなら: 切り出す対象はまさに上記 281 param
  (text_encoder は axmodel 既存, 残り speaker_encoder + 12×{wk/wv_text,wk/wv_speaker,k_norm})。
  可変テキスト長は DiT と同様 max-len + mask で。
- **hybrid**: text_encoder を既存 `axmodel_textenc` に出せば torch側は ~380MB までさらに縮む。
