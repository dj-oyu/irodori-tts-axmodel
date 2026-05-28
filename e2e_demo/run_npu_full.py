#!/usr/bin/env python3
"""完全NPU化 text->wav（実機 AX8850, ROOT）。torch / irodori_tts / model.safetensors 不要。

条件付けを ① cond axmodel (text KV) + bake 済定数(speaker/text-branch KV) で構成し、
DiT(allfcu16_npu3) → dacvae(T201/b0) まで全段 NPU。
tokenizer は tokenizers(Rust製) で tokenizer.json を直読み、normalize_text は stdlib のみ＝
**torch 完全非依存**（cold tokenize ~15s→~1s, IRODORI_TTS_HOME 不要）。

長文対応: 入力テキストは `split_text` で文末→読点で chunk 化 (max 28 chars)。
axmodel sessions を Pipeline class で 1度ロードし、N chunks を連続合成して
無音 (200ms) で繋ぎ単一 wav に出力する。voice_scout.py も同じ Pipeline を import 使用。

no_ref 専用（話者は --seed）。A3: ① が token_logits を出すと t_valid を自動予測（可変長）。
検証は実機で行い、結果は RESULT.md に書いて GitHub 共有。

  sudo -n PYTHONPATH=$HOME/.local/lib/python3.10/site-packages /usr/bin/python3.10 \
    e2e_demo/run_npu_full.py --text "今日はとても良い天気ですね。" --num-steps 16 --seed 0 \
    --out-wav /tmp/npu_full.wav
  # 既定: cond=axmodel_cond_textkv_dur(A3付), dacvae=T201, t-valid=自動。
  # 手動長さ: --t-valid 119。duration補正: --duration-scale 1.1 等。
  # 長文 input: --text-file path.txt または stdin
"""
from __future__ import annotations
import argparse, math, sys, time, wave
from pathlib import Path
import numpy as np
import axengine as axe


# ──────── utility (純粋関数, voice_scout からも import される) ────────

def sway_schedule(num_steps, sway_coeff=-1.0, init_scale=0.999):
    u = np.linspace(0.0, 1.0, num_steps + 1)
    u = u + sway_coeff * (np.cos(0.5 * math.pi * u) + u - 1.0)
    u = np.clip(u, 0.0, 1.0)
    t = (1.0 - u) * init_scale
    return t.astype(np.float32)


def write_wav(path, wav, sr):
    pcm = (np.clip(wav, -1.0, 1.0) * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


# --- text normalization: irodori_tts.text_normalization の移植（stdlib のみ・torch非依存）---
import re, unicodedata
_SIMPLE_REPLACE = {"\t": "", "[n]": "", r"\[n\]": "", "　": "", "？": "?", "！": "!",
                   "♥": "♡", "●": "○", "◯": "○", "〇": "○"}
_REGEX_REPLACE = {
    re.compile(r"[;▼♀♂《》≪≫①②③④⑤⑥]"): "",
    re.compile(r"[˗‐-―⁃−⎯⏤─━⸺⸻]"): "",
    re.compile(r"[～〜]"): "ー",
    re.compile(r"…{3,}"): "……",
}


def _strip_outer_brackets(text):
    pairs = {"「": "」", "『": "』", "（": "）", "【": "】", "(": ")"}
    while len(text) >= 2:
        s, e = text[0], text[-1]
        if s in pairs and pairs[s] == e:
            depth = 0; enclosing = True
            for i, c in enumerate(text):
                if c == s: depth += 1
                elif c == e: depth -= 1
                if depth == 0 and i < len(text) - 1:
                    enclosing = False; break
            if enclosing and depth == 0:
                text = text[1:-1]; continue
        break
    return text


def normalize_text(text):
    for old, new in _SIMPLE_REPLACE.items():
        text = text.replace(old, new)
    for pat, rep in _REGEX_REPLACE.items():
        text = pat.sub(rep, text)
    text = _strip_outer_brackets(text)
    text = unicodedata.normalize("NFKC", text)
    return text.replace("...", "…").replace("..", "…")


# --- markdown / URL 除去（kokoro-tts wrapper の sed パイプを python 化）---
_MD_PATTERNS = [
    (re.compile(r'https?://\S+'), ''),                       # URL
    (re.compile(r'\{[^}]*\}'), ''),                          # {curly} 注記
    (re.compile(r'\*{1,3}'), ''),                            # **bold** / *italic*
    (re.compile(r'~{2}'), ''),                               # ~~strike~~
    (re.compile(r'`{1,3}'), ''),                             # `code` / ```fence```
    (re.compile(r'\[([^\]]*)\]\([^)]*\)'), r'\1'),           # [link](url) → link
    (re.compile(r'^#{1,6}\s*', re.MULTILINE), ''),           # # heading
    (re.compile(r'^\s*[-*+]\s+', re.MULTILINE), ''),         # - / * / + list
    (re.compile(r'^\s*\d+\.\s+', re.MULTILINE), ''),         # 1. ordered list
    (re.compile(r'[<>|\\]'), ''),                            # angle / pipe / backslash
]


def strip_markdown(text: str) -> str:
    for pat, rep in _MD_PATTERNS:
        text = pat.sub(rep, text)
    return text


# break levels: 形態素解析無しで「自然な切れ目」を段階探索する正規表現。
# 上のレベル（句点）優先、なければ次のレベル（読点）… と降順で試す。
# 最終手段は char-level 強制 split（語境界無視）。
_BREAK_LEVELS = [
    re.compile(r'[。！？!?]'),                  # 1. 句点（文の境界, 最良）
    re.compile(r'[、,]'),                        # 2. 読点
    re.compile(r'[はがをにでとへもや]'),         # 3. 助詞（1文字）後
    re.compile(r'[てでしりれ](?=[^てでしりれ])'),# 4. 連用形・接続助詞 (短語誤検知避け)
]


def _break_long(text: str, max_chars: int) -> list[str]:
    """text が max_chars 超なら _BREAK_LEVELS の順で window 内の最後尾候補で split。
    どのレベルも候補なしなら char-level 強制 split（最終手段）。
    語境界を壊しにくいよう「最終候補位置 (=切るほど右)」を採用。"""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        cut = -1
        for pat in _BREAK_LEVELS:
            ms = list(pat.finditer(window))
            if ms:
                cut = ms[-1].end()
                break
        if cut <= 0:
            cut = max_chars  # 強制（語境界破壊。回避不能のとき）
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


def split_text(text: str, max_chars: int = 28) -> list[str]:
    """文を chunk 列に分割。strip_markdown + whitespace normalize の後、
    まず句点で primary split、長すぎる chunk は break level の段階探索で再分割。
    形態素解析依存なし。語境界を完全保証するものではない（最終手段は強制 split）"""
    text = strip_markdown(text)
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    parts = re.split(r'(?<=[。！？!?])', text)
    parts = [p.strip() for p in parts if p.strip()]

    chunks: list[str] = []
    for p in parts:
        chunks.extend(_break_long(p, max_chars))
    return chunks


def tokenize(text, seq=256):
    """torch非依存: tokenizers(Rust) で tokenizer.json を直読み。ID は transformers 経路と一致。"""
    import json
    from tokenizers import Tokenizer
    from huggingface_hub import hf_hub_download
    cfg = json.load(open("build/model_introspection.json"))["model_cfg"]
    repo = cfg["text_tokenizer_repo"]; add_bos = bool(cfg.get("text_add_bos", True))
    tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", local_files_only=True))
    ids = tok.encode(normalize_text(text).strip(), add_special_tokens=False).ids
    if add_bos:  # bos トークンは special_tokens_map.json から引き、vocab id に解決（ハードコードしない）
        sm = json.load(open(hf_hub_download(repo, "special_tokens_map.json", local_files_only=True)))
        bos = sm.get("bos_token")
        bos = bos.get("content") if isinstance(bos, dict) else bos
        bos_id = tok.token_to_id(bos) if bos is not None else None
        if bos_id is None:
            raise RuntimeError("text_add_bos=True だが bos_token_id を解決できない")
        ids = [bos_id] + ids
    ids = ids[:seq]
    arr = np.zeros((1, seq), np.int64); mask = np.zeros((1, seq), np.uint8)
    arr[0, :len(ids)] = ids; mask[0, :len(ids)] = 1
    return arr, mask, int(mask.sum())


# ──────── Pipeline class: axmodel sessions を1度ロード → 複数 chunk 連続合成 ────────

class Pipeline:
    """axmodel sessions を保持し synthesize(text, seed=..) を繰り返し呼べる版。
    voice_scout.py からも import される共通 Pipeline。"""

    def __init__(self, cond_path: str, dit_path: str, dacvae_path: str, constants_path: str):
        t0 = time.time()
        self.cond = axe.InferenceSession(cond_path)
        self.dit = axe.InferenceSession(dit_path)
        self.dacvae = axe.InferenceSession(dacvae_path)
        self.K = np.load(constants_path)
        ishapes = {i.name: tuple(i.shape) for i in self.dit.get_inputs()}
        self.T = ishapes["x_t"][1]
        self.latent_dim = ishapes["x_t"][2]
        self.nlayers = sum(1 for n in ishapes if n.startswith("k_text_"))
        zin = self.dacvae.get_inputs()[0]
        self.dac_T = zin.shape[2]
        self.dac_in_name = zin.name
        self._cond_in_names = [i.name for i in self.cond.get_inputs()]
        self._cond_out_names = [o.name for o in self.cond.get_outputs()]
        print(f"[pipeline] sessions ready in {time.time()-t0:.1f}s "
              f"(T={self.T}, latent={self.latent_dim}, layers={self.nlayers}, dac_T={self.dac_T})",
              flush=True)

    def synthesize(self, text: str, *, seed: int = 0, t_valid: int = 0,
                   t_valid_cap_frames: int = 0, duration_scale: float = 1.0,
                   num_steps: int = 16,
                   cfg_text: float = 3.0, cfg_spk: float = 5.0,
                   cfg_min_t: float = 0.5, cfg_max_t: float = 1.0,
                   hop: int = 1920, sr: int = 48000,
                   min_sec: float = 0.3, max_sec: float = 8.0,
                   keep_bos_frames: bool = False,
                   verbose: bool = True) -> tuple[np.ndarray, int]:
        # 1) tokenize
        ids, text_mask, ntok = tokenize(text)
        if verbose:
            print(f"  [tok] {ntok} tokens", flush=True)

        # 2) cond
        feed_cond = {"input_ids": ids.astype(np.int32), "mask": text_mask.astype(np.uint8)}
        feed_cond = {n: feed_cond[n] for n in self._cond_in_names}
        cout = {self._cond_out_names[i]: np.asarray(o, np.float32)
                for i, o in enumerate(self.cond.run(None, feed_cond))}
        token_logits = cout.pop("token_logits", None)
        text_kv = cout

        # 3) t_valid 決定（手動 > A3 + duration_scale + cap > fallback）
        if t_valid > 0:
            tv = min(t_valid, self.T)
            if verbose:
                print(f"  [A3] t_valid={tv} (manual override)", flush=True)
        elif token_logits is not None:
            token_frames = np.logaddexp(0.0, token_logits.astype(np.float64))  # softplus
            dur_mask = text_mask.astype(np.float64).copy()
            if not keep_bos_frames:
                dur_mask[..., 0] = 0.0
            pred = float((token_frames * dur_mask).sum())
            min_f = max(1, math.ceil(min_sec * sr / hop))
            max_f = min(self.T, max(1, math.floor(max_sec * sr / hop)))
            tv = int(round(pred * duration_scale))
            tv = max(min_f, min(max_f, tv))
            capped = False
            if t_valid_cap_frames > 0 and tv > t_valid_cap_frames:
                tv = max(min_f, t_valid_cap_frames); capped = True
            if verbose:
                cap_msg = f" cap={t_valid_cap_frames}{'(applied)' if capped else '(slack)'}" \
                          if t_valid_cap_frames > 0 else ""
                print(f"  [A3] frames={pred:.1f} scale={duration_scale}{cap_msg} "
                      f"-> t_valid={tv} ({tv*hop/sr:.2f}s)", flush=True)
        else:
            tv = min(119, self.T)
            if verbose:
                print(f"  [A3] no token_logits; fallback t_valid={tv}", flush=True)

        # 4) feeds 組立（独立CFG 3分岐）
        latent_mask = np.zeros((1, self.T), np.uint8); latent_mask[:, :tv] = 1
        zero_tm = np.zeros_like(text_mask.astype(np.uint8))
        spk_mask = self.K["speaker_mask"]; zero_sm = np.zeros_like(spk_mask)

        def assemble(branch: str) -> dict:
            f = {"latent_mask": latent_mask}
            f["text_mask"] = zero_tm if branch == "text" else text_mask.astype(np.uint8)
            f["speaker_mask"] = zero_sm if branch == "spk" else spk_mask
            for L in range(self.nlayers):
                if branch == "text":
                    f[f"k_text_{L}"] = self.K[f"text_zero_k_text_{L}"]
                    f[f"v_text_{L}"] = self.K[f"text_zero_v_text_{L}"]
                else:
                    f[f"k_text_{L}"] = text_kv[f"k_text_{L}"]
                    f[f"v_text_{L}"] = text_kv[f"v_text_{L}"]
                if branch == "spk":
                    f[f"k_spk_{L}"] = self.K[f"spk_zero_k_spk_{L}"]
                    f[f"v_spk_{L}"] = self.K[f"spk_zero_v_spk_{L}"]
                else:
                    f[f"k_spk_{L}"] = self.K[f"spk_real_k_spk_{L}"]
                    f[f"v_spk_{L}"] = self.K[f"spk_real_v_spk_{L}"]
            return {n: np.asarray(v, np.float32) if n not in ("text_mask", "speaker_mask", "latent_mask") else v
                    for n, v in f.items()}
        feeds = {b: assemble(b) for b in ("cond", "text", "spk")}

        # 5) DiT sampling (Euler + independent CFG, sway schedule)
        t_sched = sway_schedule(num_steps)
        rng = np.random.default_rng(seed)
        x_t = rng.standard_normal((1, self.T, self.latent_dim)).astype(np.float32)
        tot_ms = 0.0
        for i in range(num_steps):
            t = float(t_sched[i]); tn = float(t_sched[i + 1]); s = time.time()
            if cfg_min_t <= t <= cfg_max_t:
                f_c = dict(feeds["cond"]); f_c["x_t"] = x_t; f_c["t"] = np.array([t], np.float32)
                vc = self.dit.run(None, f_c)[0]
                f_t = dict(feeds["text"]); f_t["x_t"] = x_t; f_t["t"] = np.array([t], np.float32)
                vt = self.dit.run(None, f_t)[0]
                f_s = dict(feeds["spk"]); f_s["x_t"] = x_t; f_s["t"] = np.array([t], np.float32)
                vs = self.dit.run(None, f_s)[0]
                v = vc + cfg_text * (vc - vt) + cfg_spk * (vc - vs)
            else:
                f_c = dict(feeds["cond"]); f_c["x_t"] = x_t; f_c["t"] = np.array([t], np.float32)
                v = self.dit.run(None, f_c)[0]
            tot_ms += (time.time() - s) * 1000
            x_t = x_t + v * (tn - t)
        if verbose:
            print(f"  [dit] {tot_ms/1000:.1f}s ({tot_ms/max(1,num_steps):.0f}ms/step)", flush=True)

        # 6) dacvae
        z_full = np.transpose(x_t, (0, 2, 1)).astype(np.float32)
        if z_full.shape[2] >= self.dac_T:
            z = np.ascontiguousarray(z_full[:, :, :self.dac_T])
        else:
            z = np.ascontiguousarray(np.pad(z_full, ((0, 0), (0, 0), (0, self.dac_T - z_full.shape[2]))))
        audio = np.asarray(self.dacvae.run(None, {self.dac_in_name: z})[0]).reshape(-1)
        audio = audio[:int(min(tv, self.dac_T) * hop)]
        return audio, tv


# ──────── CLI ────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="", help="入力テキスト。空 + --text-file 無し + tty 入力 + stdin 無し ならエラー")
    ap.add_argument("--text-file", default="", help="テキストをファイルから読込（複数行可）")
    ap.add_argument("--max-chars-per-chunk", type=int, default=28,
                    help="1 chunk あたりの最大文字数。T_max=201 frames を超えないため。default 28")
    ap.add_argument("--inter-chunk-pause-ms", type=int, default=200,
                    help="chunk 間の無音の長さ(ms)。default 200")
    ap.add_argument("--cond", default="build/axmodel_cond_textkv_dur/compiled.axmodel")
    ap.add_argument("--constants", default="build/cond_constants.npz")
    ap.add_argument("--dit", default="build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel")
    ap.add_argument("--dacvae", default="build/axmodel_dacvae_T201/compiled.axmodel")
    ap.add_argument("--t-valid", type=int, default=0,
                    help="0 = A3自動(token_logits から duration予測); >0 で手動上書き")
    ap.add_argument("--duration-scale", type=float, default=1.0)
    ap.add_argument("--t-valid-cap-frames", type=int, default=0,
                    help="A3自動予測の上限クリップ(frames, 25fps相当)。0=無効")
    ap.add_argument("--keep-bos-frames", action="store_true")
    ap.add_argument("--min-sec", type=float, default=0.3)
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--hop", type=int, default=1920)
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--cfg-text", type=float, default=3.0)
    ap.add_argument("--cfg-spk", type=float, default=5.0)
    ap.add_argument("--cfg-min-t", type=float, default=0.5)
    ap.add_argument("--cfg-max-t", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--out-wav", default="/tmp/npu_full.wav")
    args = ap.parse_args()

    # 入力 text の確定: --text > --text-file > stdin (pipe) > エラー
    if args.text:
        text = args.text
    elif args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        print("[err] no text. give --text / --text-file / pipe to stdin", file=sys.stderr)
        sys.exit(1)

    chunks = split_text(text, max_chars=args.max_chars_per_chunk)
    if not chunks:
        print("[err] no chunks after split/normalize", file=sys.stderr); sys.exit(1)
    print(f"[chunks] {len(chunks)} pieces (max_chars={args.max_chars_per_chunk})", flush=True)
    for i, c in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {c!r}", flush=True)

    pipeline = Pipeline(args.cond, args.dit, args.dacvae, args.constants)

    audios: list[np.ndarray] = []
    syn_kwargs = dict(
        seed=args.seed, t_valid=args.t_valid, t_valid_cap_frames=args.t_valid_cap_frames,
        duration_scale=args.duration_scale, num_steps=args.num_steps,
        cfg_text=args.cfg_text, cfg_spk=args.cfg_spk,
        cfg_min_t=args.cfg_min_t, cfg_max_t=args.cfg_max_t,
        hop=args.hop, sr=args.sr, min_sec=args.min_sec, max_sec=args.max_sec,
        keep_bos_frames=args.keep_bos_frames,
    )
    t_all = time.time()
    for i, chunk in enumerate(chunks):
        print(f"[chunk {i+1}/{len(chunks)}]", flush=True)
        audio, tv = pipeline.synthesize(chunk, **syn_kwargs)
        audios.append(audio)

    # concat with inter-chunk pause
    if len(audios) == 1:
        out = audios[0]
    else:
        pause_n = max(0, int(args.inter_chunk_pause_ms / 1000 * args.sr))
        pause = np.zeros(pause_n, np.float32) if pause_n else None
        parts: list[np.ndarray] = []
        for i, a in enumerate(audios):
            if i > 0 and pause is not None:
                parts.append(pause)
            parts.append(a)
        out = np.concatenate(parts)

    write_wav(args.out_wav, out, args.sr)
    print(f"[saved] {args.out_wav} dur={len(out)/args.sr:.2f}s "
          f"({len(chunks)} chunks, total {time.time()-t_all:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
