#!/usr/bin/env python3
"""Voice scout — 100 seed の話者バリエーションを対話的に評価して voice catalog を curation する。

実機 AX8850（ROOT）。axmodel を一度ロードして persistent session で seeds をループ。
背景 thread で「N+1 を先 synth」しておき、ユーザーが N を聴いてタグ入力している間に
次が用意されている＝律速はユーザー入力＋再生のみ。

**ディスク戦略**: 再生は aplay の stdin にメモリから流し、wav の永続化は
**採用タグ(m/f/c/e/k) または note 付き**の seed のみ。skip した seed は zero footprint。
100 seed 走らせても残るのは採用候補のみ（典型 10-30 個）。

  sudo -n PYTHONPATH=$HOME/.local/lib/python3.10/site-packages /usr/bin/python3.10 \
    e2e_demo/voice_scout.py --seeds 0-99 --text "おはようございます" \
    --out-dir /tmp/voice_scout --ratings /tmp/voice_scout/ratings.csv

ユーザー前提: 競合する NPU サービス（axllm-serve@*, ax-yolo-daemon, pet-album）は外で停止済。
`deploy/voice-scout.sh` 経由なら自動で停止＋復帰される。
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import axengine as axe

# 重複を避けるため orchestration は voice_scout 内に持つが、純粋関数は run_npu_full から import。
# Pipeline class + 純粋関数を run_npu_full から import (DRY: voice_scout は orchestration のみ)
from run_npu_full import Pipeline, sway_schedule, write_wav, tokenize


def parse_seeds(spec: str) -> list[int]:
    """'0-99' / '0,5,12' / '0-9,20,30-39' に対応"""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


TAG_HELP = ("[m]=男 [f]=女 [c]=子供 [e]=高齢 [k]=keep(良) [s]=skip(不採用) "
            "[r]=replay [g]=次のテキスト [n]=note [q]=quit [?]=help")
TAG_VALID = {"m", "f", "c", "e", "k", "s"}

# `g` でローテートする例文集（挨拶/平叙/疑問/依頼/別れ。9-14 mora で揃え、抑揚で voice character を多面評価）
DEFAULT_TEXTS = [
    "おはようございます。",
    "今日はとても良い天気ですね。",
    "あれ、誰かいますか？",
    "ちょっと待ってください。",
    "ありがとう、また会いましょう。",
]


def load_existing(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                out[int(row["seed"])] = row
            except (KeyError, ValueError):
                pass
    return out


def append_rating(path: Path, seed: int, tag: str, note: str) -> None:
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["seed", "tag", "note", "timestamp"])
        w.writerow([seed, tag, note, time.strftime("%Y-%m-%dT%H:%M:%S")])


def _audio_to_wav_bytes(audio: np.ndarray, sr: int = 48000) -> bytes:
    """numpy float -> RIFF wav bytes (16-bit mono PCM, no temp file)"""
    import io, wave
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_audio(audio: np.ndarray, sr: int = 48000) -> None:
    """ディスクに書かず aplay の stdin へ流す（採用しない wav を /tmp に残さないため）。"""
    subprocess.run(["aplay", "-q", "-D", "plughw:0,0"],
                   input=_audio_to_wav_bytes(audio, sr),
                   stderr=subprocess.DEVNULL, check=False)


# 採用相当タグ（このタグが付いた seed のみ wav をディスクに残す）
TAG_KEEP_WAV = {"m", "f", "c", "e", "k"}


def synth_worker(pipeline: Pipeline, seeds: list[int], text: str, params: dict,
                 q_out: queue.Queue, stop_evt: threading.Event,
                 synth_lock: threading.Lock) -> None:
    """背景 thread: seed を逐次 synth して queue に積む。stop_evt で中断対応。
    NPU は排他リソースのため synth_lock で main thread の `g` regen と直列化する。"""
    for seed in seeds:
        if stop_evt.is_set():
            break
        try:
            with synth_lock:
                audio, tv = pipeline.synthesize(text, seed=seed, **params)
            q_out.put(("ok", seed, audio, tv))
        except Exception as e:
            q_out.put(("err", seed, str(e), 0))
    q_out.put(("done", -1, None, 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-99", help="例: 0-99 / 0,5,12 / 0-9,20,30")
    ap.add_argument("--text", default=None,
                    help="単一テキストモード（rotation 無し、g は no-op）。互換用")
    ap.add_argument("--texts", action="append", default=None,
                    help="複数テキスト（g キーでローテート）。複数回指定可。"
                         "未指定なら DEFAULT_TEXTS の 5 文")
    ap.add_argument("--out-dir", default="/tmp/voice_scout")
    ap.add_argument("--ratings", default="", help="既定: <out-dir>/ratings.csv")
    ap.add_argument("--skip-existing", action="store_true",
                    help="ratings.csv に既タグの seed をスキップ（再開時に推奨）")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="背景プリフェッチを無効化（デバッグ用）")
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--t-valid", type=int, default=0)
    ap.add_argument("--t-valid-cap-frames", type=int, default=0)
    # axmodel paths (deploy/tts.sh と同既定)
    ap.add_argument("--cond", default="build/axmodel_cond_textkv_dur/compiled.axmodel")
    ap.add_argument("--constants", default="build/cond_constants.npz")
    ap.add_argument("--dit", default="build/axmodel_kv_long_lm_allfcu16_npu3/compiled.axmodel")
    ap.add_argument("--dacvae", default="build/axmodel_dacvae_T201/compiled.axmodel")
    args = ap.parse_args()

    seeds_all = parse_seeds(args.seeds)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ratings_path = Path(args.ratings) if args.ratings else (out_dir / "ratings.csv")
    existing = load_existing(ratings_path)
    seeds = [s for s in seeds_all if not (args.skip_existing and s in existing)]

    # テキスト集: --texts > --text > DEFAULT_TEXTS。worker は texts[0] のみ prefetch。
    # `g` キーで texts[1], [2], ... をローテート再合成（main thread, lock 付き）。
    texts: list[str] = args.texts if args.texts else ([args.text] if args.text else list(DEFAULT_TEXTS))

    print(f"[scout] {len(seeds)}/{len(seeds_all)} seeds queued "
          f"(skip-existing={args.skip_existing}, already-rated={len(existing)})", file=sys.stderr)
    print(f"[scout] {len(texts)} text(s): {texts[0]!r}"
          + (f" (+{len(texts)-1} more, `g` でローテート)" if len(texts) > 1 else "")
          + f" steps={args.num_steps} out_dir={out_dir} ratings={ratings_path}", file=sys.stderr)
    # キー一覧は最初に 1 回だけ案内。以降の prompt は短くする（`?` でいつでも再表示）
    print(f"[keys] {TAG_HELP}", file=sys.stderr)

    pipeline = Pipeline(args.cond, args.dit, args.dacvae, args.constants)
    params = dict(num_steps=args.num_steps, t_valid=args.t_valid,
                  t_valid_cap_frames=args.t_valid_cap_frames)

    # 背景 prefetch thread。synth_lock で worker と main thread の `g` regen が
    # NPU を奪い合わないように直列化（NPU は排他資源、2 thread 同時呼出で SEGV）。
    q = queue.Queue(maxsize=1 if not args.no_pipeline else 0)
    stop_evt = threading.Event()
    synth_lock = threading.Lock()
    worker = threading.Thread(target=synth_worker,
                              args=(pipeline, seeds, texts[0], params, q, stop_evt, synth_lock),
                              daemon=True)
    worker.start()

    counted = 0; quit_now = False
    try:
        while not quit_now:
            kind, seed, payload, tv = q.get()
            if kind == "done":
                break
            if kind == "err":
                print(f"[err] seed={seed}: {payload}", file=sys.stderr)
                continue
            audio = payload  # 現在再生中の audio（テキストローテートで差し替わる）
            audio_canonical = audio  # texts[0] の音声。save 時はこちらを永続化する
            counted += 1
            dur = len(audio) / 48000
            print(f"\n[{counted:3d}/{len(seeds)}] seed={seed:3d} t_valid={tv} ({dur:.2f}s) "
                  f"text[1/{len(texts)}]={texts[0]!r}", file=sys.stderr)

            # 個別 prompt ループ。audio はメモリ上にのみ存在。
            # g キー: 同 seed で次の test text を再合成して切り替え（voice character を多面評価）。
            # 採用タグ(m/f/c/e/k) または note 付きの時のみ canonical(texts[0]) を wav 保存。
            note = ""
            saved_wav = False
            text_idx = 0
            while True:
                play_audio(audio)
                # prompt は短く「今聴いた seed/text」だけ。help は ? で出す。
                prompt = f"seed={seed:3d} text[{text_idx+1}/{len(texts)}] > "
                try:
                    ans = input(prompt).strip().lower()
                except EOFError:
                    quit_now = True; break
                if not ans:
                    continue
                if ans == "?":
                    print(f"  {TAG_HELP}", file=sys.stderr); continue
                if ans == "r":
                    continue  # play 先頭に戻る（同じ audio をメモリから再生）
                if ans == "g":
                    if len(texts) <= 1:
                        print("  [g] texts が 1 文のみ。--texts で複数指定すると rotate できる",
                              file=sys.stderr); continue
                    text_idx = (text_idx + 1) % len(texts)
                    nxt = texts[text_idx]
                    print(f"  → seed={seed} text[{text_idx+1}/{len(texts)}]={nxt!r} synth...",
                          file=sys.stderr, flush=True)
                    with synth_lock:
                        audio, tv = pipeline.synthesize(nxt, seed=seed, **params)
                    print(f"  ✓ seed={seed} text[{text_idx+1}/{len(texts)}] "
                          f"t_valid={tv} amp±{float(np.abs(audio).max()):.3f}",
                          file=sys.stderr)
                    continue  # play 先頭に戻り新音声を再生
                if ans == "n":
                    try:
                        note = input("  note > ").strip()
                    except EOFError:
                        note = ""
                    continue  # note を取った後もう一度 tag を聞く
                if ans == "q":
                    quit_now = True; break
                if ans in TAG_VALID:
                    # 採用候補 or note 付きのみ wav を保存。テキスト依存性を排除するため
                    # 常に canonical(texts[0]) を保存（後から re-listen 時に一貫した比較が可能）。
                    if ans in TAG_KEEP_WAV or note:
                        wav_path = out_dir / f"seed_{seed:03d}.wav"
                        write_wav(str(wav_path), audio_canonical, 48000)
                        saved_wav = True
                    append_rating(ratings_path, seed, ans, note)
                    flag = " +wav" if saved_wav else ""
                    print(f"  saved: seed={seed} tag={ans} note={note!r}{flag}", file=sys.stderr)
                    break
                print(f"  unknown: {ans!r} (? で help)", file=sys.stderr)
    finally:
        stop_evt.set()
        # queue を drain（worker が put 中ならブロック解除）
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass

    # サマリ
    if ratings_path.exists():
        final = load_existing(ratings_path)
        tag_counts: dict[str, int] = {}
        for row in final.values():
            tag_counts[row["tag"]] = tag_counts.get(row["tag"], 0) + 1
        print("\n[summary]", file=sys.stderr)
        for t in ["m", "f", "c", "e", "k", "s"]:
            print(f"  {t}: {tag_counts.get(t, 0)}", file=sys.stderr)
        print(f"  total rated: {len(final)} / wav saved: {counted}", file=sys.stderr)
        print(f"  csv: {ratings_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
