"""
run_experiment.py
=================
フェーズ1 メイン実験スクリプト

実験内容:
    - shift_hz を sweep し、フォルマントシフト量を変化させる
    - 各 shift_hz に対して:
        1. WORLD でフォルマントシフトした音声を生成
        2. HuBERT でトークンIDを比較 → トークン変化率
        3. PESQ / STOI で音質スコアを計算
    - 結果を CSV と PNG（トレードオフ曲線）で保存

Usage:
    python run_experiment.py --input path/to/audio.wav [options]
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.io import wavfile

# ローカルモジュール
from formant_shift import analyze, synthesize, shift_all_frames
from hubert_token_analyzer import HubertTokenizer, compare_tokens, compute_hidden_distance

# 音質評価（要インストール: pip install pesq pystoi）
try:
    from pesq import pesq as compute_pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False
    warnings.warn("pesq が見つかりません。pip install pesq でインストールしてください。")

try:
    from pystoi import stoi as compute_stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False
    warnings.warn("pystoi が見つかりません。pip install pystoi でインストールしてください。")


# ─────────────────────────────────────────────
#  音質評価
# ─────────────────────────────────────────────

def evaluate_audio_quality(
    original_wav: np.ndarray,
    shifted_wav: np.ndarray,
    sr: int,
) -> dict[str, float]:
    """PESQ・STOI・SNR を計算して返す"""
    results = {}

    # SNR
    noise = shifted_wav - original_wav
    signal_power = np.mean(original_wav ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    results["snr_db"] = float(10 * np.log10(signal_power / noise_power))

    # 長さをそろえる
    n = min(len(original_wav), len(shifted_wav))
    ref = original_wav[:n].astype(np.float32)
    deg = shifted_wav[:n].astype(np.float32)

    if HAS_PESQ:
        try:
            # PESQ は 8000 or 16000 Hz のみ対応
            pesq_sr = 16000 if sr >= 16000 else 8000
            if sr != pesq_sr:
                import torchaudio
                ref_t = torch.from_numpy(ref).unsqueeze(0)
                deg_t = torch.from_numpy(deg).unsqueeze(0)
                ref_t = torchaudio.functional.resample(ref_t, sr, pesq_sr).squeeze().numpy()
                deg_t = torchaudio.functional.resample(deg_t, sr, pesq_sr).squeeze().numpy()
            else:
                ref_t, deg_t = ref, deg
            mode = "wb" if pesq_sr == 16000 else "nb"
            results["pesq"] = float(compute_pesq(pesq_sr, ref_t, deg_t, mode))
        except Exception as e:
            results["pesq"] = float("nan")
            warnings.warn(f"PESQ 計算失敗: {e}")

    if HAS_STOI:
        try:
            results["stoi"] = float(compute_stoi(ref, deg, sr, extended=False))
        except Exception as e:
            results["stoi"] = float("nan")
            warnings.warn(f"STOI 計算失敗: {e}")

    return results


# ─────────────────────────────────────────────
#  シングル shift_hz 実験
# ─────────────────────────────────────────────

def run_single(
    wav: np.ndarray,
    sr: int,
    shift_hz: float,
    tokenizer: HubertTokenizer,
    target_formants: list[int] | None,
    save_audio: bool = False,
    output_dir: Path = Path("./results"),
) -> dict:
    """
    1つの shift_hz に対して実験を実行し、結果 dict を返す。
    """
    t0 = time.time()

    # ── WORLD 分析 ──────────────────────────────
    feat = analyze(wav, sr)
    orig_wav = synthesize(feat)   # WORLD再合成（ベースライン）

    # ── フォルマントシフト ──────────────────────
    shifted_feat, orig_formants, shifted_formants = shift_all_frames(
        feat, shift_hz,
        target_formants=target_formants,
        voiced_only=True,
    )
    shifted_wav = synthesize(shifted_feat)

    # 長さ揃え
    n = min(len(orig_wav), len(shifted_wav))
    orig_wav_trim = orig_wav[:n]
    shifted_wav_trim = shifted_wav[:n]

    # ── HuBERT トークン比較 ─────────────────────
    orig_tokens = tokenizer.extract(orig_wav_trim, sr)
    shifted_tokens = tokenizer.extract(shifted_wav_trim, sr)
    diff = compare_tokens(orig_tokens, shifted_tokens, shift_hz)
    hidden_dist = compute_hidden_distance(orig_tokens, shifted_tokens)

    # ── 音質評価 ────────────────────────────────
    quality = evaluate_audio_quality(orig_wav_trim, shifted_wav_trim, sr)

    # ── フォルマント統計 ────────────────────────
    if len(orig_formants) > 0:
        f1_orig = np.mean([fi.peak_freqs[0] for fi in orig_formants if len(fi.peak_freqs) > 0])
        f2_orig = np.mean([fi.peak_freqs[1] for fi in orig_formants if len(fi.peak_freqs) > 1])
        f1_shft = np.mean([fi.peak_freqs[0] for fi in shifted_formants if len(fi.peak_freqs) > 0])
        f2_shft = np.mean([fi.peak_freqs[1] for fi in shifted_formants if len(fi.peak_freqs) > 1])
    else:
        f1_orig = f2_orig = f1_shft = f2_shft = float("nan")

    result = {
        "shift_hz": shift_hz,
        "token_change_rate": diff.change_rate,
        "token_changed_frames": diff.changed_frames,
        "token_total_frames": diff.total_frames,
        "hidden_cosine_dist_mean": float(np.mean(hidden_dist)),
        "hidden_cosine_dist_std": float(np.std(hidden_dist)),
        "f1_orig_hz": f1_orig,
        "f2_orig_hz": f2_orig,
        "f1_shifted_hz": f1_shft,
        "f2_shifted_hz": f2_shft,
        "actual_f1_shift": f1_shft - f1_orig,
        "actual_f2_shift": f2_shft - f2_orig,
        "elapsed_sec": time.time() - t0,
        **quality,
    }

    # ── 音声保存（オプション） ──────────────────
    if save_audio:
        out_path = output_dir / f"shifted_{shift_hz:+.0f}Hz.wav"
        sf.write(str(out_path), shifted_wav_trim, sr)

    return result


# ─────────────────────────────────────────────
#  Sweep 実験
# ─────────────────────────────────────────────

def run_sweep(
    wav: np.ndarray,
    sr: int,
    shift_range: list[float],
    tokenizer: HubertTokenizer,
    target_formants: list[int] | None,
    output_dir: Path,
    save_audio: bool = False,
) -> list[dict]:
    """全 shift_hz に対して実験を実行する"""
    results = []
    n_total = len(shift_range)

    for i, shift_hz in enumerate(shift_range):
        print(f"[{i+1:3d}/{n_total}] shift_hz={shift_hz:+.1f} Hz ...", end=" ", flush=True)
        try:
            r = run_single(
                wav, sr, shift_hz, tokenizer,
                target_formants=target_formants,
                save_audio=save_audio,
                output_dir=output_dir,
            )
            results.append(r)
            print(
                f"token_change={r['token_change_rate']:.3f}"
                + (f"  PESQ={r['pesq']:.3f}" if "pesq" in r else "")
                + (f"  STOI={r['stoi']:.3f}" if "stoi" in r else "")
                + f"  SNR={r['snr_db']:.1f}dB"
                + f"  ({r['elapsed_sec']:.1f}s)"
            )
        except Exception as e:
            print(f"ERROR: {e}")
            warnings.warn(f"shift_hz={shift_hz} で失敗しました: {e}")

    return results


# ─────────────────────────────────────────────
#  可視化
# ─────────────────────────────────────────────

def plot_results(results: list[dict], output_dir: Path) -> None:
    """トレードオフ曲線を描画して保存する"""
    shift_list = [r["shift_hz"] for r in results]
    token_cr = [r["token_change_rate"] for r in results]
    snr = [r["snr_db"] for r in results]
    pesq_list = [r.get("pesq", float("nan")) for r in results]
    stoi_list = [r.get("stoi", float("nan")) for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Formant Shift vs. HuBERT Token Change & Audio Quality", fontsize=14, y=1.01)

    # ─ 1. トークン変化率 vs シフト量 ─
    ax = axes[0, 0]
    ax.plot(shift_list, token_cr, "o-", color="#2196F3", lw=2, ms=5)
    ax.axhline(0.5, color="red", ls="--", lw=1, label="50% 変化ライン")
    ax.set_xlabel("Formant Shift (Hz)")
    ax.set_ylabel("Token Change Rate")
    ax.set_title("Token Change Rate vs Shift")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # ─ 2. SNR vs シフト量 ─
    ax = axes[0, 1]
    ax.plot(shift_list, snr, "s-", color="#4CAF50", lw=2, ms=5)
    ax.set_xlabel("Formant Shift (Hz)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Signal-to-Noise Ratio vs Shift")
    ax.grid(alpha=0.3)

    # ─ 3. PESQ vs シフト量 ─
    ax = axes[1, 0]
    if not all(np.isnan(pesq_list)):
        ax.plot(shift_list, pesq_list, "D-", color="#FF9800", lw=2, ms=5)
        ax.set_ylabel("PESQ (MOS-LQO)")
        ax.set_ylim([1, 4.5])
    else:
        ax.text(0.5, 0.5, "PESQ not available\n(pip install pesq)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Formant Shift (Hz)")
    ax.set_title("PESQ vs Shift")
    ax.grid(alpha=0.3)

    # ─ 4. トレードオフ散布図：Token Change Rate vs PESQ ─
    ax = axes[1, 1]
    if not all(np.isnan(pesq_list)):
        sc = ax.scatter(
            token_cr, pesq_list,
            c=shift_list, cmap="RdYlGn_r", s=60, edgecolors="k", lw=0.5
        )
        plt.colorbar(sc, ax=ax, label="Shift (Hz)")
        ax.set_xlabel("Token Change Rate")
        ax.set_ylabel("PESQ")
        ax.set_title("Trade-off: Token Flip vs Audio Quality")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "STOI not available\n(pip install pystoi)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)

    plt.tight_layout()
    save_path = output_dir / "tradeoff_curve.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] 保存: {save_path}")


def save_csv(results: list[dict], output_dir: Path) -> None:
    """結果を CSV に保存する"""
    import csv
    if not results:
        return
    keys = list(results[0].keys())
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"[CSV] 保存: {csv_path}")


# ─────────────────────────────────────────────
#  エントリーポイント
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="フォルマントシフト × HuBERT トークン変化率 実験スクリプト"
    )
    p.add_argument("--input", required=True, help="入力音声ファイル (.wav)")
    p.add_argument("--output_dir", default="./results",
                   help="結果の出力ディレクトリ (default: ./results)")

    # シフト sweep の設定
    p.add_argument("--shift_min", type=float, default=-100.0,
                   help="シフト量の最小値 Hz (default: -100)")
    p.add_argument("--shift_max", type=float, default=100.0,
                   help="シフト量の最大値 Hz (default: +100)")
    p.add_argument("--shift_step", type=float, default=5.0,
                   help="シフト量のステップ Hz (default: 5)")

    # フォルマント設定
    p.add_argument("--target_formants", type=int, nargs="+", default=None,
                   help="シフト対象フォルマント番号 (0=F1, 1=F2, ...) 未指定=全て")

    # HuBERT 設定
    p.add_argument("--hubert_model", default="rinna/japanese-hubert-base",
                   help="HuBERT モデル名 (HuggingFace)")
    p.add_argument("--hubert_layer", type=int, default=9,
                   help="使用する HuBERT 層インデックス (default: 9)")

    # その他
    p.add_argument("--save_audio", action="store_true",
                   help="各 shift_hz のシフト済み音声を保存する")
    p.add_argument("--device", default=None,
                   help="'cuda' or 'cpu' (default: 自動検出)")

    return p


def main():
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 音声読み込み ────────────────────────────
    print(f"[Input] {args.input}")
    wav, sr = sf.read(args.input)
    if wav.ndim > 1:
        wav = wav[:, 0]  # モノラル化
    wav = wav.astype(np.float64)
    print(f"  サンプリングレート: {sr} Hz | 長さ: {len(wav)/sr:.2f} s")

    # ── Sweep 範囲の構築 ──────────────────────
    shift_range = list(np.arange(
        args.shift_min, args.shift_max + args.shift_step * 0.5,
        args.shift_step
    ))
    print(f"\n[Sweep] {len(shift_range)} 点: {args.shift_min:+.0f} ~ {args.shift_max:+.0f} Hz (step={args.shift_step:.0f}Hz)")
    if args.target_formants:
        print(f"  対象フォルマント: F{[i+1 for i in args.target_formants]}")
    else:
        print("  対象フォルマント: 全て")

    # ── HuBERT 初期化 ─────────────────────────
    tokenizer = HubertTokenizer(
        model_name=args.hubert_model,
        device=args.device,
        output_layer=args.hubert_layer,
    )

    # ── 実験実行 ──────────────────────────────
    print("\n[Experiment] 開始 ...\n")
    results = run_sweep(
        wav, sr,
        shift_range=shift_range,
        tokenizer=tokenizer,
        target_formants=args.target_formants,
        output_dir=output_dir,
        save_audio=args.save_audio,
    )

    # ── 結果保存 ──────────────────────────────
    save_csv(results, output_dir)
    plot_results(results, output_dir)

    # サマリー JSON
    summary = {
        "input": str(args.input),
        "n_sweep": len(results),
        "shift_range_hz": [args.shift_min, args.shift_max],
        "hubert_model": args.hubert_model,
        "hubert_layer": args.hubert_layer,
        "target_formants": args.target_formants,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── コンソールサマリー ─────────────────────
    print("\n" + "="*60)
    print("実験完了サマリー")
    print("="*60)

    # トークン変化率 > 50% になる最小シフト量を探す
    positive_results = [r for r in results if r["shift_hz"] > 0]
    threshold_results = [r for r in positive_results if r["token_change_rate"] >= 0.5]
    if threshold_results:
        min_50pct = min(threshold_results, key=lambda r: r["shift_hz"])
        print(f"\n  ▶ トークン変化率 ≥ 50% となる最小シフト: {min_50pct['shift_hz']:+.1f} Hz")
        if "pesq" in min_50pct and not np.isnan(min_50pct["pesq"]):
            print(f"    その時の PESQ: {min_50pct['pesq']:.3f}")
        print(f"    その時の SNR: {min_50pct['snr_db']:.1f} dB")
    else:
        print("  ▶ sweep 範囲内でトークン変化率 50% に達しませんでした。")
        print("    --shift_max を大きくするか、対象フォルマントを絞ってみてください。")

    print(f"\n出力: {output_dir.resolve()}/")
    print("  - results.csv        : 全 shift_hz の数値データ")
    print("  - tradeoff_curve.png : トレードオフ曲線")
    print("  - config.json        : 実験設定")
    if args.save_audio:
        print("  - shifted_*.wav      : 各 shift_hz のシフト済み音声")


if __name__ == "__main__":
    main()
