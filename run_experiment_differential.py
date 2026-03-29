"""
run_experiment_differential.py
===============================
第2ラウンド 実験C：F1・F2 逆方向シフト実験

F1を+X Hz、F2を-X Hz（またはその逆）にシフトすることで
F1-F2間の距離を意図的に変化させ、HuBERTのトークン反転率を測定する。

これは「全体的な声道スケーリング」ではなく
「母音の音響的アイデンティティの破壊」を狙った実験。

Usage:
    python run_experiment_differential.py \
        --input audio.wav \
        --output_dir ./results/differential \
        --shift_min 0 --shift_max 150 --shift_step 5
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from formant_shift import analyze, synthesize, WorldFeatures
from formant_shift import shift_formants_spline, detect_formants
from hubert_token_analyzer import HubertTokenizer, compare_tokens, compute_hidden_distance

try:
    from pesq import pesq as compute_pesq
    HAS_PESQ = True
except ImportError:
    HAS_PESQ = False

try:
    from pystoi import stoi as compute_stoi
    HAS_STOI = True
except ImportError:
    HAS_STOI = False


# ─────────────────────────────────────────────
#  逆方向シフト処理
# ─────────────────────────────────────────────

def shift_differential(
    feat: WorldFeatures,
    f1_shift_hz: float,
    f2_shift_hz: float,
    voiced_only: bool = True,
) -> WorldFeatures:
    """
    F1とF2を独立した量でシフトする。

    Args:
        feat: WORLD特徴量
        f1_shift_hz: F1のシフト量（正=上方、負=下方）
        f2_shift_hz: F2のシフト量（正=上方、負=下方）
        voiced_only: 有声フレームのみ処理

    Returns:
        シフト後のWorldFeatures
    """
    sp_shifted = feat.sp.copy()

    for i in range(len(feat.f0)):
        if voiced_only and feat.f0[i] < 1.0:
            continue

        sp_frame = feat.sp[i]

        # ── F1のみシフト ──
        sp_f1, _, _ = shift_formants_spline(
            sp_frame, feat.sr,
            shift_hz=f1_shift_hz,
            target_formants=[0],   # F1のみ
        )

        # ── F2のみシフト（F1シフト後のスペクトルに適用） ──
        sp_f1f2, _, _ = shift_formants_spline(
            sp_f1, feat.sr,
            shift_hz=f2_shift_hz,
            target_formants=[1],   # F2のみ
        )

        sp_shifted[i] = sp_f1f2

    return WorldFeatures(
        f0=feat.f0.copy(),
        sp=sp_shifted,
        ap=feat.ap.copy(),
        sr=feat.sr,
        frame_period=feat.frame_period,
    )


def evaluate_quality(orig: np.ndarray, shifted: np.ndarray, sr: int) -> dict:
    n = min(len(orig), len(shifted))
    ref = orig[:n].astype(np.float32)
    deg = shifted[:n].astype(np.float32)

    noise = deg - ref
    snr = 10 * np.log10(np.mean(ref**2) / (np.mean(noise**2) + 1e-12) + 1e-12)
    result = {"snr_db": float(snr)}

    if HAS_PESQ:
        try:
            import torchaudio, torch
            pesq_sr = 16000 if sr >= 16000 else 8000
            if sr != pesq_sr:
                r = torchaudio.functional.resample(torch.from_numpy(ref).unsqueeze(0), sr, pesq_sr).squeeze().numpy()
                d = torchaudio.functional.resample(torch.from_numpy(deg).unsqueeze(0), sr, pesq_sr).squeeze().numpy()
            else:
                r, d = ref, deg
            result["pesq"] = float(compute_pesq(pesq_sr, r, d, "wb"))
        except Exception as e:
            result["pesq"] = float("nan")

    if HAS_STOI:
        try:
            result["stoi"] = float(compute_stoi(ref, deg, sr, extended=False))
        except Exception as e:
            result["stoi"] = float("nan")

    return result


# ─────────────────────────────────────────────
#  可視化
# ─────────────────────────────────────────────

def plot_differential_results(
    results_diff: list[dict],
    results_f1: list[dict],
    results_f2: list[dict],
    output_dir: Path,
) -> None:
    """3実験の比較プロットを生成する"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "F1/F2 Differential Shift: HuBERT Token Change Rate Comparison",
        fontsize=13, y=1.01
    )

    # ── 共通データ抽出ヘルパー ──
    def extract(results, key):
        return [r[key] for r in results]

    shift_vals_diff = extract(results_diff, "shift_magnitude")
    shift_vals_f1   = extract(results_f1,   "shift_hz")
    shift_vals_f2   = extract(results_f2,   "shift_hz")

    # ─ 1. トークン変化率の比較 ─
    ax = axes[0, 0]
    ax.plot(shift_vals_diff, extract(results_diff, "token_change_rate"),
            "o-", color="#E91E63", lw=2, ms=5, label="C: F1↑ F2↓ (differential)")
    # F1・F2は正方向のみ比較
    f1_pos = [(r["shift_hz"], r["token_change_rate"]) for r in results_f1 if r["shift_hz"] >= 0]
    f2_pos = [(r["shift_hz"], r["token_change_rate"]) for r in results_f2 if r["shift_hz"] >= 0]
    if f1_pos:
        ax.plot(*zip(*f1_pos), "s--", color="#2196F3", lw=1.5, ms=4, label="A: F1 only (+)")
    if f2_pos:
        ax.plot(*zip(*f2_pos), "^--", color="#4CAF50", lw=1.5, ms=4, label="B: F2 only (+)")
    ax.axhline(0.5, color="red", ls=":", lw=1, alpha=0.7, label="50% 目標")
    ax.set_xlabel("Shift Magnitude (Hz)")
    ax.set_ylabel("Token Change Rate")
    ax.set_title("Token Change Rate: 3-Way Comparison")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    # ─ 2. PESQ 比較 ─
    ax = axes[0, 1]
    pesq_diff = extract(results_diff, "pesq")
    pesq_f1   = [r["pesq"] for r in results_f1 if r["shift_hz"] >= 0]
    pesq_f2   = [r["pesq"] for r in results_f2 if r["shift_hz"] >= 0]
    if not all(np.isnan(pesq_diff)):
        ax.plot(shift_vals_diff, pesq_diff,
                "o-", color="#E91E63", lw=2, ms=5, label="C: differential")
    if f1_pos and not all(np.isnan(pesq_f1)):
        ax.plot([x for x, _ in f1_pos], pesq_f1,
                "s--", color="#2196F3", lw=1.5, ms=4, label="A: F1 only")
    if f2_pos and not all(np.isnan(pesq_f2)):
        ax.plot([x for x, _ in f2_pos], pesq_f2,
                "^--", color="#4CAF50", lw=1.5, ms=4, label="B: F2 only")
    ax.set_xlabel("Shift Magnitude (Hz)")
    ax.set_ylabel("PESQ (MOS-LQO)")
    ax.set_title("Audio Quality: PESQ Comparison")
    ax.set_ylim([3.5, 4.7])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ─ 3. F1-F2距離の変化（differential実験のみ） ─
    ax = axes[1, 0]
    f1f2_orig = [r["f1f2_distance_orig"] for r in results_diff]
    f1f2_shft = [r["f1f2_distance_shifted"] for r in results_diff]
    ax.plot(shift_vals_diff, f1f2_orig, "k--", lw=1.5, label="Original F2-F1 distance")
    ax.plot(shift_vals_diff, f1f2_shft, "o-", color="#E91E63", lw=2, ms=5,
            label="Shifted F2-F1 distance")
    ax.set_xlabel("Shift Magnitude (Hz)")
    ax.set_ylabel("F2 - F1 Distance (Hz)")
    ax.set_title("F1-F2 Distance Change (Experiment C)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ─ 4. トレードオフ散布図（differential） ─
    ax = axes[1, 1]
    if not all(np.isnan(pesq_diff)):
        sc = ax.scatter(
            extract(results_diff, "token_change_rate"),
            pesq_diff,
            c=shift_vals_diff, cmap="plasma", s=70,
            edgecolors="k", lw=0.5
        )
        plt.colorbar(sc, ax=ax, label="Shift Magnitude (Hz)")
        ax.set_xlabel("Token Change Rate")
        ax.set_ylabel("PESQ")
        ax.set_title("Trade-off: Experiment C (F1↑ F2↓)")
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = output_dir / "differential_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] 保存: {out}")


def save_csv(results: list[dict], path: Path) -> None:
    import csv
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[CSV] 保存: {path}")


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="F1/F2 逆方向差分シフト実験（実験C）"
    )
    p.add_argument("--input", required=True)
    p.add_argument("--output_dir", default="./results/differential")
    p.add_argument("--shift_min", type=float, default=0.0,
                   help="シフト幅の最小値 Hz (default: 0)")
    p.add_argument("--shift_max", type=float, default=150.0,
                   help="シフト幅の最大値 Hz (default: 150)")
    p.add_argument("--shift_step", type=float, default=5.0)
    p.add_argument("--hubert_model", default="facebook/hubert-base-ls960")
    p.add_argument("--hubert_layer", type=int, default=9)
    p.add_argument("--device", default=None)
    p.add_argument("--f1only_csv", default=None,
                   help="実験Aの results.csv パス（比較プロット用）")
    p.add_argument("--f2only_csv", default=None,
                   help="実験Bの results.csv パス（比較プロット用）")
    return p


def load_csv(path: str) -> list[dict]:
    import csv
    with open(path) as f:
        return [
            {k: float(v) if v not in ("", "nan") else float("nan")
             for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 音声読み込み
    print(f"[Input] {args.input}")
    wav, sr = sf.read(args.input)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav.astype(np.float64)
    print(f"  SR: {sr} Hz | Length: {len(wav)/sr:.2f} s")

    # HuBERT 初期化
    tokenizer = HubertTokenizer(
        model_name=args.hubert_model,
        device=args.device,
        output_layer=args.hubert_layer,
    )

    # WORLD 分析（1回だけ）
    from formant_shift import analyze, synthesize
    feat = analyze(wav, sr)
    orig_wav = synthesize(feat)
    orig_tokens = tokenizer.extract(orig_wav, sr)

    # Sweep
    shift_range = np.arange(
        args.shift_min,
        args.shift_max + args.shift_step * 0.5,
        args.shift_step
    )
    results = []
    n = len(shift_range)

    print(f"\n[Experiment C] F1↑ / F2↓ 逆方向シフト: {n} 点\n")

    for i, mag in enumerate(shift_range):
        print(f"[{i+1:3d}/{n}] F1={mag:+.0f}Hz, F2={-mag:+.0f}Hz ...", end=" ", flush=True)

        try:
            shifted_feat = shift_differential(
                feat, f1_shift_hz=float(mag), f2_shift_hz=float(-mag)
            )
            shifted_wav = synthesize(shifted_feat)

            n_frames = min(len(orig_wav), len(shifted_wav))
            orig_trim = orig_wav[:n_frames]
            shft_trim = shifted_wav[:n_frames]

            shft_tokens = tokenizer.extract(shft_trim, sr)
            diff = compare_tokens(orig_tokens, shft_tokens, shift_hz=mag)
            quality = evaluate_quality(orig_trim, shft_trim, sr)

            # F1-F2 距離の計算
            voiced_frames = [i for i in range(len(feat.f0)) if feat.f0[i] >= 1.0]
            f1f2_orig_list, f1f2_shft_list = [], []
            for fi in voiced_frames[:20]:  # 最初の20有声フレームで代表
                pf_o, _, _ = detect_formants(feat.sp[fi], sr)
                pf_s, _, _ = detect_formants(shifted_feat.sp[fi], sr)
                if len(pf_o) >= 2:
                    f1f2_orig_list.append(pf_o[1] - pf_o[0])
                if len(pf_s) >= 2:
                    f1f2_shft_list.append(pf_s[1] - pf_s[0])

            result = {
                "shift_magnitude": float(mag),
                "f1_shift_hz": float(mag),
                "f2_shift_hz": float(-mag),
                "token_change_rate": diff.change_rate,
                "token_changed_frames": diff.changed_frames,
                "token_total_frames": diff.total_frames,
                "hidden_cosine_dist_mean": float(np.mean(compute_hidden_distance(orig_tokens, shft_tokens))),
                "f1f2_distance_orig": float(np.mean(f1f2_orig_list)) if f1f2_orig_list else float("nan"),
                "f1f2_distance_shifted": float(np.mean(f1f2_shft_list)) if f1f2_shft_list else float("nan"),
                **quality,
            }
            results.append(result)

            print(
                f"token_change={result['token_change_rate']:.3f}"
                + (f"  PESQ={result['pesq']:.3f}" if "pesq" in result else "")
                + f"  SNR={result['snr_db']:.1f}dB"
                + f"  F1-F2_dist: {result['f1f2_distance_orig']:.0f}→{result['f1f2_distance_shifted']:.0f}Hz"
            )

        except Exception as e:
            print(f"ERROR: {e}")
            warnings.warn(str(e))

    # 保存
    save_csv(results, output_dir / "results_differential.csv")

    # 比較プロット（A・Bの CSV があれば重ねて描画）
    results_f1 = load_csv(args.f1only_csv) if args.f1only_csv else []
    results_f2 = load_csv(args.f2only_csv) if args.f2only_csv else []
    plot_differential_results(results, results_f1, results_f2, output_dir)

    # サマリー
    print("\n" + "="*60)
    best = max(results, key=lambda r: r["token_change_rate"])
    print(f"最高トークン変化率: {best['token_change_rate']:.1%} @ F1={best['f1_shift_hz']:+.0f}Hz / F2={best['f2_shift_hz']:+.0f}Hz")
    if "pesq" in best:
        print(f"  その時の PESQ: {best['pesq']:.3f}")
    print(f"  その時の SNR: {best['snr_db']:.1f} dB")
    print(f"\n出力: {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
