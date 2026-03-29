"""
formant_shift.py
================
フォルマントシフト処理モジュール（フェーズ1コア）

WORLDボコーダを用いてスペクトル包絡のピーク（フォルマント）を検出し、
スプライン補間により数Hz単位で精密にシフトする。

依存: pyworld, numpy, scipy
"""

import numpy as np
import pyworld as pw
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from dataclasses import dataclass
from typing import Optional
import warnings


# ─────────────────────────────────────────────
#  データクラス
# ─────────────────────────────────────────────

@dataclass
class WorldFeatures:
    """WORLDで抽出した音声特徴量"""
    f0: np.ndarray          # 基本周波数 [T]
    sp: np.ndarray          # スペクトル包絡 [T, fft_size//2+1]
    ap: np.ndarray          # 非周期性指標 [T, fft_size//2+1]
    sr: int                 # サンプリングレート
    frame_period: float     # フレーム周期 (ms)

@dataclass
class FormantInfo:
    """検出されたフォルマント情報（1フレーム分）"""
    frame_idx: int
    peak_freqs: np.ndarray   # ピーク周波数 [Hz]
    peak_indices: np.ndarray # スペクトル包絡上のインデックス
    peak_amplitudes: np.ndarray  # ピーク振幅 [dB]


# ─────────────────────────────────────────────
#  WORLD 分析・合成
# ─────────────────────────────────────────────

def analyze(wav: np.ndarray, sr: int, frame_period: float = 5.0) -> WorldFeatures:
    """
    音声をWORLDで分解する。

    Args:
        wav: 音声波形 (float64, -1~1 正規化済み)
        sr: サンプリングレート (Hz)
        frame_period: フレーム周期 (ms), デフォルト5ms

    Returns:
        WorldFeatures
    """
    wav = wav.astype(np.float64)
    f0, t = pw.dio(wav, sr, frame_period=frame_period)
    f0 = pw.stonemask(wav, f0, t, sr)
    sp = pw.cheaptrick(wav, f0, t, sr)
    ap = pw.d4c(wav, f0, t, sr)
    return WorldFeatures(f0=f0, sp=sp, ap=ap, sr=sr, frame_period=frame_period)


def synthesize(feat: WorldFeatures) -> np.ndarray:
    """
    WorldFeaturesから音声波形を再合成する。

    Returns:
        wav: float64 波形
    """
    return pw.synthesize(
        feat.f0, feat.sp, feat.ap, feat.sr,
        frame_period=feat.frame_period
    )


# ─────────────────────────────────────────────
#  フォルマント検出
# ─────────────────────────────────────────────

def detect_formants(
    sp_frame: np.ndarray,
    sr: int,
    n_formants: int = 4,
    freq_min: float = 200.0,
    freq_max: float = 4000.0,
    prominence: float = 3.0,
    distance_hz: float = 150.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スペクトル包絡の1フレームからフォルマントを検出する。

    Args:
        sp_frame: スペクトル包絡 1フレーム分 [fft_size//2+1]
        sr: サンプリングレート
        n_formants: 検出する最大フォルマント数
        freq_min: 探索周波数下限 (Hz)
        freq_max: 探索周波数上限 (Hz)
        prominence: ピーク突出度の閾値 (dB)
        distance_hz: ピーク間の最小周波数間隔 (Hz)

    Returns:
        peak_freqs [Hz], peak_indices, peak_amplitudes [dB]
    """
    fft_size = (len(sp_frame) - 1) * 2
    freqs = np.linspace(0, sr / 2, len(sp_frame))

    # dBスケールで処理（ピーク検出が安定する）
    sp_db = 10 * np.log10(sp_frame + 1e-12)

    # 探索範囲をマスク
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    indices_in_range = np.where(mask)[0]

    # Hz → インデックス変換での最小距離
    hz_per_bin = sr / fft_size
    distance_bins = max(1, int(distance_hz / hz_per_bin))

    peaks_local, props = find_peaks(
        sp_db[mask],
        prominence=prominence,
        distance=distance_bins,
    )

    if len(peaks_local) == 0:
        return np.array([]), np.array([]), np.array([])

    # 突出度の高い順に n_formants 個選択
    order = np.argsort(props["prominences"])[::-1][:n_formants]
    peaks_local = np.sort(peaks_local[order])  # 周波数昇順に戻す

    peak_indices = indices_in_range[peaks_local]
    peak_freqs = freqs[peak_indices]
    peak_amplitudes = sp_db[peak_indices]

    return peak_freqs, peak_indices, peak_amplitudes


# ─────────────────────────────────────────────
#  スプライン補間によるフォルマントシフト
# ─────────────────────────────────────────────

def shift_formants_spline(
    sp_frame: np.ndarray,
    sr: int,
    shift_hz: float,
    target_formants: Optional[list[int]] = None,
    n_formants: int = 4,
    freq_min: float = 200.0,
    freq_max: float = 4000.0,
    **detect_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スプライン補間を用いてフォルマントをシフトする（1フレーム）。

    戦略:
        1. ピーク検出でフォルマント位置を特定
        2. シフト後の位置へ向けてスペクトル包絡をリマッピング
        3. 低域（< freq_min）・高域（> freq_max）はアンカーとして固定
           → 滑らかな補間が保証される

    Args:
        sp_frame: スペクトル包絡 1フレーム [fft_size//2+1]
        sr: サンプリングレート
        shift_hz: シフト量 (Hz)。正=上方シフト、負=下方シフト
        target_formants: シフト対象フォルマントのインデックスリスト (0-based)
                         None なら全フォルマントをシフト
        n_formants: 検出フォルマント数
        freq_min / freq_max: フォルマント探索範囲

    Returns:
        shifted_sp: シフト後スペクトル包絡
        original_peak_freqs: シフト前ピーク周波数
        shifted_peak_freqs: シフト後ピーク周波数
    """
    fft_bins = len(sp_frame)
    freqs = np.linspace(0, sr / 2, fft_bins)

    peak_freqs, peak_indices, _ = detect_formants(
        sp_frame, sr,
        n_formants=n_formants,
        freq_min=freq_min,
        freq_max=freq_max,
        **detect_kwargs,
    )

    if len(peak_freqs) == 0:
        return sp_frame.copy(), np.array([]), np.array([])

    # シフト対象の選択
    if target_formants is not None:
        mask = np.zeros(len(peak_freqs), dtype=bool)
        for i in target_formants:
            if 0 <= i < len(peak_freqs):
                mask[i] = True
    else:
        mask = np.ones(len(peak_freqs), dtype=bool)

    # ── アンカー点の構築 ──────────────────────────
    # 「元の周波数 → シフト後の周波数」のマッピングを定義し、
    # それをスプラインで補間することでスペクトルをワープする。
    #
    # アンカー点:
    #   - 0 Hz (DC) : 固定
    #   - フォルマント領域より低い帯域の代表点: 固定
    #   - 各フォルマント: target なら +shift_hz、そうでなければ固定
    #   - フォルマント領域より高い帯域の代表点: 固定
    #   - Nyquist: 固定

    src_anchors = [0.0]
    dst_anchors = [0.0]

    # 低域アンカー（freq_min の少し下）
    low_anchor = max(freq_min * 0.5, 50.0)
    src_anchors.append(low_anchor)
    dst_anchors.append(low_anchor)

    # フォルマントアンカー
    shifted_peak_freqs = peak_freqs.copy()
    for i, (pf, do_shift) in enumerate(zip(peak_freqs, mask)):
        new_freq = float(np.clip(pf + shift_hz, freq_min * 0.8, freq_max * 1.2)) \
            if do_shift else float(pf)
        shifted_peak_freqs[i] = new_freq
        src_anchors.append(float(pf))
        dst_anchors.append(new_freq)

    # 高域アンカー（freq_max の少し上）
    high_anchor = min(freq_max * 1.3, sr / 2 * 0.95)
    src_anchors.append(high_anchor)
    dst_anchors.append(high_anchor)

    # Nyquist アンカー
    nyquist = sr / 2
    src_anchors.append(nyquist)
    dst_anchors.append(nyquist)

    # 重複・単調性を保証してソート
    src_anchors = np.array(src_anchors)
    dst_anchors = np.array(dst_anchors)
    order = np.argsort(src_anchors)
    src_anchors = src_anchors[order]
    dst_anchors = dst_anchors[order]

    # 重複する src を除去
    _, unique_idx = np.unique(src_anchors, return_index=True)
    src_anchors = src_anchors[unique_idx]
    dst_anchors = dst_anchors[unique_idx]

    # dst が単調増加でない場合は警告して非シフトを返す
    if not np.all(np.diff(dst_anchors) > 0):
        warnings.warn(
            f"shift_hz={shift_hz:.1f}Hz でマッピングが単調でなくなります。"
            "シフト量を小さくするか、対象フォルマントを絞ってください。",
            RuntimeWarning,
        )
        return sp_frame.copy(), peak_freqs, peak_freqs

    # ── スプライン補間でスペクトルをリマッピング ──
    # dst→src の逆マッピングを作り、「シフト後のグリッドで元のスペクトルを参照」する
    inv_spline = CubicSpline(dst_anchors, src_anchors, extrapolate=True)

    # 出力グリッド（シフト後周波数軸）に対応する元周波数を計算
    src_freqs_for_output = inv_spline(freqs)
    src_freqs_for_output = np.clip(src_freqs_for_output, 0, nyquist)

    # 元スペクトルを線形補間でリサンプル（dBで処理すると安定）
    sp_db = 10 * np.log10(sp_frame + 1e-12)
    sp_interp = np.interp(src_freqs_for_output, freqs, sp_db)
    shifted_sp = 10 ** (sp_interp / 10)

    return shifted_sp, peak_freqs, shifted_peak_freqs


# ─────────────────────────────────────────────
#  バッチ処理（全フレーム）
# ─────────────────────────────────────────────

def shift_all_frames(
    feat: WorldFeatures,
    shift_hz: float,
    target_formants: Optional[list[int]] = None,
    voiced_only: bool = True,
    **shift_kwargs,
) -> tuple[WorldFeatures, list[FormantInfo], list[FormantInfo]]:
    """
    全フレームに対してフォルマントシフトを適用する。

    Args:
        feat: WORLD特徴量
        shift_hz: シフト量 (Hz)
        target_formants: シフト対象フォルマントインデックス (None=全て)
        voiced_only: True なら有声フレームのみシフト

    Returns:
        shifted_feat: シフト後WorldFeatures
        original_formants: 各フレームの元フォルマント情報
        shifted_formants: 各フレームのシフト後フォルマント情報
    """
    sp_shifted = feat.sp.copy()
    original_formants = []
    shifted_formants = []

    for i in range(len(feat.f0)):
        # 無声フレームはスキップ（オプション）
        if voiced_only and feat.f0[i] < 1.0:
            continue

        shifted_sp, orig_pf, shift_pf = shift_formants_spline(
            feat.sp[i], feat.sr,
            shift_hz=shift_hz,
            target_formants=target_formants,
            **shift_kwargs,
        )
        sp_shifted[i] = shifted_sp

        if len(orig_pf) > 0:
            _, orig_idx, orig_amp = detect_formants(feat.sp[i], feat.sr)
            original_formants.append(FormantInfo(
                frame_idx=i,
                peak_freqs=orig_pf,
                peak_indices=orig_idx,
                peak_amplitudes=orig_amp,
            ))
            shifted_formants.append(FormantInfo(
                frame_idx=i,
                peak_freqs=shift_pf,
                peak_indices=np.array([]),   # シフト後は概算
                peak_amplitudes=np.array([]),
            ))

    shifted_feat = WorldFeatures(
        f0=feat.f0.copy(),
        sp=sp_shifted,
        ap=feat.ap.copy(),
        sr=feat.sr,
        frame_period=feat.frame_period,
    )
    return shifted_feat, original_formants, shifted_formants
