"""
hubert_token_analyzer.py
========================
HuBERT 離散トークン分析モジュール（フェーズ1コア）

シフト前後の音声から HuBERT 離散トークンを抽出し、
フレーム単位でトークンIDの変化を比較する。

依存: torch, torchaudio, transformers
"""

from __future__ import annotations
import numpy as np
import torch
import torchaudio
from transformers import HubertModel, AutoProcessor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings


# ─────────────────────────────────────────────
#  設定・定数
# ─────────────────────────────────────────────

HUBERT_SR = 16000          # HuBERT の想定サンプリングレート
DEFAULT_MODEL = "rinna/japanese-hubert-base"   # 日本語HuBERT
# 英語 / 汎用の場合: "facebook/hubert-base-ls960"


# ─────────────────────────────────────────────
#  データクラス
# ─────────────────────────────────────────────

@dataclass
class TokenSequence:
    """HuBERT から得たトークン系列と中間表現"""
    token_ids: np.ndarray        # 離散トークンID [T_hubert]
    hidden_states: np.ndarray    # 最終層の連続表現 [T_hubert, D]
    layer_hidden: dict           # 各層の隠れ状態 (layer_idx -> [T, D])

@dataclass
class TokenDiff:
    """シフト前後のトークン比較結果"""
    shift_hz: float
    total_frames: int
    changed_frames: int
    change_rate: float                    # changed / total
    changed_indices: np.ndarray           # 変化したフレームのインデックス
    original_ids: np.ndarray
    shifted_ids: np.ndarray
    # フォルマント別の変化率（F1, F2, F3...に対応するフレームの変化率）
    # フェーズ2以降で利用
    per_formant_change: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
#  HuBERT ラッパー
# ─────────────────────────────────────────────

class HubertTokenizer:
    """
    HuBERT モデルをラップし、音声から離散トークンを抽出するクラス。

    k-means クラスタリングによる量子化は soft-VC 等が使う手法だが、
    ここでは「各フレームで最も活性化するユニット（argmax）」を
    擬似的なトークンIDとして使用する。
    これはフレーム単位の変化検出に十分な近似精度を持つ。

    本番（フェーズ3）では SoftVC の事前学習済み k-means モデルを
    使うことを推奨する。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        output_layer: int = 9,    # SoftVC は第9層を使用
        use_kmeans: bool = False,  # True にすると k-means クラスタを使う
        kmeans_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_layer = output_layer
        self.use_kmeans = use_kmeans

        print(f"[HubertTokenizer] Loading {model_name} on {self.device} ...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        ).to(self.device).eval()

        # k-means モデルの読み込み（オプション）
        self.kmeans = None
        if use_kmeans and kmeans_path:
            import joblib
            self.kmeans = joblib.load(kmeans_path)
            print(f"[HubertTokenizer] k-means loaded from {kmeans_path}")

        print("[HubertTokenizer] Ready.")

    @torch.no_grad()
    def extract(
        self,
        wav: np.ndarray,
        sr: int,
        return_all_layers: bool = False,
    ) -> TokenSequence:
        """
        音声から HuBERT トークン系列を抽出する。

        Args:
            wav: 音声波形 (float64 or float32)
            sr: サンプリングレート
            return_all_layers: True なら全層の隠れ状態を返す

        Returns:
            TokenSequence
        """
        # リサンプリング
        wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        if sr != HUBERT_SR:
            wav_t = torchaudio.functional.resample(wav_t, sr, HUBERT_SR)

        # 正規化
        wav_t = wav_t / (wav_t.abs().max() + 1e-8)

        # processor (padding / attention_mask)
        inputs = self.processor(
            wav_t.squeeze().numpy(),
            sampling_rate=HUBERT_SR,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self.device)

        outputs = self.model(
            input_values,
            output_hidden_states=True,
        )

        # 指定層の隠れ状態
        hidden = outputs.hidden_states[self.output_layer]  # [1, T, D]
        hidden_np = hidden.squeeze(0).cpu().numpy()        # [T, D]

        # 離散トークンID の生成
        if self.use_kmeans and self.kmeans is not None:
            token_ids = self.kmeans.predict(hidden_np).astype(np.int32)
        else:
            # argmax 近似：各フレームで最大活性を示す次元をIDとする
            token_ids = np.argmax(hidden_np, axis=-1).astype(np.int32)

        # 全層の隠れ状態（オプション）
        layer_hidden = {}
        if return_all_layers:
            for layer_idx, hs in enumerate(outputs.hidden_states):
                layer_hidden[layer_idx] = hs.squeeze(0).cpu().numpy()

        return TokenSequence(
            token_ids=token_ids,
            hidden_states=hidden_np,
            layer_hidden=layer_hidden,
        )


# ─────────────────────────────────────────────
#  トークン比較
# ─────────────────────────────────────────────

def compare_tokens(
    original: TokenSequence,
    shifted: TokenSequence,
    shift_hz: float,
    min_length: Optional[int] = None,
) -> TokenDiff:
    """
    シフト前後のトークン系列を比較する。

    HuBERT のフレームレートは ~50 fps (20ms/frame)、
    WORLD のフレームレートは 200 fps (5ms/frame) など異なるため、
    短い方に合わせてトリミングする。

    Args:
        original: 元音声のトークン系列
        shifted: シフト後音声のトークン系列
        shift_hz: シフト量 (Hz)
        min_length: 比較するフレーム数の上限（None=自動）

    Returns:
        TokenDiff
    """
    n = min(len(original.token_ids), len(shifted.token_ids))
    if min_length is not None:
        n = min(n, min_length)

    orig_ids = original.token_ids[:n]
    shft_ids = shifted.token_ids[:n]

    changed_mask = orig_ids != shft_ids
    changed_indices = np.where(changed_mask)[0]
    change_rate = changed_mask.sum() / n if n > 0 else 0.0

    return TokenDiff(
        shift_hz=shift_hz,
        total_frames=n,
        changed_frames=int(changed_mask.sum()),
        change_rate=float(change_rate),
        changed_indices=changed_indices,
        original_ids=orig_ids,
        shifted_ids=shft_ids,
    )


def compute_hidden_distance(
    original: TokenSequence,
    shifted: TokenSequence,
    metric: str = "cosine",
) -> np.ndarray:
    """
    フレームごとの隠れ状態距離を計算する（連続表現の変化量）。

    Args:
        metric: "cosine" or "l2"

    Returns:
        distances: [T] フレームごとの距離
    """
    n = min(len(original.hidden_states), len(shifted.hidden_states))
    h_orig = original.hidden_states[:n]
    h_shft = shifted.hidden_states[:n]

    if metric == "cosine":
        dot = np.sum(h_orig * h_shft, axis=-1)
        norm = np.linalg.norm(h_orig, axis=-1) * np.linalg.norm(h_shft, axis=-1) + 1e-8
        return 1.0 - dot / norm  # cosine distance (0=同一, 1=直交)
    elif metric == "l2":
        return np.linalg.norm(h_orig - h_shft, axis=-1)
    else:
        raise ValueError(f"Unknown metric: {metric}")
