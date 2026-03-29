# フェーズ1：WORLDフォルマントシフト × HuBERT トークン変化率 実験

## ファイル構成

```
formant_watermark_exp/
├── formant_shift.py          # WORLDフォルマントシフトモジュール
├── hubert_token_analyzer.py  # HuBERTトークン分析モジュール
├── run_experiment.py         # メイン実験スクリプト
├── requirements.txt          # 依存ライブラリ
└── README.md
```

## セットアップ

```bash
pip install -r requirements.txt
```

> **注意** pyworld は Python 3.11 以降でビルドが必要な場合があります。
> `pip install pyworld` で失敗する場合は以下を試してください：
> ```bash
> pip install pyworld --no-binary pyworld
> ```

---

## 基本的な使い方

### 1. 全フォルマントを -100〜+100 Hz でスイープ（5Hz刻み）

```bash
python run_experiment.py \
    --input path/to/audio.wav \
    --output_dir ./results/sweep_all \
    --shift_min -100 \
    --shift_max 100 \
    --shift_step 5
```

### 2. F1・F2 のみをターゲットにして細かくスイープ

```bash
python run_experiment.py \
    --input path/to/audio.wav \
    --output_dir ./results/sweep_f1f2 \
    --shift_min -50 \
    --shift_max 50 \
    --shift_step 2 \
    --target_formants 0 1
```

### 3. 英語HuBERTを使用する場合

```bash
python run_experiment.py \
    --input path/to/audio.wav \
    --hubert_model facebook/hubert-base-ls960 \
    --hubert_layer 9 \
    --output_dir ./results/en_model
```

### 4. シフト済み音声も保存する

```bash
python run_experiment.py \
    --input audio.wav \
    --save_audio \
    --shift_min -30 --shift_max 30 --shift_step 5
```

---

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `results.csv` | 全 shift_hz ごとの数値データ |
| `tradeoff_curve.png` | トークン変化率・PESQ・SNR のトレードオフ曲線 |
| `config.json` | 実験設定（再現用） |
| `shifted_*.wav` | `--save_audio` 指定時のシフト済み音声 |

### CSV カラムの説明

| カラム | 説明 |
|---|---|
| `shift_hz` | フォルマントシフト量 (Hz) |
| `token_change_rate` | HuBERT トークン変化率（0〜1） |
| `token_changed_frames` | 変化したフレーム数 |
| `hidden_cosine_dist_mean` | 隠れ状態のコサイン距離（平均） |
| `f1_orig_hz` / `f1_shifted_hz` | F1 の実測値（前後） |
| `f2_orig_hz` / `f2_shifted_hz` | F2 の実測値（前後） |
| `snr_db` | 信号対雑音比 (dB) |
| `pesq` | PESQ スコア（1〜4.5） |
| `stoi` | STOI スコア（0〜1） |

---

## 設計メモ

### フォルマントシフトの方式

```
1. WORLD で sp（スペクトル包絡）を抽出
2. dB スケールで find_peaks → フォルマント位置を検出
3. アンカー点（DC・フォルマント・Nyquist）を設定
4. CubicSpline でスペクトルをリマッピング（ワープ）
5. 合成して出力
```

低域（<200Hz）と Nyquist 付近はアンカーとして固定するため、
基本周波数や高周波特性は保持されます。

### HuBERT トークンの近似

本スクリプトでは `argmax(hidden_states, axis=-1)` を擬似トークンIDとして使用します。
これは soft-VC 等で使われる k-means 量子化の近似ですが、
「フレームがどの方向に最大活性を持つか」の変化を捉えるため、
**定性的な変化率の傾向把握**には十分な精度があります。

フェーズ2以降で定量的な実験をする際は、
soft-VC の事前学習済み k-means モデルを使うことを推奨します：

```python
tokenizer = HubertTokenizer(
    use_kmeans=True,
    kmeans_path="path/to/kmeans_model.pkl"
)
```

### 次のステップ（フェーズ2：DDSP）

フェーズ1の結果から「何Hz のシフトでトークンが反転するか」の境界が判明したら：

- その境界付近を **可微分なフォルマントシフト操作（DDSP）** で学習
- 「最小シフト量でトークンを反転させる」最適化問題として定式化
- Neural Embedder で end-to-end 化

---

## トラブルシューティング

**Q: `pyworld` のインストールに失敗する**
```bash
# Windows の場合は wheels を試す
pip install pyworld --find-links https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
```

**Q: HuBERT モデルのダウンロードが遅い**
```bash
# HuggingFace キャッシュを指定する
export HF_HOME=/path/to/cache
```

**Q: CUDA メモリ不足**
```bash
# CPU に切り替える（遅くなりますが動作します）
python run_experiment.py --input audio.wav --device cpu
```

**Q: 音声が短すぎてトークンが取れない**
- HuBERT は最低 1 秒程度の音声が必要です
- 推奨: 3〜10 秒のクリーンな発話音声
