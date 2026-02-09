# 言語条件付き潜在計画（JEPA-WM / MetaWorld）実装ガイド

作成日: 2026-02-01  
最終更新: 2026-02-01

---

## 目次

1. [プロジェクト概要](#1-プロジェクト概要)
2. [実装コードの構成](#2-実装コードの構成)
3. [要件定義と実装の対応表](#3-要件定義と実装の対応表)
4. [現在の実験設定](#4-現在の実験設定)
5. [検証可能な仮説・項目](#5-検証可能な仮説項目)
6. [次回実験のセットアップ手順](#6-次回実験のセットアップ手順)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. プロジェクト概要

### 1.1 研究目的

**言語指示から推定した目標潜在 `z_goal` が、画像ベースの目標なしで潜在空間での計画（MPC + CEM）に十分機能するか**を、MetaWorldタスクで定量評価する。

### 1.2 アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────────┐
│                        GoalHead                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Visual Enc.  │    │ Text Encoder │    │   Fusion     │       │
│  │  (DINOv2)    │    │ (SentenceTF) │    │  (MLP/Attn)  │       │
│  │   [凍結]     │    │    [凍結]    │    │   [学習]     │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │               │
│         └───────────────────┴───────────────────┘               │
│                             │                                   │
│                        z_goal_pred                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Latent Planning (MPC + CEM)                   │
│                                                                  │
│   z_t ──► World Model ──► z_{t+H} ──► ||z_{t+H} - z_goal||     │
│           [凍結]                          (最小化)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 実装コードの構成

### 2.1 ディレクトリ構造

```
jepa-wms/
├── app/
│   └── lang_goal/                    # 🆕 追加モジュール
│       ├── __init__.py               # モジュール初期化
│       ├── env_utils.py              # 環境変数・パスユーティリティ
│       ├── data_index.py             # データインデックス生成
│       ├── goal_head.py              # GoalHeadモデル定義
│       ├── text_encoder.py           # テキストエンコーダラッパー
│       ├── goalhead_dataset.py       # 学習用データセット
│       └── train_goalhead.py         # GoalHead学習スクリプト
│
├── evals/
│   └── lang_goal/                    # 🆕 追加モジュール
│       ├── __init__.py               # モジュール初期化
│       ├── lang_goal_agent.py        # 言語条件付きエージェント
│       └── eval.py                   # 評価スクリプト
│
├── configs/
│   └── lang_goal/                    # 🆕 設定ファイル
│       ├── data_index.yaml           # データインデックス設定
│       ├── goalhead_train.yaml       # GoalHead学習設定
│       └── mw_lang_goal_eval.yaml    # 評価設定
│
├── assets/
│   └── lang_goal/                    # 🆕 アセット
│       └── task_to_text.yaml         # タスク→言語指示マッピング
│
├── scripts/
│   └── download_weights.sh           # 🆕 重みダウンロードスクリプト
│
└── docs/
    ├── lang_goal_planning.md         # 🆕 詳細ドキュメント
    └── implementation_guide.md       # 🆕 このファイル
```

### 2.2 各ファイルの役割

| ファイル | 役割 |
|---------|------|
| `app/lang_goal/env_utils.py` | 環境変数の検証・パス展開・ログ設定 |
| `app/lang_goal/data_index.py` | Parquet走査、エピソード抽出、train/val/test分割、言語指示付与 |
| `app/lang_goal/goal_head.py` | GoalHeadモデル（MLP/Transformer/CrossAttn選択可能） |
| `app/lang_goal/text_encoder.py` | SentenceTransformerラッパー（凍結） |
| `app/lang_goal/goalhead_dataset.py` | PyTorch Dataset（画像・テキスト・目標潜在のロード） |
| `app/lang_goal/train_goalhead.py` | 学習ループ、検証、チェックポイント保存、WandB連携 |
| `evals/lang_goal/lang_goal_agent.py` | CEM計画、言語条件付きエージェント |
| `evals/lang_goal/eval.py` | オンライン評価、指標算出、結果保存 |

---

## 3. 要件定義と実装の対応表

### 3.1 機能要件（FR）

| 要件ID | 要件名 | 実装ファイル | 状態 |
|--------|--------|-------------|------|
| FR-1 | データインデックス生成 | `app/lang_goal/data_index.py` | ✅ 完了 |
| FR-2 | 言語指示の生成と管理 | `assets/lang_goal/task_to_text.yaml` + `data_index.py` | ✅ 完了 |
| FR-3 | モデルロード（凍結） | `app/lang_goal/train_goalhead.py` | ✅ 完了 |
| FR-4 | GoalHeadの定義 | `app/lang_goal/goal_head.py` | ✅ 完了 |
| FR-5 | GoalHead学習（教師あり） | `app/lang_goal/train_goalhead.py` | ✅ 完了 |
| FR-6 | 潜在計画（Goal画像なし） | `evals/lang_goal/lang_goal_agent.py` | ✅ 完了 |
| FR-7 | ベースライン比較 | `evals/lang_goal/eval.py` | ✅ 完了 |
| FR-8 | 評価指標の算出 | `evals/lang_goal/eval.py` | ✅ 完了 |
| FR-9 | 実行コマンドと設定 | `configs/lang_goal/*.yaml` | ✅ 完了 |

### 3.2 非機能要件（NFR）

| 要件ID | 要件名 | 実装箇所 | 状態 |
|--------|--------|---------|------|
| NFR-1 | 再現性（seed固定、ログ記録） | `train_goalhead.py`, `env_utils.py` | ✅ 完了 |
| NFR-2 | ログと成果物の配置 | 全スクリプトで `$JEPAWM_LOGS` 使用 | ✅ 完了 |
| NFR-3 | エラー設計（対処コマンド付き） | `env_utils.py` 全体 | ✅ 完了 |

### 3.3 受け入れ条件（AC）

| 条件ID | 条件 | 検証方法 | 状態 |
|--------|------|---------|------|
| AC-1 | 重み取得 | `./scripts/download_weights.sh` 実行 | ✅ 完了 |
| AC-2 | GoalHead学習 | `train_goalhead.py` 実行、loss記録確認 | ✅ 実行中 |
| AC-3 | オフライン誤差評価 | val/testでL2/cosine出力 | ✅ 学習中に出力 |
| AC-4 | オンライン計画評価 | `evals.main` 実行 | ⏳ 学習完了後 |

---

## 4. 現在の実験設定

### 4.1 学習設定（GoalHead）

現在実行中の設定（`configs/lang_goal/goalhead_train.yaml`より）：

```yaml
# モデル構成
goal_head:
  fusion_mode: "concat"           # concat / cross_attn / film
  hidden_dim: 512
  num_layers: 3
  dropout: 0.1
  use_proprio: false              # 固有受容覚（proprio）使用しない

# 学習パラメータ
num_epochs: 100
batch_size: 32
learning_rate: 1.0e-4
weight_decay: 1.0e-5
mixed_precision: true
clip_grad: 1.0

# 損失関数
l2_weight: 1.0                    # L2距離
cosine_weight: 0.1                # コサイン類似度（補助）

# テキストエンコーダ
text_encoder:
  model_name: "all-MiniLM-L6-v2"  # SentenceTransformer
  
# 凍結モジュール
# - DINOv2 Visual Encoder: 凍結
# - JEPA Predictor: 凍結
# - Text Encoder: 凍結
```

### 4.2 データ設定

| 項目 | 値 |
|------|-----|
| データセット | MetaWorld（42タスク） |
| 総エピソード数 | 12,600（タスクあたり300） |
| 訓練データ | 8,820エピソード（70%） |
| 検証データ | 1,890エピソード（15%） |
| テストデータ | 1,890エピソード（15%） |
| 分割seed | 42（固定） |

### 4.3 ハードウェア設定

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA RTX 5070 Ti（Blackwell） |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |

### 4.4 現在の学習状況

WandBで確認可能: https://wandb.ai/surumen-/jepa-lang-goal

| Epoch | Train Loss | Val Loss | Val Cosine Sim |
|-------|------------|----------|----------------|
| 0 | 0.2691 | 0.1182 | 0.9700 |
| 1+ | 進行中 | - | - |

---

## 5. 検証可能な仮説・項目

### 5.1 主要な研究仮説

#### 仮説1: 言語からの目標潜在推定精度
> 「言語指示 + 開始画像」から推定した `z_goal_pred` は、真の目標画像から得た `z_goal_gt` に十分近いか？

**検証指標:**
- L2距離: `||z_goal_pred - z_goal_gt||_2`
- コサイン類似度: `cos(z_goal_pred, z_goal_gt)`

**現在の結果（Epoch 0）:**
- Cosine Similarity: **0.97**（非常に高い）

#### 仮説2: 言語ベース計画の実行可能性
> 推定した `z_goal_pred` を目標として潜在計画（MPC + CEM）を行った場合、タスクは成功するか？

**検証指標:**
- 成功率（Success Rate）
- 成功までのステップ数

#### 仮説3: ベースラインとの性能差（劣化量）
> 言語ベースの目標は、画像ベースの目標と比較してどの程度劣化するか？

**検証指標:**
- 成功率差: `success(lang_goal) - success(img_goal)`
- ステップ差: `steps(lang_goal) - steps(img_goal)`

### 5.2 追加で検証可能な項目

| 検証項目 | 設定変更箇所 | 期待される知見 |
|---------|-------------|---------------|
| **Fusion方式の比較** | `goal_head.fusion_mode` | concat vs cross_attn vs film の性能差 |
| **テキストエンコーダの比較** | `text_encoder.model_name` | MiniLM vs CLIP vs 大規模LLMの影響 |
| **タスク難易度との相関** | 評価時のタスク選択 | 簡単なタスク（reach）vs 難しいタスク（assembly）での精度差 |
| **CEM設定の感度** | `planner.*` | horizon / population / iterations の影響 |
| **proprio使用の効果** | `goal_head.use_proprio` | 固有受容覚情報の追加効果 |

### 5.3 実験マトリクス（提案）

```
┌────────────────────┬─────────┬─────────┬─────────┐
│                    │ concat  │ cross   │  film   │
├────────────────────┼─────────┼─────────┼─────────┤
│ MiniLM-L6-v2       │   ✅    │   ⬜    │   ⬜    │
│ all-mpnet-base-v2  │   ⬜    │   ⬜    │   ⬜    │
│ CLIP-ViT-B/32      │   ⬜    │   ⬜    │   ⬜    │
└────────────────────┴─────────┴─────────┴─────────┘
✅ = 現在実行中, ⬜ = 未実行
```

---

## 6. 次回実験のセットアップ手順

### 6.1 環境構築（初回のみ）

```bash
# 1. リポジトリに移動
cd ~/workspace/jepa_lang_goal/jepa-wms

# 2. conda環境を有効化
conda activate jepa-wms

# 3. 環境変数を設定（.bashrcに追記推奨）
export JEPAWM_HOME=$HOME/workspace/jepa_lang_goal
export JEPAWM_DSET=$HOME/datasets/jepa_lang_goal
export JEPAWM_LOGS=$HOME/logs/jepa_lang_goal
export JEPAWM_CKPT=$HOME/checkpoints/jepa_lang_goal
export JEPAWM_OSSCKPT=$HOME/oss_checkpoints/jepa_lang_goal

# 4. マクロファイル生成
python setup_macros.py

# 5. RTX 50シリーズ用PyTorch（必要な場合のみ）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 6.2 重みダウンロード（初回のみ）

```bash
./scripts/download_weights.sh
```

### 6.3 データインデックス生成（初回のみ）

```bash
python -m app.lang_goal.data_index --config configs/lang_goal/data_index.yaml
```

### 6.4 GoalHead学習

```bash
# 基本実行
python -m app.lang_goal.train_goalhead --config configs/lang_goal/goalhead_train.yaml

# デバッグモード（少ないエポック）
python -m app.lang_goal.train_goalhead --config configs/lang_goal/goalhead_train.yaml --debug
```

### 6.5 評価実行

```bash
# 言語条件付き計画の評価
python -m evals.main --fname configs/lang_goal/mw_lang_goal_eval.yaml --debug
```

### 6.6 結果の確認

```bash
# ログディレクトリ
ls $JEPAWM_LOGS/lang_goal/

# GoalHeadチェックポイント
ls $JEPAWM_CKPT/lang_goal/goalhead/

# WandBダッシュボード
# https://wandb.ai/surumen-/jepa-lang-goal
```

### 6.7 ワンライナーコマンド（コピペ用）

```bash
# 全環境変数設定 + GoalHead学習
source ~/miniconda3/etc/profile.d/conda.sh && conda activate jepa-wms && \
export JEPAWM_HOME=$HOME/workspace/jepa_lang_goal && \
export JEPAWM_DSET=$HOME/datasets/jepa_lang_goal && \
export JEPAWM_LOGS=$HOME/logs/jepa_lang_goal && \
export JEPAWM_CKPT=$HOME/checkpoints/jepa_lang_goal && \
export JEPAWM_OSSCKPT=$HOME/oss_checkpoints/jepa_lang_goal && \
cd /home/ryo52/workspace/jepa_lang_goal/jepa-wms && \
python -m app.lang_goal.train_goalhead --config configs/lang_goal/goalhead_train.yaml
```

---

## 7. トラブルシューティング

### 7.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| `FileNotFoundError: Data path does not exist` | 環境変数未展開 | `source ~/.bashrc` を実行 |
| `CUDA error: no kernel image` | GPU非対応PyTorch | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `ModuleNotFoundError: No module named 'app.lang_goal'` | カレントディレクトリが違う | `cd $JEPAWM_HOME/jepa-wms` |
| `WandB: Network error` | ネットワーク問題 | `wandb offline` でオフラインモード |
| `OutOfMemoryError` | バッチサイズ過大 | `batch_size` を減らす |
| `torchcodec RuntimeError` | バージョン不整合 | `pip install --upgrade torchcodec` |

### 7.2 ログの確認方法

```bash
# GPU使用状況
watch -n 1 nvidia-smi

# 学習進捗（WandB）
# https://wandb.ai/surumen-/jepa-lang-goal
```

### 7.3 チェックポイントからの再開

```bash
python -m app.lang_goal.train_goalhead \
    --config configs/lang_goal/goalhead_train.yaml \
    --resume $JEPAWM_CKPT/lang_goal/goalhead/<run_id>/best_model.pth
```

---

## 付録A: 環境変数一覧

| 変数名 | 用途 | デフォルト値 |
|--------|------|-------------|
| `JEPAWM_HOME` | プロジェクトルート | `$HOME/workspace/jepa_lang_goal` |
| `JEPAWM_DSET` | データセット | `$HOME/datasets/jepa_lang_goal` |
| `JEPAWM_LOGS` | ログ出力先 | `$HOME/logs/jepa_lang_goal` |
| `JEPAWM_CKPT` | チェックポイント | `$HOME/checkpoints/jepa_lang_goal` |
| `JEPAWM_OSSCKPT` | OSS重み | `$HOME/oss_checkpoints/jepa_lang_goal` |

---

## 付録B: 参考リンク

- [JEPA-WMs GitHub](https://github.com/facebookresearch/jepa-wms)
- [JEPA-WMs 論文](https://arxiv.org/abs/2512.24497)
- [MetaWorld ベンチマーク](https://meta-world.github.io/)
- [WandB プロジェクト](https://wandb.ai/surumen-/jepa-lang-goal)

---

*このドキュメントは 2026-02-01 に作成されました*
