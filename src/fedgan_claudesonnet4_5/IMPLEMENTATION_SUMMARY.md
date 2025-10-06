# FedGAN プロトタイプ実装完了

## 実装完了日時
2025年10月6日

## 実装概要

FedGAN論文（`docs/fedgan_summary_gpt5_codex.md`）の読解結果に基づき、FedGANのプロトタイプ実装を完了しました。

## ディレクトリ構造

```
src/fedgan_claudesonnet4_5/
├── README.md                  # 使い方、パラメータ説明
├── requirements.txt           # 依存パッケージ（DCGAN Tutorialと同じバージョン）
├── main.py                    # CLIメインエントリポイント
├── models/
│   ├── __init__.py
│   ├── generator.py          # Generatorモデル（DCGAN構造）
│   └── discriminator.py      # Discriminatorモデル（DCGAN構造）
├── agents/
│   ├── __init__.py
│   └── agent.py              # Agentクラス（局所GAN訓練）
├── server/
│   ├── __init__.py
│   └── intermediary.py       # Intermediaryクラス（パラメータ集約）
└── utils/
    ├── __init__.py
    ├── data_loader.py        # MNISTロード・パーティション
    └── visualization.py      # GIF生成
```

## 実装した主要機能

### 1. GAN モデル（models/）
- **Generator**: DCGAN Tutorial準拠
  - 入力: 100次元ノイズベクトル
  - 出力: 28x28x1画像（tanh活性化）
  - 構造: Dense → BatchNorm → LeakyReLU → Conv2DTranspose層
  
- **Discriminator**: DCGAN Tutorial準拠
  - 入力: 28x28x1画像
  - 出力: 真偽判定スカラー
  - 構造: Conv2D → LeakyReLU → Dropout層

### 2. Agent クラス（agents/agent.py）
- 局所GeneratorとDiscriminatorを保持
- 局所データでK回の訓練ステップを実行
- パラメータの送受信機能（get_parameters/set_parameters）
- エポックごとの画像生成（固定シードで進捗可視化）
- 損失履歴の記録とCSV/PNG出力

### 3. Intermediary クラス（server/intermediary.py）
- 全エージェントからパラメータを収集
- データサイズに基づく重み付き平均計算
  ```
  w_n = Σ(j=1 to B) p_j * w_n^j
  p_j = |R_j| / Σ|R_k|
  ```
- 集約パラメータを全エージェントに配布

### 4. データ処理（utils/data_loader.py）
- MNISTデータセットのロード
- クラスベースの非IIDパーティショニング
  - 例: 3エージェント → [0-3], [4-6], [7-9]のクラス分割
- 正規化: [-1, 1]範囲（tanh出力に対応）

### 5. 可視化（utils/visualization.py）
- Animation GIF生成
- エポック番号付き画像出力

### 6. メイン実行（main.py）
- argparseによるCLIインターフェース
- FedGANアルゴリズムのメインループ実装
- 出力ファイル管理

## CLI パラメータ

### FedGAN特有
- `--num_agents`: エージェント数（デフォルト: 3）
- `--epochs`: 総エポック数（デフォルト: 50）
- `--sync_interval`: 同期間隔K（デフォルト: 5）

### GAN設定
- `--batch_size`: バッチサイズ（デフォルト: 32）
- `--learning_rate`: 学習率（デフォルト: 1e-4）
- `--noise_dim`: ノイズ次元（デフォルト: 100）
- `--num_examples_to_generate`: 生成画像数（デフォルト: 16）

### 出力設定
- `--output_dir`: 出力ディレクトリ（デフォルト: tmp_outputs）
- `--gif_duration`: GIFフレーム時間（デフォルト: 0.5秒）
- `--visualization_seed`: 可視化用シード（デフォルト: 42）

## 出力ファイル

各エージェントごとに以下を出力:

```
tmp_outputs/
├── agent_0/
│   ├── images/
│   │   ├── image_at_epoch_0001.png  # エポックごとの生成画像
│   │   ├── image_at_epoch_0002.png
│   │   └── ...
│   ├── agent_0_losses.csv           # 損失履歴（CSV）
│   ├── agent_0_losses.png           # 損失グラフ（PNG）
│   └── training_progress.gif        # 訓練進捗アニメーション
├── agent_1/
│   └── （同様）
└── agent_2/
    └── （同様）
```

## 実装の特徴

### ✅ DCGAN Tutorialの設定を踏襲
- モデル構造: DCGAN Tutorial準拠
- Optimizer: Adam (lr=1e-4, beta_1=0.5)
- データ正規化: [-1, 1]
- 損失関数: Binary Cross Entropy

### ✅ FedGAN Algorithm 1の実装
1. 初期化: 全エージェント共通パラメータ
2. 局所訓練: K回のステップ
3. 同期: 重み付き平均でパラメータ集約
4. 配布: 集約パラメータを全エージェントに

### ✅ 可視化機能
- 固定シードでエポックごとに画像生成
- Animation GIF（エポック番号付き）
- 損失グラフ（Generator/Discriminator両方）

### ✅ Single-Process実装
- 通信実装なし（将来の拡張のためクラス分離）
- Agent/Intermediaryを明確に分離
- シンプルなメインループ

## 実装上の注意点

### 学習用と可視化用のシード
- **学習**: バッチごとにランダム（`tf.random.normal`）
- **可視化**: 固定シードで進捗追跡

### バッチ処理
- 各エージェントがK回のステップを実行
- データセットは自動シャッフル
- エポックごとに全エージェント同期

### 重み付き平均
- 各エージェントのデータサイズに基づく
- NumPy配列で計算してTensorFlowモデルに設定

## 使用例

### デフォルト実行
```bash
cd src/fedgan_claudesonnet4_5
python main.py
```

### カスタムパラメータ
```bash
python main.py \
  --num_agents 3 \
  --epochs 50 \
  --sync_interval 5 \
  --batch_size 32 \
  --output_dir tmp_outputs
```

## 次のステップ（実装完了後）

1. **動作検証**
   - 依存パッケージのインストール
   - 実際の訓練実行
   - 出力ファイルの確認

2. **性能評価**
   - 生成画像の品質確認
   - 損失グラフの分析
   - 同期間隔Kの影響評価

3. **拡張機能（オプション）**
   - FID（Frechet Inception Distance）評価
   - Agent-Server間の実際の通信実装
   - 他のデータセット対応

## 依存パッケージ

```
tensorflow==2.16.2
tensorflow-metal>=0.8.4
numpy>=1.26.0
matplotlib>=3.7.0
imageio>=2.31.0
pillow>=10.0.0
pandas>=2.2.0
```

## 参考
- FedGAN論文要約: `docs/fedgan_summary_gpt5_codex.md`
- DCGAN Tutorial: `src/gan_tf_tutorial/`
