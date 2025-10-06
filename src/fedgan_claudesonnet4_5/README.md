# FedGAN: Federated Generative Adversarial Network

このプロジェクトは、FedGAN論文（Federated Generative Adversarial Network）に基づいたプロトタイプ実装です。

## 概要

FedGANは、複数のエージェント（Agent）が独立した局所データを用いてGANを訓練し、仲介者（Intermediary/Server）を通じてパラメータを同期する連合学習フレームワークです。

### 主要な特徴

- **分散学習**: 各エージェントが独自のGeneratorとDiscriminatorを持ち、局所データで訓練
- **パラメータ同期**: K回の局所更新ごとに、仲介者サーバが重み付き平均でパラメータを集約
- **Non-IIDデータ**: MNISTの異なるクラスを各エージェントに割り当て、非独立同分布データで訓練
- **通信効率**: 同期間隔Kを調整することで、通信コストと性能のトレードオフを制御

## アーキテクチャ

```
fedgan_claudesonnet4_5/
├── main.py                    # メインエントリポイント（CLI）
├── requirements.txt           # 依存パッケージ
├── README.md                  # このファイル
├── agents/
│   └── agent.py              # Agentクラス（局所GAN訓練）
├── server/
│   └── intermediary.py       # Intermediaryクラス（パラメータ集約）
├── models/
│   ├── generator.py          # Generatorモデル（DCGAN）
│   └── discriminator.py      # Discriminatorモデル（DCGAN）
└── utils/
    ├── data_loader.py        # データロードとパーティション
    └── visualization.py      # GIF生成など
```

## セットアップ

### 必要な環境

- Python 3.12
- TensorFlow 2.16.2
- その他の依存パッケージはrequirements.txtを参照

### インストール

```bash
# 仮想環境の作成（推奨）
python3.12 -m venv venv
source venv/bin/activate  # macOS/Linux
# または: venv\Scripts\activate  # Windows

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使い方

### 基本的な実行

```bash
python main.py
```

デフォルト設定:
- エージェント数: 3
- エポック数: 50
- 同期間隔K: 5
- 出力ディレクトリ: tmp_outputs

### カスタムパラメータでの実行

```bash
python main.py \
  --num_agents 3 \
  --epochs 50 \
  --sync_interval 5 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --output_dir tmp_outputs
```

### すべてのCLIオプション

```
FedGAN特有のパラメータ:
  --num_agents N            エージェント数（デフォルト: 3）
  --epochs N                総エポック数（デフォルト: 50）
  --sync_interval K         同期間隔K（デフォルト: 5）

GANパラメータ:
  --batch_size N            バッチサイズ（デフォルト: 32）
  --learning_rate RATE      学習率（デフォルト: 1e-4）
  --noise_dim N             ノイズベクトルの次元（デフォルト: 100）
  --num_examples_to_generate N  
                           エポックごとに生成する画像数（デフォルト: 16）

出力パラメータ:
  --output_dir DIR          出力ディレクトリ（デフォルト: tmp_outputs）
  --gif_duration SECS       GIFのフレーム時間（秒）（デフォルト: 0.5）
  --visualization_seed N    可視化用シード（デフォルト: 42）
```

## 出力ファイル

実行後、以下のファイルが生成されます:

```
tmp_outputs/
├── agent_0/
│   ├── images/
│   │   ├── image_at_epoch_0001.png
│   │   ├── image_at_epoch_0002.png
│   │   └── ...
│   ├── agent_0_losses.csv       # 損失履歴（CSV）
│   ├── agent_0_losses.png       # 損失グラフ（PNG）
│   └── training_progress.gif    # 訓練進捗アニメーション
├── agent_1/
│   └── （同様の構造）
└── agent_2/
    └── （同様の構造）
```

### 出力ファイルの説明

- **images/**: 各エポックで生成された画像
  - 固定されたシードを使用して、訓練の進捗を可視化
  - エポック番号が画像に含まれる

- **agent_X_losses.csv**: エポックごとの損失値
  - カラム: `epoch`, `gen_loss`, `disc_loss`

- **agent_X_losses.png**: GeneratorとDiscriminatorの損失グラフ
  - 両方の損失を1つのグラフに表示

- **training_progress.gif**: 訓練進捗のアニメーションGIF
  - 各エポックの生成画像を順番に表示
  - 訓練の進捗を視覚的に確認可能

## アルゴリズム

このプロトタイプは、FedGAN論文のAlgorithm 1を実装しています:

1. **初期化**: 全エージェントが同一のパラメータで初期化
2. **局所訓練**: 各エージェントがK回の局所更新を実行
3. **同期**: 
   - 全エージェントがパラメータを仲介者に送信
   - 仲介者がデータサイズに基づく重み付き平均を計算
   - 平均パラメータを全エージェントに配布
4. **繰り返し**: ステップ2-3を指定エポック数繰り返す

### 重み付き平均の計算

```
w_n = Σ(j=1 to B) p_j * w_n^j
θ_n = Σ(j=1 to B) p_j * θ_n^j
```

ここで:
- `p_j = |R_j| / Σ|R_k|` (エージェントjのデータ比率)
- `w_n^j`: エージェントjのDiscriminatorパラメータ
- `θ_n^j`: エージェントjのGeneratorパラメータ

## GANモデル構造

### Generator

DCGAN Tutorialに基づく構造:
- 入力: ノイズベクトル（次元100）
- 隠れ層: Dense → BatchNorm → LeakyReLU → Reshape → Conv2DTranspose (複数層)
- 出力: 28x28x1画像（tanh活性化、値域[-1, 1]）

### Discriminator

DCGAN Tutorialに基づく構造:
- 入力: 28x28x1画像
- 隠れ層: Conv2D → LeakyReLU → Dropout (複数層)
- 出力: 1次元スカラー（ロジット）

## データパーティション

MNISTの10クラス（数字0-9）を各エージェントに分散:

- **3エージェントの場合** (デフォルト):
  - Agent 0: クラス [0, 1, 2, 3]
  - Agent 1: クラス [4, 5, 6]
  - Agent 2: クラス [7, 8, 9]

この非IID分布により、FedGANの連合学習能力を検証できます。

## 実装の注意点

### Single-Process実装

このプロトタイプは、単一プロセス内で全エージェントと仲介者を実行します。実際のAgent-Server間の通信は実装していませんが、将来的な拡張を容易にするため、Agent/Intermediaryクラスを分離しています。

### 学習用シードと可視化用シード

- **学習用**: バッチごとにランダムなノイズを生成（`tf.random.normal`）
- **可視化用**: 固定シードで生成して、訓練進捗を追跡可能に

### バッチ処理

各エージェントは局所データを繰り返し処理し、K回のステップで訓練します。データセットは自動的にシャッフルされます。

## トラブルシューティング

### TensorFlowのインストールエラー

macOS (Apple Silicon)の場合、`tensorflow-metal`が必要です:

```bash
pip install tensorflow==2.16.2
pip install tensorflow-metal>=0.8.4
```

### メモリ不足エラー

バッチサイズを減らしてください:

```bash
python main.py --batch_size 16
```

### 生成画像の品質が低い

- エポック数を増やす: `--epochs 100`
- 学習率を調整: `--learning_rate 2e-4`
- 同期間隔を調整: `--sync_interval 10`

## 参考文献

- FedGAN論文: "Communication-Efficient Learning of Deep Networks from Decentralized Data"
- TensorFlow DCGAN Tutorial: https://www.tensorflow.org/tutorials/generative/dcgan

## ライセンス

このプロトタイプは研究・教育目的で作成されています。
