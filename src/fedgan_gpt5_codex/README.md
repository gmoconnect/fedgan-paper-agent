# FedGAN Prototype (GPT-5 Codex Plan)

このディレクトリには、FedGAN 論文の読解結果と実装方針（`docs/fedgan_summary_gpt5_codex.md`）に基づくプロトタイプ実装が含まれています。モデルは TensorFlow の DCGAN チュートリアル構成をベースにし、単一プロセス内で複数エージェントのローカル学習とサーバ同期をシミュレーションします。

## 主要機能

- 各エージェントは Generator / Discriminator をローカルに保持し、同期間隔 `K` 回のローカル更新後にパラメータをサーバへ送信
- サーバ（`Intermediary`）がデータサイズに基づく加重平均でパラメータを集約
- エポックごとに固定シードで生成画像を保存し、Epoch 番号入りのフレームを GIF 化
- Generator / Discriminator の損失を CSV + PNG に出力（各エージェント分）

## セットアップ

Python 3.12 の仮想環境を作成し、`requirements.txt` をインストールしてください。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r src/fedgan_gpt5_codex/requirements.txt
```

Apple Silicon + macOS の場合、`tensorflow-metal` が GPU アクセラレーションを提供します。

## 使い方

モジュールとして CLI を実行します（プロジェクトルートで `PYTHONPATH=src` を指定してください）。

```bash
PYTHONPATH=src python -m fedgan_gpt5_codex.main \
    --num-agents 3 \
    --epochs 50 \
    --sync-interval 5 \
    --batch-size 32 \
    --output-dir outputs/fedgan_codex_run
```

主なオプション:

- `--num-agents`: エージェント数（デフォルト 3）
- `--epochs`: 全体の学習エポック数（デフォルト 50）
- `--sync-interval`: 同期までのローカル更新回数 `K`（デフォルト 5）
- `--classes-per-agent`: JSON 文字列でクラス割り当てを指定（未指定なら均等分割）
- `--gif-duration`: GIF のフレーム表示時間（秒）
- `--samples-to-generate`: 各エポックで可視化に使用する生成枚数

学習成果物は `--output-dir`（デフォルト: `src/fedgan_gpt5_codex/outputs`）以下に保存されます。

```
outputs/
  agent_0/
    images/image_epoch_0001.png
    training_progress.gif
    agent_0_loss.csv
    agent_0_loss.png
  agent_1/
  agent_2/
```

GIF 各フレームには Epoch 番号が描画され、学習進捗をアニメーションで確認できます。損失曲線は Generator と Discriminator を同一グラフにまとめています。

## 注意事項

- このプロトタイプは単一プロセスでの挙動確認を目的としています。通信層の実装は含まれていません。
- FID などの生成品質評価はスコープ外です。
- 実際の学習（50 Epoch）には GPU 環境を推奨します。
