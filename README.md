# FedGAN paper implementation with Github Copilot Agent

## Using model

- GPT-5 mini
- Claude Sonnet 4.5(preview)
- GPT-5-Codex(preview)

## How to use

- Add FedGAN paper at paper/arXiv-2006.07228v2
  - Download from https://arxiv.org/abs/2006.07228
- run Agent with prompt

## prompt sample

### analysis paper for prototype implementation

```txt
FedGANの論文（paper/arXiv-2006.07228v2/neurips_2020.tex）を日本語で要約し、
doc/fedgan_summary_{model}.md にMarkdownで保存

- 「Algorithm 1: Federated Generative Adversarial Network (FedGAN)」および、それに関連する記載を重点的に
- DatasetはMNIST部分のみで良い
- FedGANの処理シーケンスをMermaidで作成
  - 目的: シーケンスを元にPython(Tensorflow)でFedGANの実装をしたい
  - 注意: TeX 記法がMermaidのパーサーエラーを引き起こすため、Texから転記する場合、すべてプレーンテキスト（カンマ・バックスラッシュなどの特殊文字を除去）に置き換え
- 次に実施できること（提案）もMarkdownの最後に記載
  - まずは、最小限動くプロトタイプが良い（まずはSingle-Process、Single-Loopでも良い）
  - プロトタイプの設計、実装にあたり、決めるべき値や追加で必要な情報があれば、Markdownに記載
  - Actorは、Server（Intermediary）、Agent1、Agent2、AgentBとする
```

