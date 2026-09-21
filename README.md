# 概要  
日本語の高音質な学習及びリアルタイム変換を重視した個人的なApplioのフォークであり他の言語のことは一切考えられていません｡  
コンセプトとしては重くても良いから､高音質にするであり､高音質設定はかなり重いものが使用可能です(torch compile対応で多少は軽くなります)  
基本的にバイブコーディングで実装されており不安定な物が多く､追加モデルの中には手動ダウンロードが必要なものなどもあります(hugging faceで同意が必要なものなど)  
Windows11でしか確認しておりません｡

以下のような特徴を持ちます
- 主に日本語の変換品質を大幅に上げるため1024次元ボコーダーなどを含めた複数の追加モデルに対応
- リアルタイム推論時のテンプレートや学習時の学習率の引き継ぎ､学習率,mel減量,学習減量率など複数の便利な機能を追加
- リアルタイム推論にて､WDM-KSに対応し､遅延を少なくする機能をある程度搭載(少々不安定です)  
- torch compile に対応し､変換速度を向上しています
- 構成の違うモデルでも出来るだけG/Dを引き継ぎウォームアップを行えます｡

## 機能詳細

学習&リアルタイム推論共通
- vocoderにMRF HiFi-GAN,RefineGan,SiFi-GAN,Codename RingFormerを追加  
- F0にCrepe系モデルを10種類追加､通常使えるtiny,fullの他にsmall,medium,largeに加え､crepe speachニ対応またそれに対してmangio crepe実装を追加  
- 1024次元を含む追加のEmbedder Modelにいくつか対応､japanese-hubert-base-k2(768次元),japanese-hubert-large(1024次元),kushinada-hubert-large(1024次元)  
- 画面上部にtorch compile設定を追加

学習関連
- vocoderやembedder Modelなどが違う場合でも出来る限りG/Dを引き継いだうえで学習開始できるようにした｡
- 重み付けを引き継いで次の学習を行う設定を追加(必要はG/Dを学習前にそのフォルダにコピー配置してください)  
- 学習率や学習率減衰､メル減量をGUI上から変更可能にした
- 学習開始時に任意ステップの間Gの学習率を上げる設定を追加(学習引き継ぎ時にGをあまり引き継げない場合Dがあまりにも勝ち学習できない場合などに有効)

リアルタイム推論関連
- テンプレート機能に対応､選択モデル､indexファイル､ピッチやindexレートやボリューム､チャンクサイズなどをテンプレートで一括保存する｡現在出力デバイスもこの設定に入っているが外すかも｡
-  WDM-KSに対応
-  特にWDM-KSを使用している際にリアルタイム推論中ある程度遅延を下げるようにした｡


> [!WARNING]
> 通常のApplioで使われるHiFiGAN+Japanese-hubert-baseなど以外の組み合わせは基本的に1から､若しくは学習済のG/Dを一部引き継いでの学習となります(RefineGANの24/32の初期埋め込みは今のところ追加予定なし)  
> 初期埋め込みがない場合や､出自の違うG/Dを初期値として使う場合は､まともな音が出るまで少なくとも ** 10万ステップ ** 程度は必要な可能性が高いです(RTX4090 10-16並列で10時間ぐらい目安) 


> [!NOTE]
> 個人的感想ですが､推奨は以下になります｡ かなり重いですがF0で軽いものを使えばそこまでは変わらないと思います  
> vocoder: MRF HiFi-GAN  ･･･  HiFi-GAN学習モデルがほぼ引き継げるのが最大のメリット､収束は遅めで数値上のmelやklの天井は低めに見えますが､他の実装だと長く学習してると高音の癖が大きくでたり耳障りな付帯音や急峻な音量変化が途中から出てきがちですがこれはそういうった問題が少なく､HiFi-GANよりは全体的に良いという若干保守的な評価です(私のソースが悪い可能性は高いですが･･･)  
> pitch extraction algrithm(F0): mangio-crepe-full-speach   ･･･ 響きなどを含めた聴覚上の音質が高いです(品質向上に大して非常に重い為､次点で mangio-crepe系の軽いものも候補です)  
> embedder Model: kushinada-hubert-large   ･･･ 品質と話者性の両方が高いです､同じ1024次元モデルのjapanese-hubert-largeより声の再現力が高くおすすめです｡手動ダウンロードと配置が必要です  


<h1 align="center">
  <a href="https://applio.org" target="_blank"><img src="https://github.com/IAHispano/Applio/assets/133521603/78e975d8-b07f-47ba-ab23-5a31592f322a" alt="Applio"></a>
</h1>

<p align="center">
    <img alt="Contributors" src="https://img.shields.io/github/contributors/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Release" src="https://img.shields.io/github/release/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Stars" src="https://img.shields.io/github/stars/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Fork" src="https://img.shields.io/github/forks/iahispano/applio?style=for-the-badge&color=FFFFFF" />
    <img alt="Issues" src="https://img.shields.io/github/issues/iahispano/applio?style=for-the-badge&color=FFFFFF" />
</p>

<p align="center">A simple, high-quality voice conversion tool, focused on ease of use and performance.</p>

<p align="center">
  <a href="https://applio.org" target="_blank">🌐 Website</a>
  •
  <a href="https://docs.applio.org" target="_blank">📚 Documentation</a>
  •
  <a href="https://discord.gg/urxFjYmYYh" target="_blank">☎️ Discord</a>
</p>

<p align="center">
  <a href="https://github.com/IAHispano/Applio-Plugins" target="_blank">🛒 Plugins</a>
  •
  <a href="https://huggingface.co/IAHispano/Applio/tree/main/Compiled" target="_blank">📦 Compiled</a>
  •
  <a href="https://applio.org/playground" target="_blank">🎮 Playground</a>
  •
  <a href="https://colab.research.google.com/github/iahispano/applio/blob/main/assets/Applio.ipynb" target="_blank">🔎 Google Colab (UI)</a>
  •
  <a href="https://colab.research.google.com/github/iahispano/applio/blob/main/assets/Applio_NoUI.ipynb" target="_blank">🔎 Google Colab (No UI)</a>
</p>

> [!NOTE]  
> Applio will no longer receive frequent updates. Going forward, development will focus mainly on security patches, dependency updates, and occasional feature improvements. This is because the project is already stable and mature with limited room for further improvements. Pull requests are still welcome and will be reviewed.

## Introduction

Applio is a powerful voice conversion tool focused on simplicity, quality, and performance. Whether you're an artist, developer, or researcher, Applio offers a straightforward platform for high-quality voice transformations. Its flexible design allows for customization through plugins and configurations, catering to a wide range of projects.

## Terms of Use and Commercial Usage

Using Applio responsibly is essential.

- Users must respect copyrights, intellectual property, and privacy rights.
- Applio is intended for lawful and ethical purposes, including personal, academic, and investigative projects.
- Commercial usage is permitted, provided users adhere to legal and ethical guidelines, secure appropriate rights and permissions, and comply with the [MIT license](./LICENSE).

The source code and model weights in this repository are licensed under the permissive [MIT license](./LICENSE), allowing modification, redistribution, and commercial use.

However, if you choose to use this official version of Applio (as provided in this repository, without significant modification), you must also comply with our [Terms of Use](./TERMS_OF_USE.md). These terms apply to our integrations, configurations, and default project behavior, and are intended to ensure responsible and ethical use without limiting their use in any way.

For commercial use, we recommend contacting us at [support@applio.org](mailto:support@applio.org) to ensure your usage aligns with ethical standards. All audio generated with Applio must comply with applicable copyright laws. If you find Applio helpful, consider supporting its development [through a donation](https://ko-fi.com/iahispano).

By using the official version of Applio, you accept full responsibility for complying with both the MIT license and our Terms of Use. Applio and its contributors are not liable for misuse. For full legal details, see the [Terms of Use](./TERMS_OF_USE.md).

## Getting Started

### 1. Installation

Run the installation script based on your operating system:

- **Windows:** Double-click `run-install.bat`.
- **Linux/macOS:** Execute `run-install.sh`.

### 2. Running Applio

Start Applio using:

- **Windows:** Double-click `run-applio.bat`.
- **Linux/macOS:** Run `run-applio.sh`.

This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring

To monitor training or visualize data:

- **Windows:** Run `run-tensorboard.bat`.
- **Linux/macOS:** Run `run-tensorboard.sh`.

For more detailed instructions, visit the [documentation](https://docs.applio.org).

## References

Applio is made possible thanks to these projects and their references:

- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [rvc-cli](https://github.com/blaisewf/rvc-cli) by blaisewf

### Contributors

<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
