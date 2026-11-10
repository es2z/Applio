# リアルタイム変換: 今後の実行方針（2026-11-10）

ローカルの Claude Code 向け。前提は `docs/HANDOFF-realtime-latency-quality.md` と
`docs/realtime-latency-quality.md`。ここに書いた順に進める。各項目の作り方の細部は、
実機と音を確認できる側で判断してよい。方針は従来どおり
（オプトイン、初期値 = 従来の動作、音色は耳で決める、数値は手がかり）。

## 現状（ユーザーから）

- trim + ring を使用。Chunk 1280 ms で実遅延は約 1.7 s。目標は音質を保ったまま遅延を下げること。
- Chunk を縮めると（trim / ring でも）音への影響が大きい、という実感がある。
- **SPIN についての前提（2026-11-10 のユーザーの説明。CLAUDE.md の SPIN 節より新しい）:**
  - SPIN 追加学習済みの embedder（以下 kushinada'）の特徴量で学習した変換モデルは、
    変換品質が著しく低かった。どの Index を使っても直らなかった。
  - 通常の kushinada で学習したモデルに、kushinada' で作った Index を使うと、
    スコア上も聴感上も良くなった。今使っているのはこの組み合わせ。
- データセット（KU100 バイノーラル）は、近い耳のチャンネルを発話単位に近い形で選択済み。
  ささやき・息は手動で除外済み。吹かれ・ポップは聴いた限り無い。

## 1. 本家 Applio の最近のコミットを取り込む

衝突しない有用なものから。実装が軽く、本家で使われている変更であることが理由。
ただし本家は計測値を公開していないので、音に関わるものは A/B で確かめる。
この fork の realtime は本家から大きく離れている（ring 転送、`RealtimeOptions`）ので、
差分を読んで「そのまま入るか、考え方だけ移すか、見送るか」を項目ごとに決める。

候補（タイトルしか確認していない。`IAHispano/Applio` の `rvc/realtime` の履歴）:

| コミット | タイトル | メモ |
|---|---|---|
| `c1a3782` | Added a phase vocoder to improve crossfade quality | Chunk を縮めるとクロスフェード中の割合が増える（240 ms で 58%）ので優先。時刻固定 3 種と同じ問題への別解。A/B する |
| `2d16f92` | stabilize VAD gating and preserve sustained vowels | この fork のゲート（`gate_hold` など）と突き合わせる |
| `d973943` | skip gated inference and capture CUDA output processing | 引き継ぎ文書の A2 |
| `7e7dd0a` | add FP16 realtime decoder and graph-safe autotune | torch.compile との組み合わせを確認 |
| `f581097` | safely abort audio streams on stop | クラッシュ修正。legacy 転送に該当するか確認 |
| `1d1a9a7` | separate audio queues for output and monitor streams | 同上 |
| `f5f0564` | fade generated overlap when speech is gated | `gate_fade_ms` と同じ狙い。差があれば取り込む |

ONNX 系（`5048706`, `065d200`, `7718da0`）は規模が大きく、今回は対象外。

## 2. prematched 学習（Index で置き換えた特徴量で微調整）

出典は kNN-VC（Baas ほか、Interspeech 2023）。近傍検索で置き換えた特徴量はフレーム間の
つながりが不自然で、それが歪みになる。そこで学習側の特徴量も近傍で置き換えてから学習し、
推論時の入力に慣れさせる。kNN-VC の `prematch_dataset.py` は、自分自身を除いた同じフォルダーの
発話から、コサイン距離の上位 4 本の平均を取る。

この fork では、リアルタイムで Index と混ぜた特徴量をモデルに渡しているのに、モデルは学習時に
それを見ていない。今の構成（kushinada で学習、Index は kushinada'）では、混ぜる相手が別の
embedder の特徴量なので、このずれがさらに大きいはず。

**ただし上の前提がリスクになる。** kushinada' の特徴量だけで学習すると品質が著しく落ちた。
prematched は学習の入力を kushinada' 側へ寄せるので、寄せすぎると同じ失敗に近づく可能性がある。
なぜ kushinada' での学習が失敗したのかは分かっていない。そのため次の形で進める。

1. **学習の前に、ずれの大きさを測る（学習なし）。** 今の G で、同じクリップを
   (a) 元の特徴量、(b) 実際の Index Rate で kushinada' の近傍と混ぜた特徴量、(c) 近傍だけ（rate 1）
   から生成し、元の音声との mel 距離を比べる。`rvc/train/evaluation.py` は `net_g.infer`
   （特徴量から生成する経路）で mel を出しているので流用できる。posterior 経路の損失は
   特徴量を通らないので、この比較には使えない。(a) と (b) がほぼ同じなら、prematched で
   埋めるずれは小さく、やる意味は薄い。
2. **置き換えではなく混在にする。** 元の特徴量のクリップも残し、prematched は一部にする
   （割合と、Index Rate を 0〜実際に使う値の範囲で振るかどうかは判断に任せる）。
   rate 1 の特徴量だけで学習しない。
3. **短い学習で比べてから本番にする。** 層の聴き比べ（`docs/TASK-kushinada-layer-comparison.md`）と
   同じやり方で、同じ始点・同じ条件の「prematched あり / なし」を作り、ユーザーが使った
   スコアと耳で比べる。悪化したらここでやめる。

作るときの注意:

- 検索する側は kushinada の特徴量、探す相手は kushinada' の特徴量（リアルタイムと同じ組み合わせ）。
  検索の式は `rvc/realtime/index_search.py` の `ExactIndex` と揃える。
- 近傍は、そのクリップ自身のフレームを除いて探す（除かないと自分自身が返るだけになる）。
  `.index` ファイルは行がシャッフルされていてクリップとの対応が残らない
  （`extract_index.py`）ので、探す相手は kushinada' 側フォルダーの `extracted/*.npy` から作る。
- 抽出条件は抽出仕様に記録し、既存の特徴量と混ざらないようにする（再開の拒否も同様）。
- kNN-VC は F0 入力の無い別のモデルなので、RVC で同じ効果が出るかは未検証。

## 3. データセットの改善

どちらも「レポートを出すだけ、音は変えない」ツールにする。消すかどうかは聴いて決める。

1. **損失による外れ値探し。** 学習済み G で、クリップごとの mel 距離（シード固定）を
   全クリップについて出し、上位 3〜5% を聴く。`rvc/train/evaluation.py` が `net_g.infer` で
   出している mel を流用できる（特徴量や F0 の問題も拾える）。音声そのものの問題だけを
   見たいなら posterior 経路の損失も併記する。損失が高い = 悪いデータ、ではない
   （笑い声や高い声も上位に来る）。
2. **録音位置の分布。** 元のステレオで、クリップごとの左右のレベル差（帯域別）と時間差を出す。
   両耳は同じ声を聞いているので、内容の影響が打ち消されて位置だけが残る
   （片チャンネルのスペクトルは内容と声の強さでも変わるので不向き）。
   分布が 1 つに固まっていれば何もしない。分かれていれば、少数側の除外か、
   位置ごとに別の話者 ID を付けた学習を検討する。元のステレオファイルが必要。

## 4. リアルタイム条件の特徴量で微調整

抽出時にリアルタイムと同じ窓を滑らせ、「再生される位置」の特徴量を集めて、今の G/D から続けて学習する。

- 先に、未来文脈の必要量を測る（先読み 0〜300 ms で、窓の最新端の特徴量が全文脈の値から
  どれだけずれるか）。十分近いなら、この学習はやらない（備考 2）。
- embedder の特徴量だけ。F0 は全文脈のまま（精度の落ちた F0 で学習すると、モデルが F0 に
  従わなくなる方向に働く）。
- 未来文脈の長さは抽出時に乱数で振る（特定の Chunk 設定専用にしない）。
- 2 の prematched と組み合わせるか別々に試すかは、2 の結果を見て決める。
- 未来を見ないと決まらない情報は戻らない。全文脈と同等にはならない見込み。

## 並行して確認すること（設定だけ）

Chunk 240〜480 ms（40 ms の倍数）で、時刻固定 3 種、`apex_noise_envelope = fixed`、
`gain_mode = aligned`、`lookahead_s`、Extra 2.0〜2.5 s を 1 つずつ足して聴く
（`tools/realtime_stream_eval.py`）。ユーザーのテンプレートの `experimentTab` で、
未使用のものを先に確認する。

`index_search = exact` / `index_top_k` は実装済み（865b13c）だが GPU・実 Index では未計測。
`tools/index_recall.py` で従来の検索との差を測れる。リアルタイムで選んでいる embedder が
元の kushinada か SPIN 版かは未確認で、元の kushinada なら検索側を SPIN に揃える余地がある。

## 備考（参照しやすければ試す。必須ではない）

1. **近傍選択の滑らかさ**（kNN-SVC、ICASSP 2025、MIT、`SmoothKen/knn-svc`）。不適切な候補を除く
   距離と、候補の重みの最適化で、つながりを滑らかにする。`ExactIndex.retrieve` の重み付けの
   選択肢として足せる。式は未確認（README で分かったのは `--post_opt post_opt_0.2` の指定だけ）。
   論文かコードで確認できたら試す。
2. **HuBERT が実際に使う文脈は短い**（Meng, Goldwater, Tang、Interspeech 2025）。有効な文脈は
   比較的短く、構造の変更も追加学習もなしでストリーミング動作できる、という結論。数値は未確認。
   kushinada でも同じなら、Chunk を縮めたときの劣化は embedder の未来文脈ではなく、
   ブロックごとの揺れの側にある。4 の計測がその確認になる。

## やらないこと

- モデル側のチャンクマスク学習や、因果的な embedder への置き換え（DualVC 3 / Conan の方式）。
- F0 推定器の追加（PESTO）。
- MOS 予測器（DNSMOS など）によるデータ選別、Applio 内のノイズ抑制。

## 参照

- kNN-VC: https://arxiv.org/abs/2305.18975 / https://github.com/bshall/knn-vc
- kNN-SVC: https://arxiv.org/abs/2504.05686 / https://github.com/SmoothKen/knn-svc
- Effective Context in Neural Speech Models: https://www.isca-archive.org/interspeech_2025/meng25b_interspeech.html
- 本家の履歴: https://github.com/IAHispano/Applio/commits/main/rvc/realtime
