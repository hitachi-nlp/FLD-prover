# todo




# 方針
* できるだけシンプルにする．
* run_summarization.pyからはできるだけ離さない．
* 他の生成系スクリプトでも使い回せるようにする．
    - 問い: LLMでも使えるか？
    - このことが明示されるようなアーキテクチャにする．
* RuleTaker互換のスクリプトにする．
    - 方針: (a)と(b)が近い形式なので，(a)(b)を入力として学習できるスクリプトを作るべき．
    - RuleTaker -> NLProofS前処理 -> (a) RuleTaker-dataset
    - FLD-generator               -> (b) RuleTaker-dataset format




# 開発

## 学習
* ./01.train.py
* settings -> setting
* [done] losses Noneになる
* [done] tensorboard

## [pending] 予測

## [done] 評価

## (最後)
* common.pyなどをライブラリに仕舞う
* ライブラリを分ける．
    - FLD_task
    - FLD_prover
* (最後) CoT prompt -> どういう形式にするか？
    - promptに柔軟性を与えたい．
        * serialize.py のフォーマットを変える．
            * common.pyなどに影響がある可能性がある．よって，最後にする．

## done
* [done] リファクタリング
    * FLD_task
    * FLD_prover

