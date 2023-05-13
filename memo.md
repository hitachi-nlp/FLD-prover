# todo




# 方針
* 必要な最小限で1-pass通す．
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
* proof_common, common, evaluateなど
    - 不要な機能を削る．
    - ライブラリに仕舞う
    - 旧仕様(PROOF)に従っているソースを直す．
        - ack 'PRO|Unk|UNK'
        - FLD_task.schemaのスキーマを活用すること．
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

