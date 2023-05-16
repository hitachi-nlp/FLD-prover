# todo
* [done] DPとDDPで結果が変わる．
    - 学習が進んでいない
        - tensorboardを調べる
    - prediction_step x マルチGPUが良くない
* [pending]
    - dataloader_num_workers > 0 が = 0に比べて非常に低速．
        - [pending] ボトルネックにはなっていない．




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
* 実験の再現
    1. [doing] 旧データセットでの性能計測
        * 高速化
        * 簡単なのからやる．
    1. universal_elimなどを入れた新データセットでの性能計測

* [done] ./A00.run_prover.sh の再現
    - ./A00.run_prover.sh では，zero_one_accuracy=0.8
    - ./01.train.pyあと，zero_one_accuracy=0.5
        * gradient_accumulation_steps=1 -> 0.6
    - negative_proofであった．
* [pending] 逃したヒューリスティクスが無いか確認する．
    * [pending] 再現できればそれで終わりなので．
    * ./NLProofS/src/prover/datamodule.py
          - shuffle_context
    * ./NLProofS/src/prover/evaluate/__init__.py
    * ./NLProofS/src/prover/evaluate/entailmentbank.py
    * ./NLProofS/src/prover/evaluate/ruletaker.py
    * ./NLProofS/src/prover/evaluate/scoring.py
    * ./NLProofS/src/prover/main.py
    * ./NLProofS/src/prover/model.py
    * ./NLProofS/src/prover/proof.py
    * ./NLProofS/src/prover/search.py
    * ./NLProofS/src/prover/utils.py
    * ./NLProofS/src/verifier/datamodule.py
    * ./NLProofS/src/verifier/main.py
    * ./NLProofS/src/verifier/model.py

* [done] サンプルデータの更新
* [done] ./01.train.py
    * [done] warningを消す
    * [done] 中身がきちんとしていそうか．
* [done] losses Noneになる
* [done] tensorboard

## [done] 予測

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
* CoT prompt -> どういう形式にするか？
    - promptに柔軟性を与えたい．
        * serialize.py のフォーマットを変える．
            * common.pyなどに影響がある可能性がある．よって，最後にする．

## done
* [done] リファクタリング
    * FLD_task
    * FLD_prover
