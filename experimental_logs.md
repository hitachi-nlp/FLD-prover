# 再現実験
* 2023-05-23 現状
    - FLD.2の実験をしたところ．性能差があるので，対処が必要．

## todo
* 誤り分析をする．
    * depthごと・ラベルごとの集計
        - UNKNOWNだけ合っているのでは？
* 違いの原因を探る．
    - [done] ハイパラ
        * steps = 20000?
            - ./NLProofS/30.show_tf_results.py を見ると，RT_large_steps(20000step)になっている．これが最も疑わしい．
    - アルゴリズム
        - [todo] 逃したヒューリスティクスが無いか確認する．
        - [rejected] sample_negative_proof
    - メトリクス
        - スコアラー

## sFLD-impl
* 目標性能: 82.2

* ./outputs/01.train.py/2023-05-16.sFLD-impl
    - sample_negative_proof=True
        * 実測性能: 0.767
        * 考察:
            - 学習曲線を伸ばしても達しないように見える．
    - sample_negative_proof=False
        * 実測性能: 0.764
        * 考察
* ./outputs/01.train.py/2023-05-17.sFLD-impl.large_steps
        * 実測性能: 0.792
        * 考察
            * 大分近づいた．これくらいの差は，アルゴリズムの詳細で変りそう．
                * [done] 他のデータセットでもやってみる．


## [done] FLD-impl
* 目標性能: 74.6
* ./outputs/01.train.py/2023-05-16.FLD-impl
    - sample_negative_proof=True
        * 実測性能: 0.683
        * 考察:
            * sFDL-implと同様，6ポイントくらい低い．よって，性能が低い問題は，統計的揺らぎではない．
* ./outputs/01.train.py/2023-05-17.FLD-impl.large_steps
    * 実測性能: 0.738
    * 考察
        - ほぼ目標性能を達成した．


## FLD.2
* 目標性能: 66.8
* ./outputs/01.train.py/FLD.2.large_steps
    * 実測性能: 0.599


# "逃したヒューリスティクスが無いか確認する．"
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

