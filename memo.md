# todo
* "LLM評価"



# LLM評価
* 問題
    - 低すぎる．
        * "出力形式に慣れていない"
        * [done] shotが足りない？
            - [done] 20shot
                - 性能が下がった..．なぜ？
                    * 揺らぎ?
                        * [done] examplesを増やしてみる．
                            - 揺らぎだった．
                * [rejected] context_lenから溢れているだろうか？
                    - APIの場合，max_lenを超えると，エラーが出る．エラーが出ていないので，はみ出ていない．
        * [transferred] 揺らぎ？
            - [transferred] max_samplesを増やしてみる．
        * バグ？
            - [todo] webから試してみる．
    - answer_accuracyとの乖離が激しい
        - 出力形式に慣れていない
            - 出力形式に慣れていないのでproof_accuracyが低いが，理解はしているのでanswer_accuracyは高い
                 - 実例を見るに，こういった例はそれなりにある．
                     * [todo] どうするか？
                         - [todo] プロンプトでなんとかしてみる．
        - [rejected] 揺らぎである．
            - 
* 参考情報
    - [pricing](https://openai.com/pricing)
    * [Models - OpenAI API](https://platform.openai.com/docs/models/model-endpoint-compatibility)
    -https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        - 100token / 75words -> 1.33 [tokens] / [words]

## エラー解析

### ./outputs/13.analyze_llm_errors/20230601.fix_translation/dtst_nm=20230529.use_fixed_translation_for_LLM.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000/answr_accrcy_thrshld=0.1/mtrcs.llwd_addtnl_prf_stps=5/mtrcs.rply.dtst.n_sht=20/mtrcs.rply.dtst.prmpt_typ=in_context_examples.COT.v1/mtrcs.rply.dtst.sd=0/mtrcs.rply.mx_smpls=30/mtrcs.rply.mdl_nm=gpt-4/mtrcs.smlrty_thrshld=False/errors.txt

* 記法
    - LLM is wrong
    - gold is wrong
    - LLM is OK
        - goldとは違うが，大体あっている場合．
            - e.g.) ステップを飛ばす，「sent1 -> {sent1の内容}」のような自明なステップを挟み込む．

間違っている方を挙げる．
* example-0: LLM is wrong
* example-1: LLM is OK
* example-2: LLM is wrong
* example-3: gold is wrong
* example-6: LLM is wrong
    - and, or
* example-7: LLM is wrong
    - double negation
* example-8: LLM is OK
* example-9: LLM is wrong
    - double negation
* example-10: LLM is wrong
    - hallucination
        - UNKNOWNなのに__PROVED__
* example-11: gold is wrong
* example-12: gold is wrong
* example-13: LLM is wrong
    - double negation
* example-15: LLM is wrong
* example-16: LLM is wrong
* example-17: LLM is OK
* example-18: gold is wrong
    - goldが２つあるべき．
* example-19: LLM is wrong
* example-20: LLM is wrong
    - hallucination
* example-21: LLM is wrong
    - ただ，goldは日常センスとはかけ離れている．
* example-22: gold is wrong
* example-24: LLM is wrong
    - hallucination
* example-25: LLM is wrong
    - hallucination
* example-26: LLM is wrong
* example-27: LLM is wrong
    - hallucination
* example-28: LLM is OK

* 総計
    - correct = 5/30
    - wrong = 25 / 30
        * LLM is OK: 4
        * gold is wrong: 5
    - corrected accuracy = 14 / 30 = 46.7 %
    - コメント: 大体３倍合っていると思って良い．


* 何を間違うか？
    - negation (たまに)
    - and, or (たまに)
    - ステップ数が多い場合，理由を問わず途中で死ぬ(hallucination)
    - この揺れを評価側でなんとかできないか？



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



# Pitfalls
* localでやると高速に見える．ジョブで投げると低速に見える．
    - ターミナルに出力される頻度の問題である．真実は，tqdmの出力(X it / sec)を見よ．同じである．



# [todo] 開発

## [todo] 実験の再現
1. [再現実験](./experimental_logs.md)
1. universal_elimなどを入れた新データセットでの性能計測

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



# done
* [done] リファクタリング
    * FLD_task
    * FLD_prover
* [done] DPとDDPで結果が変わる．
    - 学習が進んでいない
        - tensorboardを調べる
    - prediction_step x マルチGPUが良くない
* [rejected]
    - dataloader_num_workers > 0 が = 0に比べて非常に低速．
        - [rejected] ボトルネックにはなっていない．
* [done] ./A00.run_prover.sh の再現
    - ./A00.run_prover.sh では，zero_one_accuracy=0.8
    - ./01.train.pyあと，zero_one_accuracy=0.5
        * gradient_accumulation_steps=1 -> 0.6
    - negative_proofであった．
* [done] サンプルデータの更新
* [done] ./01.train.py
    * [done] warningを消す
    * [done] 中身がきちんとしていそうか．
* [done] losses Noneになる
* [done] tensorboard
