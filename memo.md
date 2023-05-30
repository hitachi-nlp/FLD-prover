# todo
* SerializedDeductionStep -> SerializedDeduction
* load -> load_example
* top-level で呼びたいものは? loadとか．


## LLM評価
* なぜ全く解けないのか？
    * 翻訳が悪さをしているのでは？
        - GPT-4を試してみる
    * context_lenに入っていない？
        - 多分そう．few-shot exampleが全然入っていない．
            * todo: GPT-4でやってみる．
* メモリ機能

* 参考情報
    - [pricing](https://openai.com/pricing)
    * [Models - OpenAI API](https://platform.openai.com/docs/models/model-endpoint-compatibility)
    * [Pricing](https://openai.com/pricing)



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
