# todo




# 方針
* できるだけシンプルにする．
* run_summarization.pyからはできるだけ離さない．
* 他の生成系スクリプトでも使い回せるようにする．
* RuleTaker互換のスクリプトにする．
    - 方針: (a)と(b)が近い形式なので，(a)(b)を入力として学習できるスクリプトを作るべき．
    - RuleTaker -> NLProofS前処理 -> (a) RuleTaker-dataset
    - FLD-generator               -> (b) RuleTaker-dataset format




# 開発
* 機能
    - 学習
    - 評価
* NLProofSの様々な学習ヒューリスティックス
