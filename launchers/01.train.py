#!/usr/bin/env python
import logging
from pathlib import Path
import time
import os
import json

import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger

from FLD_user_shared_settings import (
    get_base_setting,
    get_checkpoints,
    get_dataset_setting,
    get_batch_setting,
    get_save_eval_step_setting,
    get_model_setting,
    get_tokenizer_setting,
    get_qsub_cpu_gpu_setting,
    get_learning_setting,
    get_generation_setting,
    make_output_dir,
    make_command,
    run_by_engine,
)

logger = logging.getLogger(__name__)


_OTHER_MODELS = [

        # ============================ multilingual ============================
        # ('google/mt5-base', 'seq2seq', 'google/mt5-base'),
        # ('google/mt5-large', 'seq2seq', 'google/mt5-large'),


        # ============================ Japanese ============================
        # -------------- < 1B params --------------

        # ('retrieva-jp/t5-small-long', 'seq2seq', 'retrieva-jp/t5-base-long'),
        # ('retrieva-jp/t5-base-long', 'seq2seq', 'retrieva-jp/t5-base-long'),
        # ('retrieva-jp/t5-large-long', 'seq2seq', 'retrieva-jp/t5-large-long'),
        # ('megagonlabs/t5-base-japanese-web', 'seq2seq', 'retrieva-jp/t5-base-long'),

        # ('cyberagent/open-calm-small', 'causal', 'cyberagent/open-calm-small'),
        # ('cyberagent/open-calm-medium', 'causal', 'cyberagent/open-calm-medium'),
        # ('cyberagent/open-calm-large', 'causal', 'cyberagent/open-calm-large'),

        # ('rinna/japanese-gpt-neox-small', 'causal', 'cyberagent/open-calm-small'),

        # ('facebook/xglm-564M', 'causal', 'facebook/xglm-564M'),  # should use deepspeed


        # --------------- rejected models ---------------------

        # ('line-corporation/japanese-large-lm-3.6b-instruction-sft', 'causal', 'cyberagent/open-calm-3b'),
        # ('rinna/japanese-gpt-neox-3.6b-instruction-ppo', 'causal', 'cyberagent/open-calm-3b'),
        # ('stabilityai/japanese-stablelm-instruct-alpha-7b-v2', 'causal', 'matsuo-lab/weblab-10b'),

        # ('matsuo-lab/weblab-10b-instruction-sft', 'causal', 'matsuo-lab/weblab-10b'),
        # ('elyza/ELYZA-japanese-Llama-2-13b-fast-instruct', 'causal', 'matsuo-lab/weblab-10b'),
        # ('llm-jp/llm-jp-13b-instruct-full-jaster-v1.0', 'causal', 'matsuo-lab/weblab-10b'),
        # ('tokyotech-llm/Swallow-13b-instruct-hf', 'causal', 'matsuo-lab/weblab-10b'),


        # ('cyberagent/calm2-7b-chat', 'causal', 'cyberagent/open-calm-7b'),   # the training fails somehow

]


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # output_top_dir = Path('./outputs/01.train.py/20230729.case_study_finalize')
    # output_top_dir = Path('./outputs/01.train.py/20230801.case_study_finalize.fix')
    # output_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.fix.rerun')

    # output_top_dir = Path('./outputs/01.train.py/20230802.case_study_finalize.steps-20000')

    # output_top_dir = Path('./outputs/01.train.py/20231103.knowledge')

    # output_top_dir = Path('./outputs/01.train.py/20231203.jpn')
    # output_top_dir = Path('./outputs/01.train.py/20231203.jpn.no_subproof_for_unknown')
    # output_top_dir = Path('./outputs/01.train.py/20231206.new_models')
    # output_top_dir = Path('./outputs/01.train.py/debug')

    # output_top_dir = Path('./outputs/01.train.py/2023-12-06.elyza_fix')
    # output_top_dir = Path('./outputs/01.train.py/2023-12-06.elyza_before')

    # output_top_dir = Path('./outputs/01.train.py/2023-12-06.no_subproof_for_unknown.max_new_tokens=None')
    # output_top_dir = Path('./outputs/01.train.py/2023-12-06.D8')

    # output_top_dir = Path('./outputs/01.train.py/2023-12-12.logical_circuit')

    # output_top_dir = Path('./outputs/01.train.py/2023-12-12.logical_circuit')

    # output_top_dir = Path('./outputs/01.train.py/20231223.seed--1.timeout_fix')
    # output_top_dir = Path('./outputs/01.train.py/20231225.swallow-70b')
    # output_top_dir = Path('./outputs/01.train.py/20231225.swallow-70b.node-10')
    # output_top_dir = Path('./outputs/01.train.py/20231225.swallow-70b.node-8')

    # output_top_dir = Path('./outputs/01.train.py/20231213.jpn')
    # output_top_dir = Path('./outputs/01.train.py/20231213.jpn.seed--1')
    # output_top_dir = Path('./outputs/01.train.py/20231226.jpn.epoch--10')
    # output_top_dir = Path('./outputs/01.train.py/20231230.jpn.seed--0')
    # output_top_dir = Path('./outputs/01.train.py/20231230.jpn.swallow-70b.seed--0')

    # output_top_dir = Path('./outputs/01.train.py/20231230.jpn.seed--2')
    # output_top_dir = Path('./outputs/01.train.py/20230120.jpn.large')

    # output_top_dir = Path('./outputs/01.train.py/20230120.jpn.punipuni')

    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2')
    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.node--4')
    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.deepspeed-0.13')
    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.deepspeed-0.13.fp32')
    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.deepspeed-0.13.fp16.batch_size_32')
    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.deepspeed-0.13.fp16.batch_size_32.padding=max_len')

    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.batch_size_32')
    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.fp32')
    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2.fp32.half_batch_size')

    # output_top_dir = Path('./outputs/01.train.py/20240127.logical_cirtuit.llama2')
    # output_top_dir = Path('./outputs/01.train.py/2024-01-29.enhance_arguments')

    # output_top_dir = Path('./outputs/01.train.py/2024-01-31.multitask')

    # output_top_dir = Path('./outputs/01.train.py/2024-01-31.multitask')
    # output_top_dir = Path('./outputs/01.train.py/2024-02-02.multitask.step-2500')

    # output_top_dir = Path('./outputs/01.train.py/2024-02-14.translation_speedup')
    # output_top_dir = Path('./outputs/01.train.py/2024-02-18.continual_training.2024-02-14.translation_speedup.translation-v3')

    # output_top_dir = Path('./outputs/01.train.py/2024-03-21.trial_learning')

    # output_top_dir = Path('./outputs/01.train.py/2024-03-22.trial_learning')
    # output_top_dir = Path('./outputs/01.train.py/2024-03-22.other_logical_datasets')

    # output_top_dir = Path('./outputs/01.train.py/2024-03-23.other_logical_datasets')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.latest.num_proc=per_gpu.bs-10.nccl_timeout')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint.trial2')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint.4k_with_bs_half')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint.4k_with_bs_half.num_proc=1')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint.4k_with_bs_half.num_proc=4')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint.4k_with_bs_half.num_proc=4')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint.4k_with_bs_half.num_proc=all')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.bs_and_checkpoint.4k_with_bs_half.num_proc=all.short')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-12.1')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-12.1.multinode')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-12.1.eval_steps')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-11.8')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-11.8.bs-32')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-11.8.bs-5')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-11.8.bs-5.no_save')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-11.8.bs-5.no_save.koreeda_library')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-11.8.bs-5.no_save.max_new_tokens=1')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.cuda-11.8.bs-5.no_save.max_new_tokens=1.node--2')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.large_nodes')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.large_nodes.proof_intermediate_steps=False')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.large_nodes.proof_intermediate_steps=False.num_proc=1')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.large_nodes.proof_intermediate_steps=False.num_proc=1.small_nodes')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-01.large_nodes.proof_intermediate_steps=False.num_proc=1.small_nodes.large_batch')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-04.many_datasets')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-04.many_datasets.proof_intermediate_step=False')
    output_top_dir = Path('./outputs/01.train.py/2024-4-04.context=4096')

    # XXX ****************** Monitor deepspeed after launching, as it sometimes hangs!!!!!!! ***************

    DATASETS_DIRS = [
        # './outputs.FLD/00.create_corpus/20230729.case_study_finalize',
        './outputs.FLD/00.create_corpus/20230801.case_study_finalize.fix',
        './outputs.FLD/00.create_corpus/20230826.jpn',
        './outputs.FLD/00.create_corpus/20230901.random_transitive_verbs',
        './outputs.FLD/00.create_corpus/20230904.jpn',
        './outputs.FLD/00.create_corpus/20230912.jpn',
        './outputs.FLD/00.create_corpus/20230916.jpn',
        # './outputs.FLD/00.create_corpus/20231010.large_vocab.small',
        './outputs.FLD/00.create_corpus/20231010.large_vocab',
        './outputs.FLD/00.create_corpus/20231012.large_vocab',
        './outputs.FLD/00.create_corpus/20231021.knowledge',
        './outputs.FLD/00.create_corpus/20231103.knowledge',
        './outputs.FLD/00.create_corpus/20231203.jpn',
        './outputs.FLD/00.create_corpus/20231213.jpn',
        './outputs.FLD/00.create_corpus/20230120.jpn.large',

        './outputs.FLD/00.create_corpus/20230120.jpn.punipuni',
        './outputs.FLD/00.create_corpus/2024-01-29.enhance_arguments',
        './outputs.FLD/00.create_corpus/2024-02-14.translation_speedup',

        './outputs.FLD/00.create_corpus/20230122.past_FLD',
        './outputs.FLD/00.create_corpus/2024-03-29',
    ]



    logic_dataset_unames = [

        # ---------------------------------- 20230729.case_study_finalize ------------------------------------
        # '20230729.case_study_finalize.D3',
        # '20230729.case_study_finalize.D8',

        # 'hf.hitachi-nlp/FLD.v2__default',
        # 'hf.hitachi-nlp/FLD.v2__star',

        # ---------------------------------- 20231021.knowledge ------------------------------------
        # '20231021.knowledge.D3',
        # '20231021.knowledge.D3.w_knowledge',
        # '20231021.knowledge.D3.w_knowledge.complex-0.3',

        # ---------------------------------- 20231021.knowledge.D3 ------------------------------------
        # '20231021.knowledge.D3',
        # '20231021.knowledge.D3.complex-0.3',
        # '20231021.knowledge.D3.complex-0.3.w_knowledge',

        # ---------------------------------- 20231101.knowledge.D3 ------------------------------------
        # '20231103.knowledge.D3.knowledge_factor-5.0',

        # ---------------------------------- 20231213.jpn ------------------------------------
        # '20231213.jpn.D1_wo_dist',
        # '20231213.jpn.D1',
        # '20231213.jpn.D3',
        # '20231213.jpn.D8',

        # ---------------------------------- 20230118.jpn ------------------------------------
        # '20230118.jpn.wordnet.D3',
        # '20230118.jpn.wordnet.D3.argument_pred_arg_only',
        # '20230118.jpn.wordnet.D3.argument_pred_arg_only.no_kaku',
        # '20230118.jpn.BCCWJ.D3',
        # '20230118.jpn.punipuni.D3',

        # ---------------------------------- 20230120.jpn.punipuni ------------------------------------

        # '20230120.jpn.wordnet_repro_w_proposition.D1_wo_dist',
        # '20230120.jpn.wordnet_repro_w_proposition.D1',
        # '20230120.jpn.wordnet_repro_w_proposition.D3',
        # '20230120.jpn.wordnet_repro_w_proposition.D8',

        # '20230120.jpn.wordnet_repro_wo_proposition.D1_wo_dist',
        # '20230120.jpn.wordnet_repro_wo_proposition.D1',
        # '20230120.jpn.wordnet_repro_wo_proposition.D3',
        # '20230120.jpn.wordnet_repro_wo_proposition.D8',

        # '20230120.jpn.BCCWJ.D1_wo_dist',
        # '20230120.jpn.BCCWJ.D1',
        # '20230120.jpn.BCCWJ.D3',
        # '20230120.jpn.BCCWJ.D8',

        # '20230120.jpn.punipuni.D1_wo_dist',
        # '20230120.jpn.punipuni.D1',
        # '20230120.jpn.punipuni.D3',
        # '20230120.jpn.punipuni.D8',


        # -------------------------------- 20240127.logical_cirtuit.llama2 --------------------------------

        # '20231012.D3.large_vocab.smpl_stncs.cntx_shffls-3.trnsl_vrnts-3',
        # '20231103.knowledge.D3.knowledge_factor-5.0',

        # ---------------------------------- 2024-01-29.enhance_argumentsL ------------------------------------
        # '2024-01-29.enhance_arguments.past_reproduce',
        # '2024-01-29.enhance_arguments.theorems',
        # '2024-01-29.enhance_arguments.theorems.allow_smaller_proofs',
        # '2024-01-29.enhance_arguments.past_reproduce.D8',

        # ---------------------------------- 2024-02-14.translation_speedup ------------------------------------
        # '2024-02-14.translation_speedup.past_reproduce',
        # '2024-02-14.translation_speedup.D8',
        # '2024-02-14.translation_speedup.propositional-0.2',
        # '2024-02-14.translation_speedup.theorems',
        # '2024-02-14.translation_speedup.theorems.allow_smaller_proofs',
        # '2024-02-14.translation_speedup.translation-v2',
        '2024-02-14.translation_speedup.translation-v3',
        # '2024-02-14.translation_speedup.translation-v3.propositional-0.2',
        # '2024-02-14.translation_speedup.translation-v3.propositional-0.5'


        # ---------------------------------- other datasets ------------------------------------
        # 'hf.tasksource/ruletaker',
        # 'hf.hitachi-nlp/proofwriter_processed_OWA__depth-3ext',
        # 'hf.qbao775/PARARULE-Plus',

        # # 'hf.tasksource/robustLR',  # XXX not available yet

        # ---------------------------------- 2024-03-29.H100 ------------------------------------
        # '2024-03-29.JSAI_best',    # the same as "2024-02-14.translation_speedup.translation-v3"
        # '2024-03-29.JSAI_best.D8',
        # '2024-03-29.JSAI_best.theorems',
        # '2024-03-29.FLD_v2',
    ]

    num_train_examples_skip = None
    # num_train_examples_skip = 1000
    # num_train_examples_skip = 256 * 1250

    """
    (
        FLD_dataset_prob,
        other_datasets,
        streaming,
    ),
    """
    multitask_setting_list = [

        # streaming=True does not work in HAI cluster under the current proxy,
        # as the connection to huggingface.co via pyarrow library fails,
        # possibly due to the redirection forced by the proxy.

        # (
        #     1.0,
        #     [],
        #     False,
        # ),

        (
            0.5,
            [
                (1.0, 'DKYoon/SlimPajama-6B', None)
            ],
            False,
        ),

        # (
        #     0.0,
        #     [
        #         (1.0, 'DKYoon/SlimPajama-6B', None)
        #     ],
        #     False,
        # ),


        # (
        #     0.0,
        #     [
        #         (1.0, 'tatsu-lab/alpaca', None)
        #     ],
        #     False,
        # ),
    ]

    learnings = [
        # 'debug.ZS',
        # 'debug.micro',
        # 'debug.tiny',
        # 'debug.tiny.bs-32',
        # 'debug.tiny.bs-32.max_train_samples-10000',
        # 'debug.middle',

        # 'FT.step-5000',
        # 'FT.step-10000',

        # 'FT.step-30_bs-64',
        # 'FT.step-100_bs-64',

        # 'FT.step-1000__bs-128',
        # 'FT.step-2500__bs-128',
        # 'FT.step-5000__bs-128',

        'FT.step-800__bs-256',      # NeurIPS 100k
        # 'FT.step-1250__bs-256',   # JSAI
        # 'FT.step-2500__bs-256',

        # 'FT.step-1250__bs-512',
        # 'FT.step-2500__bs-512',

        # ---- JFLD experiments ----
        # 'LLM_FS.shot-5',
        # 'LLM_FS.shot-100',
        # 'LLM_FS.shot-1000',
        # 'LLM_FS.shot-10000',
        # 'LLM_FS.shot-30000',
    ]


    model_settings = [
        # ============================ english      ============================

        # ('t5-base', 'seq2seq', 't5-base'),                   # for debug
        # ('gpt2-medium', 'causal', 'gpt2-medium.short_cntx'),   # for debug

        # see [this paper](https://arxiv.org/abs/2401.16818) for comparison of 1B-class models
        # ('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 'causal', 'cyberagent/open-calm-3b'),
        # ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'causal', 'cyberagent/open-calm-3b'),

        # ('stabilityai/stablelm-2-1_6b', 'causal', 'cyberagent/open-calm-3b'),

        # ('h2oai/h2o-danube-1.8b-base', 'causal', 'cyberagent/open-calm-3b'),

        ('meta-llama/Llama-2-7b-hf', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Llama-2-13b-hf', 'causal', 'meta-llama/Llama-2-13b-hf'),
        # ('meta-llama/Llama-2-70b-hf', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('2024-02-14.translation_speedup.translation-v3', 'causal', 'cyberagent/open-calm-7b'),

        # ============================ japanese     ============================

        # ('line-corporation/japanese-large-lm-3.6b', 'causal', 'cyberagent/open-calm-3b'),
        # ('rinna/japanese-gpt-neox-3.6b', 'causal', 'cyberagent/open-calm-3b'),
        # ('cyberagent/calm2-7b', 'causal', 'cyberagent/open-calm-7b'),
        # ('stabilityai/japanese-stablelm-base-alpha-7b', 'causal', 'matsuo-lab/weblab-10b'),

        # ('matsuo-lab/weblab-10b', 'causal', 'matsuo-lab/weblab-10b'),
        # ('elyza/ELYZA-japanese-Llama-2-13b-fast', 'causal', 'matsuo-lab/weblab-10b'),
        # ('stockmark/stockmark-13b', 'causal', 'matsuo-lab/weblab-10b'),
        # ('pfnet/plamo-13b', 'causal', 'matsuo-lab/weblab-10b'),
        # ('llm-jp/llm-jp-13b-v1.0', 'causal', 'matsuo-lab/weblab-10b'),
        # ('tokyotech-llm/Swallow-13b-hf', 'causal', 'matsuo-lab/weblab-10b'),

        # ('tokyotech-llm/Swallow-70b-hf', 'causal', 'tokyotech-llm/Swallow-70b-hf'),
        # ('tokyotech-llm/Swallow-70b-instruct-hf', 'causal', 'tokyotech-llm/Swallow-70b-hf'),
    ]

    context_lengths = [
        # 2048,
        4096,
    ]

    proof_intermediate_steps_args = [
        True,
        # False,
    ]

    resume_from_checkpoint = None
    # resume_from_checkpoint = './outputs/01.train.py/checkpoint.2024-02-18'

    # save_model = False
    save_model = True

    # dry_run = True
    dry_run = False

    # run_mode = 'vanilla'
    # run_mode = 'torchrun'
    run_mode = 'deepspeed'

    # engine = SubprocessEngine()
    # engine = QsubEngine('ABCI', 'rt_G.small', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_G.large', n_resource=1)

    # engine = QsubEngine('ABCI', 'rt_F', n_resource=1)   # <= 10B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=2)   # >= 10B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=16)   # 70B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=32)   # 70B model

    engine = QsubEngine('haic', 'xhn_s.large', n_resource=1)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=2)

    hours = 24



















    # ------------------------------------ fixed settings -------------------------------------------
    if isinstance(engine, SubprocessEngine):
        region = 'haic'
        n_gpus_per_node = 1
        n_cpus_per_node = 1
        n_total_gpus = 1
        is_V100 = False

        # n_gpus_per_node = 4
        # n_total_gpus = 4

        # gpu_name_for_batch_size = 'V100.mem=16.run=vanilla.cntx=2048'
        # gpu_name_for_batch_size = 'V100.mem=16.run=deepspeed.cntx=2048'
        gpu_name_for_batch_size = 'H100.mem=80.run=vanilla.cntx=2048'

    # take_interval_between_jobs = False
    take_interval_between_jobs = True

    skip_if_exists = False
    # skip_if_exists = True

    instruction_args = [
        # False,       # better for chat-model?
        True,      # better for non-chat model, somehow.
    ]

    seeds = [
        0,
        # 1,
        # 2,
    ]

    lrates = [
        # much better on FLD performance than 1e-05, but could degratde on other downstream tasks?
        # 1e-4,

        1e-5,   # NLP_2024
    ]

    base_setting_name = 'default'

    use_test_as_val = False
    use_test_as_train = False

    max_steps = None
    eval_steps = None
    num_evals = None

    sample_negative_proof_args = [
        # True,
        False,    # better for 'all_at_once'
    ]

    no_subproof_for_unknown_args = [
        True,   # better
        # False,
    ]

    # max_eval_samples = 5
    # max_eval_samples = 301
    # max_eval_samples = 151
    max_eval_samples = 152

    epoch = None

    # hf_bug_zero_lr_offset = 0
    hf_bug_zero_lr_offset = 20

    # slow eneration is most likely the repetitions coming from underfitting, so we can safely discard such generations.
    # generation_timeout = 1
    generation_timeout = 3600 * 2

    # too long evaluation. we cut it off due to the same reason as above.
    # evaluation_timeout = 1
    evaluation_timeout = 3600 * 10

    warmup_ratio = None
    warmup_steps = None
    steps_upper = None
    train_effective_batch_size = None

    # script_type = 'run_prover'
    script_type = 'run_causal_prover'

    # seq2seq_proof_sampling = 'stepwise'
    seq2seq_proof_sampling = 'all_at_once'

    i_job = 0
    for logic_dataset_uname in logic_dataset_unames:
        for context_len in context_lengths:

            if isinstance(engine, QsubEngine):
                region = engine.region
                n_cpus_per_node, n_gpus_per_node, n_total_gpus, gpu_name_for_batch_size = get_qsub_cpu_gpu_setting(engine, context_len, run_mode)
                is_V100 = engine.resource.find('rt_G') >= 0 or engine.resource.find('rt_F') >= 0

            for logic_dataset_prob, other_datasets, streaming in multitask_setting_list:
                for learning in learnings:

                    for sample_negative_proof in sample_negative_proof_args:
                        for proof_intermediate_steps in proof_intermediate_steps_args:
                            for no_subproof_for_unknown in no_subproof_for_unknown_args:
                                for seed in seeds:
                                    for model_name, lm_type, model_name_for_batch_size in model_settings:

                                        n_resouce_org = engine.n_resource
                                        if model_name.find('70b') >= 0 and engine.n_resource < 2:
                                            logger.warning(f'70B model requires at least 2 nodes, without that the training or inference (generation) will be sig-killed. We use 3 nodes.')
                                            engine.n_resource = 2


                                        if learning.find('LLM_FS') >= 0:
                                            if learning == 'LLM_FS.shot-30000':
                                                _hours = 50
                                            else:
                                                _hours = 25
                                            if model_name.find('70b') >= 0:
                                                _hours = min(_hours * 2, 72)
                                        else:
                                            _hours = hours

                                        # V100 is only compatible with fp16, but not bf16,
                                        # but fp16 and deepspeed sometimes causes "Loss scale already at minimum" error.
                                        # Therefore, we use fp32 for V100.
                                        # https://github.com/microsoft/DeepSpeed/issues/4017#issuecomment-1754820339
                                        fp32 = (
                                            model_name.find('t5-') >= 0\
                                            or (is_V100\
                                                and (model_name.find('rinna/japanese-gpt2-medium') >= 0\
                                                     or model_name.find('llama') >= 0\
                                                     or os.path.exists(model_name + '/config.json') and json.load(open(model_name + '/config.json')).get('_name_or_path', '').find('llama') >= 0)
                                                )
                                        )
                                        # fp16 = not fp32
                                        bf16 = not fp32

                                        if lm_type == 'causal':
                                            proof_sampling = 'all_at_once'
                                        else:
                                            proof_sampling = seq2seq_proof_sampling

                                        for lrate in lrates:
                                            for instruction in instruction_args:

                                                setting = {}

                                                setting.update(get_base_setting(base_setting_name))

                                                setting.update(
                                                    get_learning_setting(
                                                        script_type,
                                                        learning,
                                                        epoch=epoch,
                                                        steps_upper=steps_upper,
                                                        warmup_steps=warmup_steps,
                                                        warmup_ratio=warmup_ratio,
                                                        train_effective_batch_size=train_effective_batch_size,
                                                        num_evals=num_evals,
                                                        max_eval_samples=max_eval_samples,
                                                        logic_dataset_prob=logic_dataset_prob,
                                                        hf_bug_zero_lr_offset=hf_bug_zero_lr_offset,
                                                        n_gpus=n_total_gpus,
                                                    )
                                                )


                                                other_dataset_probs = [other_dataset[0] for other_dataset in other_datasets]
                                                other_dataset_names = [other_dataset[1] for other_dataset in other_datasets]
                                                other_dataset_config_names = [other_dataset[2] for other_dataset in other_datasets]
                                                setting.update(
                                                    get_dataset_setting(
                                                        script_type,
                                                        dataset_uname=logic_dataset_uname,
                                                        top_dirs=DATASETS_DIRS,
                                                        other_dataset_names=other_dataset_names,
                                                        other_dataset_config_names=other_dataset_config_names,
                                                        other_dataset_probs=other_dataset_probs,
                                                        use_test_as_val=setting.get('use_test_as_val', use_test_as_val),
                                                        use_test_as_train=setting.get('use_test_as_train', use_test_as_train),
                                                        streaming=streaming,
                                                        instruction=instruction,
                                                    )
                                                )

                                                setting.update(
                                                    get_save_eval_step_setting(
                                                        max_steps = max_steps or setting['max_steps'],
                                                        eval_steps = eval_steps or setting['eval_steps'],
                                                        do_save_model=save_model,
                                                    )
                                                )

                                                setting.update(
                                                    get_batch_setting(
                                                        script_type,
                                                        gpu_name=gpu_name_for_batch_size,
                                                        n_gpus=n_total_gpus,
                                                        model_name=model_name_for_batch_size if proof_sampling == 'all_at_once' else model_name_for_batch_size + '.stepwise',
                                                        train_effective_batch_size=setting.get('train_effective_batch_size', None),
                                                        batch_size_per_gpu_factor = 1/2 if fp32 else 1.0,
                                                    )
                                                )

                                                if run_mode == 'deepspeed':
                                                    # for max_eval_arg_name in ['max_eval_samples', 'max_predict_samples', 'logic_max_eval_samples']:
                                                    for max_eval_arg_name in ['logic_max_eval_samples']:
                                                        max_eval_arg_sample = setting.get(max_eval_arg_name, None)
                                                        if max_eval_arg_sample is not None and setting['eval_effective_batch_size'] > max_eval_arg_sample:
                                                            raise ValueError(f'{max_eval_arg_name}={max_eval_arg_sample} should be larger than eval_effective_batch_size={setting["eval_effective_batch_size"]}, as it will lead to exception')

                                                setting.update(get_model_setting(model_name))

                                                setting.update(get_tokenizer_setting(model_name))

                                                setting.update(
                                                    get_generation_setting(
                                                        script_type,
                                                        generation_timeout=generation_timeout,
                                                        evaluation_timeout=evaluation_timeout,
                                                        generation_max_length=setting.get('max_target_length', None),
                                                   ),
                                                )

                                                setting.update({
                                                    'do_train': True,
                                                    # 'do_eval': True,   # automatically set by evaluation_strategy=step
                                                    'do_eval_in_outerloop': False,
                                                    'do_predict': False,
                                                })

                                                setting.update({
                                                    'script_type': script_type,
                                                    'seed': seed,

                                                    'logic_dataset_uname': logic_dataset_uname,
                                                    # 'other_dataset_name': other_dataset_names,    # should avoid list in the setting
                                                    # 'other_dataset_config_name': other_dataset_config_names,

                                                    'resume_from_checkpoint': resume_from_checkpoint,
                                                    'num_train_examples_skip': num_train_examples_skip,

                                                    'base_setting_name': base_setting_name,

                                                    'lm_type': lm_type,
                                                    # 'fp16': fp16,
                                                    'bf16': bf16,

                                                    # 'save_total_limit': save_total_limit,

                                                    # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                                                    'proof_sampling': proof_sampling,
                                                    'learning': learning,
                                                    'sample_negative_proof': sample_negative_proof,
                                                    'proof_intermediate_steps': proof_intermediate_steps,
                                                    'no_subproof_for_unknown': no_subproof_for_unknown,

                                                    'lr_scheduler_type': 'linear',
                                                    'learning_rate': lrate,
                                                    'weight_decay': 0.0,

                                                    'preprocessing_num_workers': max(1, int(n_cpus_per_node / n_gpus_per_node)),
                                                    # 'dataloader_num_workers': max(1, int(n_cpus_per_node / n_gpus_per_node)),

                                                    'lora': False,

                                                    'ddp_timeout': 3600 * 10,

                                                    'gpu_name_for_batch_size': gpu_name_for_batch_size,
                                                    'use_auth_token': True,
                                                    'log_examples': True,
                                                })

                                                if seed >= 2:  # for compatibility with older experiments of jpn
                                                    setting.update({
                                                        'random_sample_max_train_samples': True,
                                                        'random_sample_max_eval_samples': True,
                                                        'random_sample_logic_max_eval_samples': True,
                                                    })

                                                output_dir = make_output_dir(setting, output_top_dir)
                                                if skip_if_exists and (output_dir / 'log.txt').exists():
                                                    logger.info(f'Skipping "{output_dir}"')
                                                    continue
                                                command = make_command(script_type,
                                                                       output_dir,
                                                                       setting,
                                                                       run_mode,
                                                                       region,
                                                                       n_gpus_per_node=n_gpus_per_node)

                                                run_by_engine(
                                                    engine,
                                                    command,
                                                    output_dir,
                                                    delay = i_job * 0.5,
                                                    hours=_hours,
                                                    dry_run=dry_run
                                                )
                                                i_job += 1

                                                if streaming and take_interval_between_jobs:
                                                    logger.info('sleep for a wihle to avoid "Too many requests" exception for huggingface hub')
                                                    time.sleep(60 * 10)

                                        engine.n_resource = n_resouce_org

    logger.info('------------- ./01.train.py finished !! -----------')


if __name__ == '__main__':
    main()
