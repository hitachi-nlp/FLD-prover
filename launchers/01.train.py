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

    # output_top_dir = Path('./outputs/01.train.py/2023-12-06.no_subproof_for_unknown.max_new_tokens=None')
    # output_top_dir = Path('./outputs/01.train.py/2023-12-06.D8')

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

    # output_top_dir = Path('./outputs/01.train.py/2024-4-06.debug')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-06.context-4k')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-07.instruction_tuning')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-07.various_corpora')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-07.proof_intermediate_steps=False')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-07.context=4k')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-07.large_models')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-09.save_cache')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-09.precision')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-09.no_aug')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-09.FLD_variation')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-15.FLD.v2.centered')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-17.rec_adam')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-18.rec_adam')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-20.production')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-20.production.additional')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-20.production.additional.additional')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-22.production.llama3')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-23.refine_production')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-24.finalize_production')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-24.finalize_production.additional')

    # output_top_dir = Path('./outputs/01.train.py/2024-4-24.finalize_production.other_llms')
    # output_top_dir = Path('./outputs/01.train.py/2024-05-03.ablation')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-24.finalize_production.other_llms.mistral_tokenizer')
    # output_top_dir = Path('./outputs/01.train.py/2024-05-08.ref_prob')
    # output_top_dir = Path('./outputs/01.train.py/2024-4-24.finalize_production.other_llms.tokenizer_unk')

    # output_top_dir = Path('./outputs/01.train.py/2024-5-13.large_models')
    # output_top_dir = Path('./outputs/01.train.py/2024-5-19.flight')
    # output_top_dir = Path('./outputs/01.train.py/2024-5-22.submission_final')




    # output_top_dir = Path('./outputs/01.train.py/2024-06-08.LPT')
    # output_top_dir = Path('./outputs/01.train.py/2024-06-08.LPT.1')
    # output_top_dir = Path('./outputs/01.train.py/2024-06-09')
    output_top_dir = Path('./outputs/01.train.py/2024-06-09.LPT')







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
        './outputs.FLD/00.create_corpus/2024-05-03.ablation',
        './outputs.FLD/00.create_corpus/2024-05-08.ref_prob',
        './outputs.FLD/00.create_corpus/2024-05-19.ablation_with_theorems/',
        './outputs.FLD/00.create_corpus/2024-06-08.LPT',
    ]







    model_settings = [
        # ============================ english      ============================

        # ('meta-llama/Llama-2-7b-hf', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),


        ('meta-llama/Meta-Llama-3-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),


        # ('Qwen/Qwen1.5-7B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('Qwen/Qwen1.5-32B', 'causal', 'meta-llama/Llama-2-70b-hf'),
        # ('Qwen/Qwen1.5-72B', 'causal', 'meta-llama/Llama-2-70b-hf'),


        # ('mistralai/Mistral-7B-v0.1', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('mistralai/Mixtral-8x7B-v0.1', 'causal', 'meta-llama/Llama-2-70b-hf'),


        # ('lmsys/vicuna-7b-v1.5', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('lmsys/vicuna-33b-v1.3', 'causal', 'meta-llama/Llama-2-70b-hf'),


        # ('tiiuae/falcon-40b', 'causal', 'meta-llama/Llama-2-70b-hf'),

        
        # ('CohereForAI/c4ai-command-r-plus', 'causal', 'meta-llama/Llama-2-70b-hf'),






        # ('t5-base', 'seq2seq', 't5-base'),                   # for debug
        # ('gpt2-medium', 'causal', 'gpt2-medium.short_cntx'),   # for debug


        # see [this paper](https://arxiv.org/abs/2401.16818) for comparison of 1B-class models
        # ('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 'causal', 'cyberagent/open-calm-3b'),
        # ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'causal', 'cyberagent/open-calm-3b'),


        # ('meta-llama/Llama-2-7b-hf', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Llama-2-13b-hf', 'causal', 'meta-llama/Llama-2-13b-hf'),
        # ('meta-llama/Llama-2-70b-hf', 'causal', 'meta-llama/Llama-2-70b-hf'),



        # ('stabilityai/StableBeluga-7B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('stabilityai/StableBeluga-13B', 'causal', 'meta-llama/Llama-2-13b-hf'),


        # ('microsoft/Orca-2-7b', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('microsoft/Orca-2-13b', 'causal', 'meta-llama/Llama-2-13b-hf'),



        # ('microsoft/phi-2', 'causal', 'meta-llama/Llama-2-7b-hf'),


        # XXX! does not work for now!
        # ('stabilityai/stablelm-2-12b', 'causal', 'meta-llama/Llama-2-13b-hf'),


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
        # '2024-02-14.translation_speedup.translation-v3',
        # '2024-02-14.translation_speedup.translation-v3.propositional-0.2',
        # '2024-02-14.translation_speedup.translation-v3.propositional-0.5'


        # ---------------------------------- 2024-03-29.H100 ------------------------------------
        # '2024-02-14.translation_speedup.translation-v3',

        # '2024-03-29.JSAI_best',    # the same as "2024-02-14.translation_speedup.translation-v3"
        # '2024-03-29.JSAI_best.D8',
        # '2024-03-29.JSAI_best.theorems',

        # '2024-03-29.JSAI_best.D8.no_aug',

        # '2024-03-29.JSAI_best.no_aug.dstrctr-10',
        # '2024-03-29.JSAI_best.no_aug.cmplx-0.25',

        # '2024-03-29.FLD_v2.D8',




        # ---------------------------------- NeurIPS 2024 ------------------------------------

        # 'hf.hitachi-nlp/ruletaker',
        # 'hf.hitachi-nlp/PARARULE-Plus',
        # 'hf.hitachi-nlp/FLD.v2__default',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.2',


        # -- rejected --
        # '2024-03-29.FLD_v2',
        # '2024-03-29.FLD_v2.theorems-0.3.fix',
        # '2024-03-29.FLD_v2.theorems-0.1.fix',
        # '2024-03-29.FLD_v2.theorems-0.03.fix',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems-0.3',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems-0.03',

        # '2024-03-29.JSAI_best.no_aug.trnsl-v2',
        # '2024-03-29.JSAI_best.no_aug.trnsl-v2.theorems-0.3',
        # '2024-03-29.JSAI_best.no_aug.trnsl-v2.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-v2.theorems-0.03',

        # '2024-03-29.JSAI_best.no_aug',
        # '2024-03-29.JSAI_best.no_aug.theorems-0.3',
        # '2024-03-29.JSAI_best.no_aug.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.theorems-0.03',

        # 'hf.hitachi-nlp/proofwriter_processed_OWA__depth-3ext',
        # # 'hf.tasksource/robustLR',  # the training dataset is small, might be "test-only" dataset.


        # ------------------------------- 2024-05-03.ablation --------------------------------
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.voc-100',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.dstrct-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.rule-G_MP',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-3-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-1',


        # -- rejected --
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.stps-3',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.stps-5-3',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.stps-8-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.stps-1-2',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.stps-1-1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.stps-1-0',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.rule-G_MP.stps-3',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small.trnsl-old',


        # ----------------------------------- ./outputs/00.create_corpus/2024-05-08.ref_prob --------------------------
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-3-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-1-2',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-3-3',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-5-5',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.theorems-0.2',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.stps-3-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.stps-1-2',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.stps-3-3',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.stps-5-3',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.stps-5-5',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.05',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.2',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.30',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.40',



        # ------------------------------------- ./outputs/00.create_corpus/2024-05-19.ablation_with_theorems --------------------------
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.voc-100',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.dstrct-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.stps-3-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.transl_sttng-1',



        # ---------------------------------- LPT ------------------------------------

        # 'hf.hitachi-nlp/ruletaker',
        # 'hf.hitachi-nlp/PARARULE-Plus',
        # 'hf.hitachi-nlp/FLD.v2__default',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1',

        # 'hf.hitachi-nlp/proofwriter_processed_OWA__depth-3ext',

        '2024-03-29.JSAI_best.no_aug.trnsl-thing.large',
    ]








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


        # (
        #     0.5,
        #     [
        #         (1.0, 'DKYoon/SlimPajama-6B', None)
        #     ],
        #     False,
        # ),

        # (
        #     0.0,
        #     [
        #         (1.0, 'DKYoon/SlimPajama-6B', None)
        #     ],
        #     False,
        # ),



        (
            0.00,
            [
                (1.0, 'DarqueDante/SlimPajama-62B-Text-1of6', None)
            ],
            False,
        ),


        (
            0.01,
            [
                (1.0, 'DarqueDante/SlimPajama-62B-Text-1of6', None)
            ],
            False,
        ),


        (
            0.03,
            [
                (1.0, 'DarqueDante/SlimPajama-62B-Text-1of6', None)
            ],
            False,
        ),


    ]







    learnings = [
        # 'debug.ZS',
        # 'debug.micro',
        # 'debug.tiny',
        # 'debug.tiny.bs-32',
        # 'debug.tiny.bs-32.max_train_samples-10000',
        # 'debug.middle',
        # 'debug.large',
        # 'debug.very_large',

        # 'FT.step-5000',
        # 'FT.step-10000',

        # 'FT.bs-64__step-30',
        # 'FT.bs-64__step-100',

        # 'FT.bs-128__step-1000',
        # 'FT.bs-128__step-2500',
        # 'FT.bs-128__step-5000',


        # --------- dataset = 100k, logic=1.0--------------------
        # 'FT.bs-256__step-390.wrmp-200',
        # 'FT.bs-256__step-390.wrmp-200.few_save',

        # --------- dataset = 200k, logic=1.0--------------------
        # 'FT.bs-256__step-780.wrmp-400',

        # --------- dataset = 300k, logic=1.0--------------------
        # 'FT.bs-256__step-1172.wrmp-600',


        # --------- dataset = 300k, logic=1.0 -----------
        # 'FT.bs-256__step-1170.wrmp-200',
        # 'FT.bs-512__step-586.wrmp-200',


        # --------- dataset = 300k, logic=0.5 -----------
        # 'FT.bs-384__step-1560',
        # 'FT.bs-512__step-1200',
        # 'FT.bs-640__step-940',


        # --------- dataset = 300k, logic=0.25 -----------
        # 'FT.bs-768__step-1560',


        # --------- others --------------------
        # 'FT.bs-256__step-152',      # Alpaca 3 epochs with context 2048


        # ---- JFLD experiments ----
        # 'LLM_FS.shot-5',
        # 'LLM_FS.shot-100',
        # 'LLM_FS.shot-1000',
        # 'LLM_FS.shot-10000',
        # 'LLM_FS.shot-30000',


        # ---- LPT ----------
        'LPT.bs-1024__step-24000.wrmp-100',
    ]






    # (optimizer, regularization, anneal_target_task_weight, fisher_coef)
    optimizer_setings = [
        (None, None, None, None, None),
        # ('rec_adam', 'l2', 0.5, 'immediately_from_beginning', 0),
        # ('rec_adam', 'l2', 0.5, 'immediately_from_beginning', 300),
        # ('rec_adam', 'l2', 0.5, 'immediately_from_beginning', 1000),
        # ('rec_adam', 'l2', 0.5, 'immediately_from_beginning', 3000),
        # ('rec_adam', 'l2', 0.5, 'immediately_from_beginning', 10000),
    ]


    lrates = [
        # 1e-5,
        # 3e-6,    # the best
        # 1e-6,

        1e-4,    # LPT, from Pythia
    ]

    update_parameters_args = [
        'all',
        # 'attention',
        # 'mlp',
    ]

    from_scratch_args = [
        # False,
        True,
    ]


    # weight_decay = None
    weight_decay = 0.01   # LPT, from Pythia


    # ------------------------------- HAIC --------------------------------
    # engine = SubprocessEngine('haic', 'xhn_s.small', n_resource=1)
    # engine = SubprocessEngine('haic', 'xhn_s.large', n_resource=1)

    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=1)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=2)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=3)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=4)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=5)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=6)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=7)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=8)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=10)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=11)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=12)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=16)

    engine = QsubEngine('haic', 'xhn_l.large', n_resource=4)

    skip_if_exists = False
    # skip_if_exists = True



    hours = 24





















    # ------------------------------------ fixed settings -------------------------------------------


    # ------------------------------- ABCI --------------------------------
    # engine = QsubEngine('ABCI', 'rt_G.small', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_G.large', n_resource=1)

    # engine = QsubEngine('ABCI', 'rt_F', n_resource=1)   # <= 10B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=2)   # >= 10B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=16)   # 70B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=32)   # 70B model



    # run_mode = 'vanilla'
    # run_mode = 'torchrun'
    run_mode = 'deepspeed'

    dry_run = False

    resume_from_checkpoint = None
    # resume_from_checkpoint = './outputs/01.train.py/checkpoint.2024-02-18'

    # take_interval_between_jobs = False
    take_interval_between_jobs = True


    context_lengths = [
        2048,
        # 4096,
    ]

    proof_intermediate_steps_args = [
        # 'include',
        # 'exclude',
        'randomly_include',   # the best
    ]



    instruction_args = [
        # False,       # better for chat-model?
        True,      # better for non-chat model, somehow.
    ]

    float_precision = 'bf16'

    save_model_on_eval = True
    save_model_at_end = False

    seeds = [
        0,
        # 1,
        # 2,
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

    epoch = None


    warmup_ratio = None
    warmup_steps = None
    steps_upper = None
    train_effective_batch_size = None

    i_job = 0
    for logic_dataset_uname in logic_dataset_unames:
        if logic_dataset_uname == 'hf.hitachi-nlp/FLD.v2__default' or logic_dataset_uname.find('proofwriter') >= 0:
            logic_dataset_concatenate_all_configs = True
            logic_dataset_concatenate_all_splits_into_train = True
        else:
            logic_dataset_concatenate_all_configs = False
            logic_dataset_concatenate_all_splits_into_train = False

        for context_len in context_lengths:

            region = engine.region
            n_cpus_per_node, n_gpus_per_node, n_total_gpus, gpu_name_for_batch_size = get_qsub_cpu_gpu_setting(engine, context_len, run_mode)
            is_V100 = engine.resource.find('rt_G') >= 0 or engine.resource.find('rt_F') >= 0

            for logic_dataset_prob, other_datasets, streaming in multitask_setting_list:
                for learning in learnings:

                    for optimizer, rec_adam_regularization, rec_adam_target_task_weight, rec_adam_anneal_schedule, rec_adam_fisher_coef in optimizer_setings:

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
                                            should_force_fp32 = (
                                                model_name.find('t5-') >= 0\
                                                or (is_V100\
                                                    and (model_name.find('rinna/japanese-gpt2-medium') >= 0\
                                                         or model_name.find('llama') >= 0\
                                                         or os.path.exists(model_name + '/config.json') and json.load(open(model_name + '/config.json')).get('_name_or_path', '').find('llama') >= 0)
                                                    )
                                            )
                                            if should_force_fp32 and float_precision in ['fp16', 'bf16']:
                                                logger.warning(f'Forcing to use fp32 for {model_name}.')
                                                fp32 = True
                                                fp16 = False
                                                bf16 = False
                                            elif float_precision == 'fp32':
                                                fp32 = True
                                                fp16 = False
                                                bf16 = False
                                            elif float_precision == 'fp16':
                                                fp32 = False
                                                fp16 = True
                                                bf16 = False
                                            elif float_precision == 'bf16':
                                                fp32 = False
                                                fp16 = False
                                                bf16 = True

                                            for lrate in lrates:
                                                lrate_org = lrate
                                                if optimizer == 'rec_adam':
                                                    lrate = lrate * 2

                                                for update_parameters in update_parameters_args:
                                                    for from_scratch in from_scratch_args:
                                                        for instruction in instruction_args:

                                                            setting = {}

                                                            setting.update(get_base_setting(base_setting_name))

                                                            setting.update(
                                                                get_learning_setting(
                                                                    learning,
                                                                    epoch=epoch,
                                                                    steps_upper=steps_upper,
                                                                    warmup_steps=warmup_steps,
                                                                    warmup_ratio=warmup_ratio,
             
                                                                    optimizer=optimizer,
                                                                    rec_adam_regularization=rec_adam_regularization,
                                                                    rec_adam_anneal_type='sigmoid',
                                                                    rec_adam_target_task_weight=rec_adam_target_task_weight,
                                                                    rec_adam_fisher_coef=rec_adam_fisher_coef,
                                                                    rec_adam_anneal_schedule=rec_adam_anneal_schedule,

                                                                    update_parameters=update_parameters,

                                                                    train_effective_batch_size=train_effective_batch_size,
                                                                    num_evals=num_evals,
                                                                    # max_eval_samples=max_eval_samples,

                                                                    n_gpus=n_total_gpus,
                                                                )
                                                            )


                                                            other_dataset_probs = [other_dataset[0] for other_dataset in other_datasets]
                                                            other_dataset_names = [other_dataset[1] for other_dataset in other_datasets]
                                                            other_dataset_config_names = [other_dataset[2] for other_dataset in other_datasets]
                                                            setting.update(
                                                                get_dataset_setting(
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
                                                                    save_model_at_end=save_model_at_end,
                                                                    save_model_on_eval=save_model_on_eval,
                                                                )
                                                            )

                                                            setting.update(
                                                                get_batch_setting(
                                                                    gpu_name=gpu_name_for_batch_size,
                                                                    n_gpus=n_total_gpus,
                                                                    model_name=model_name_for_batch_size,
                                                                    train_effective_batch_size=setting.get('train_effective_batch_size', None),
                                                                    batch_size_per_gpu_factor = 1/2 if fp32 or optimizer == 'rec_adam' else 1.0,
                                                                )
                                                            )

                                                            # if run_mode == 'deepspeed':
                                                            #     for max_eval_arg_name in ['logic_eval_max_samples']:
                                                            #         max_eval_arg_sample = setting.get(max_eval_arg_name, None)
                                                            #         if max_eval_arg_sample is not None and setting['eval_effective_batch_size'] > max_eval_arg_sample:
                                                            #             raise ValueError(f'{max_eval_arg_name}={max_eval_arg_sample} should be larger than eval_effective_batch_size={setting["eval_effective_batch_size"]}, as it will lead to exception')

                                                            setting.update(get_model_setting(model_name, from_scratch=from_scratch))
                                                            setting.update(get_tokenizer_setting(model_name))
                                                            setting.update(
                                                                get_generation_setting(
                                                                    generation_max_length=setting.get('max_target_length', None),
                                                                    generation_max_prompt_length=setting.get('max_prompt_length', None),
                                                               ),
                                                            )
                                                            setting.update({
                                                                'do_train': True,
                                                                # 'do_eval': True,   # automatically set by evaluation_strategy=step
                                                                'do_eval_in_outerloop': False,
                                                                'do_predict': False,
                                                            })
                                                            setting.update({
                                                                'seed': seed,

                                                                'logic_dataset_uname': logic_dataset_uname,
                                                                # 'other_dataset_name': other_dataset_names,    # should avoid list in the setting
                                                                # 'other_dataset_config_name': other_dataset_config_names,

                                                                'logic_dataset_concatenate_all_configs': logic_dataset_concatenate_all_configs,
                                                                'logic_dataset_concatenate_all_splits_into_train': logic_dataset_concatenate_all_splits_into_train,

                                                                'resume_from_checkpoint': resume_from_checkpoint,

                                                                'base_setting_name': base_setting_name,

                                                                'lm_type': lm_type,
                                                                'float_precision': float_precision,
                                                                'fp16': fp16,
                                                                'bf16': bf16,

                                                                # 'save_total_limit': save_total_limit,

                                                                # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                                                                'learning': learning,
                                                                'sample_negative_proof': sample_negative_proof,
                                                                'proof_intermediate_steps': proof_intermediate_steps,
                                                                'no_subproof_for_unknown': no_subproof_for_unknown,

                                                                'learning_rate': lrate,
                                                                'weight_decay': weight_decay,

                                                                # 'preprocessing_num_workers': max(1, int(n_cpus_per_node / n_gpus_per_node)),
                                                                # 'preprocess_batch_size': 1000,

                                                                # 'preprocessing_num_workers': max(1, int(n_cpus_per_node / n_gpus_per_node / 2)),
                                                                # 'preprocess_batch_size': 500,

                                                                # [XXX] may hang???
                                                                'preprocessing_num_workers': max(1, int(n_cpus_per_node)),
                                                                'preprocess_batch_size': 1000,

                                                                # 'dataloader_num_workers': max(1, int(n_cpus_per_node / n_gpus_per_node)),

                                                                'ddp_timeout': 3600 * 10,

                                                                'gpu_name_for_batch_size': gpu_name_for_batch_size,
                                                                'use_auth_token': True,
                                                                'log_examples': True,
                                                            })

                                                            if seed >= 2:  # for compatibility with older experiments of jpn
                                                                setting.update({
                                                                    'train_random_sampling': True,
                                                                    'eval_random_sampling': True,
                                                                    'logic_eval_random_sampling': True,
                                                                })

                                                            output_dir = make_output_dir(setting, output_top_dir)
                                                            if skip_if_exists and (output_dir / 'log.txt').exists():
                                                                logger.info(f'Skipping "{output_dir}"')
                                                                continue
                                                            command = make_command(output_dir,
                                                                                   setting,
                                                                                   run_mode,
                                                                                   region,
                                                                                   n_gpus_per_node=n_gpus_per_node)

                                                            run_by_engine(
                                                                engine,
                                                                command,
                                                                output_dir,
                                                                # delay = i_job * 0.5,
                                                                hours=_hours,
                                                                force=True,  # assuming that we do not have many jobs
                                                                dry_run=dry_run
                                                            )
                                                            i_job += 1

                                                            if streaming and take_interval_between_jobs:
                                                                logger.info('sleep for a wihle to avoid "Too many requests" exception for huggingface hub')
                                                                time.sleep(60 * 10)

                                                        lrate = lrate_org
                                                        engine.n_resource = n_resouce_org

    logger.info('------------- ./01.train.py finished !! -----------')



if __name__ == '__main__':
    main()
