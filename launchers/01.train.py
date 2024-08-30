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






DATASETS_DIRS = [

    # './outputs.FLD/00.create_corpus/20230729.case_study_finalize',
    # './outputs.FLD/00.create_corpus/20230801.case_study_finalize.fix',

    # './outputs.FLD/00.create_corpus/20231203.jpn',
    # './outputs.FLD/00.create_corpus/20231213.jpn',
    # './outputs.FLD/00.create_corpus/20230120.jpn.large',


    # './outputs.FLD/00.create_corpus/20230120.jpn.punipuni',
    

    # './outputs.FLD/00.create_corpus/2024-03-29',
    # './outputs.FLD/00.create_corpus/2024-05-03.ablation',
    # './outputs.FLD/00.create_corpus/2024-05-08.ref_prob',
    # './outputs.FLD/00.create_corpus/2024-05-19.ablation_with_theorems/',
    # './outputs.FLD/00.create_corpus/2024-06-08.LPT',
    # './outputs.FLD/00.create_corpus/2024-06-19.transfer',


    # './outputs.FLD/00.create_corpus/2024-07-21.neurips_additional',
    # './outputs.FLD/00.create_corpus/2024-08-09.depth_fix',

    # './outputs.FLD/00.create_corpus/2024-08-10.rerun-4288b3b',
    # './outputs.FLD/00.create_corpus/2024-08-10.rerun-2cab8a2',
    # './outputs.FLD/00.create_corpus/2024-08-10.rerun-618e7c3.2024-03-29.FLD_v2',
    # './outputs.FLD/00.create_corpus/2024-08-10.rerun-618e7c3.2024-03-29.FLD_v2.proc-18',

    './outputs.FLD/00.create_corpus/2024-08-12.neurips_camera_ready.towards_best_corpora',

    './outputs.FLD-augmentation/00.augment.py/2024-08-25',

    # './outputs.FLD/00.create_corpus/2024-08-12.neurips_camera_ready.towards_best_corpora/dataset_name=2024-08-16.neurips_camera_ready.FLD.small_vocab',

    './outputs.FLD/00.create_corpus/2024-08-30.fix_ref_prob',
]







@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # =================================== neurips.additional ===================================
    # output_top_dir = Path('./outputs/01.train.py/2024-06-22.neurip.additional')
    # output_top_dir = Path('./outputs/01.train.py/2024-06-22.neurip.additional.rebuttal')


    # =================================== LPT ===================================

    # output_top_dir = Path('./outputs/01.train.py/2024-07-08.ALPT_first')
    # output_top_dir = Path('./outputs/01.train.py/2024-07-08.steps')

    # output_top_dir = Path('./outputs/01.train.py/2024-07-08.ALPT.re_run')
    # output_top_dir = Path('./outputs/01.train.py/2024-07-08.ALPT.re_run.LPT')

    # output_top_dir = Path('./outputs/01.train.py/2024-07-09.BLPT.re_run.scratch')
    # output_top_dir = Path('./outputs/01.train.py/2024-07-09.BLPT.re_run.scratch.PT')


    # =================================== ALPT_strong ===================================
    # output_top_dir = Path('./outputs/01.train.py/2024-08-05.ALPT_strong')


    # =================================== 2024-08-12.neurips_camera_ready.towards_best_corpora ========================================
    # output_top_dir = Path('./outputs/01.train.py/2024-08-12.neurips_camera_ready.towards_best_corpora')


    # =================================== 2024-08-16.neurips_camera_ready ========================================
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready')
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.1')
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.2')
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.3')
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.4')
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.5')
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.6.rec_adam')

    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.7.proof_intermediate_steps_prob')
    # output_top_dir = Path('./outputs/01.train.py/2024-08-16.neurips_camera_ready.7.proof_intermediate_steps_prob.deepspeed_fix')

    # output_top_dir = Path('./outputs/01.train.py/2024-08-30.fix_ref_prob')



    # =================================== 2024-08-26.really_camera_ready ========================================
    # XXX: REALY CAMERA READY
    output_top_dir = Path('./outputs/01.train.py/2024-08-26.really_camera_ready')



    # =================================== ./outputs.FLD-augmentation/00.augment.py/2024-08-25 ========================================
    # output_top_dir = Path('./outputs/01.train.py/00.augment.py.2024-08-25')





    model_settings = [

        # ============================ english      ============================

        # ('meta-llama/Llama-2-7b-hf', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Llama-2-70b-hf', 'causal', 'meta-llama/Llama-2-7b-hf'),

        # ('meta-llama/Meta-Llama-3-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('Qwen/Qwen1.5-7B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('Qwen/Qwen1.5-72B', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('mistralai/Mistral-7B-v0.1', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('mistralai/Mixtral-8x7B-v0.1', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('mistralai/Mistral-Nemo-Base-2407', 'causal', 'meta-llama/Llama-2-13b-hf'),

        # ('gpt2-medium', 'causal', 'gpt2-medium.short_cntx'),   # for debug

        # see [this paper](https://arxiv.org/abs/2401.16818) for comparison of 1B-class models
        # ('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 'causal', 'cyberagent/open-calm-3b'),

        # ('meta-llama/Meta-Llama-3-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3-8B-Instruct', 'causal', 'meta-llama/Llama-2-7b-hf'),

        # ('meta-llama/Meta-Llama-3-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),
        # ('meta-llama/Meta-Llama-3-70B-Instruct', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('meta-llama/Meta-Llama-3.1-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3.1-8B-Instruct', 'causal', 'meta-llama/Llama-2-7b-hf'),

        # ('meta-llama/Meta-Llama-3.1-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),
        # ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('Qwen/Qwen2-7B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('Qwen/Qwen2-72B', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('google/gemma-2-9b', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('google/gemma-2-27b', 'causal', 'google/gemma-2-27b'),

        # XXX does not work for now
        # ('microsoft/Phi-3-small-8k-instruct', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('microsoft/Phi-3-medium-4k-instruct', 'causal', 'meta-llama/Llama-2-13b-hf'),



        # ============================ japanese     ============================

        # ('tokyotech-llm/Swallow-13b-hf', 'causal', 'matsuo-lab/weblab-10b'),
        # ('tokyotech-llm/Swallow-70b-hf', 'causal', 'tokyotech-llm/Swallow-70b-hf'),
        # ('tokyotech-llm/Swallow-70b-instruct-hf', 'causal', 'tokyotech-llm/Swallow-70b-hf'),


        # ============================ LPT     ============================

        # ('EleutherAI/pythia-1b', 'causal', 'EleutherAI/pythia-1b'),
        # ('EleutherAI/pythia-6.9b', 'causal', 'meta-llama/Llama-2-7b-hf'),

        # ALPT after LPT
        # ('mdl_nm=EleutherAI@pythia-6.9b__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing.large__dtst_nms=cerebras@SlimPajama-627B__lgc_dtst_prb=0.0__lrnng=LPT.bs-2048__step-12000.wrmp-100__lrnng_rt=0.0003.chk-12000', 'causal', 'meta-llama/Llama-2-7b-hf')

        # logic before LPT
        # ('mdl_nm=EleutherAI@pythia-6.9b__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing.large__dtst_nms=None__lgc_dtst_prb=1.0__lrnng=FT.bs-1024__step-2930.wrmp-300__lrnng_rt=3e-05__augmnttn=False.chk-2928', 'causal', 'meta-llama/Llama-2-7b-hf'),

        # ('mdl_nm=EleutherAI@pythia-6.9b__lgc_dtst_unm=2024-03-29.JSAI_best.no_aug.trnsl-thing.large__dtst_nms=None__lgc_dtst_prb=1.0__lrnng=FT.bs-1024__step-9765.wrmp-300__lrnng_rt=0.0001__rc_adm_fshr_cf=None__augmnttn=False.chk-9765', 'causal', 'meta-llama/Llama-2-7b-hf'), 


        # ======================================================== neurips camera ready     ========================================================

        # ('meta-llama/Meta-Llama-3.1-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        ('meta-llama/Meta-Llama-3.1-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),
        
        # ('mistralai/Mistral-7B-v0.1', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('mistralai/Mixtral-8x7B-v0.1', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('Qwen/Qwen2-7B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('Qwen/Qwen2-72B', 'causal', 'meta-llama/Llama-2-70b-hf'),


        # ======================================================== augment     ========================================================
        # ('meta-llama/Meta-Llama-3.1-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3.1-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('meta-llama/Meta-Llama-3.1-8B-Instruct', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'causal', 'meta-llama/Llama-2-70b-hf'),


    ]







    logic_dataset_unames = [


        # ---------------------------------- NeurIPS 2024 ------------------------------------

        # 'hf.hitachi-nlp/ruletaker',
        # 'hf.hitachi-nlp/PARARULE-Plus',
        # 'hf.hitachi-nlp/FLD.v2__default',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.2',



        # ------------------------------- 2024-05-03.ablation --------------------------------
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.voc-100',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.dstrct-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.rule-G_MP',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-3-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-1',



        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.rule-G_MP.stps-3',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small.trnsl-old',



        # ------------------------------------- ./outputs/00.create_corpus/2024-05-19.ablation_with_theorems --------------------------
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.voc-100',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.dstrct-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.stps-3-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.20.theorems-0.1.transl_sttng-1',



        # =================================== neurips.additional ===================================

        # 'hf.hitachi-nlp/ruletaker',
        # 'hf.hitachi-nlp/PARARULE-Plus',
        # 'hf.hitachi-nlp/FLD.v2__default',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing',
        # 'hf.hitachi-nlp/proofwriter_processed_OWA__depth-3ext',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.voc-100.fixed',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.voc-50.fixed',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.dstrct-0',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-3-0',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10.stps-1-2',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.rule-G_MP',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small.trnsl-old',



        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.voc-50.fixed.ref_prob=0.20.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.rule-G_MP.ref_prob=0.20.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl_sttng-1.ref_prob=0.20.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small.ref_prob=0.20.theorems-0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.transl-small.trnsl-old.ref_prob=0.20.theorems-0.1',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.1.theorems=0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.1.theorems=0.15',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.1.theorems=0.2',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.2',


        # ---------------------------------- LPT ------------------------------------
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.large',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.steps',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.steps-5',


        # ------------------------------------ ALPT_strong ------------------------------------

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.large',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.2',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems=0.03',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems=0.05',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.1.theorems=0.1',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.1.theorems=0.2',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.steps',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.steps-5',

        # '2024-03-29.JSAI_best.no_aug.trnsl-v2',

        # 'hf.hitachi-nlp/FLD.v2__default',
        # '2024-03-29.FLD_v2',


        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems=0.03.adjust_theorems',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems=0.05.adjust_theorems',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.theorems=0.10.adjust_theorems',

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.4-4',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.5-3',



        # =================================== 2024-08-09.fix_depth_problem ========================================

        # '2024-03-29.JSAI_best.no_aug.trnsl-thing',
        # '2024-03-29.JSAI_best.no_aug.trnsl-thing.large',




        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing',

        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.05',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.10',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15',

        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15.steps-3-4',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15.steps-4-4',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15.steps-4-5',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15.steps-5-3',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15.steps-5',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob=0.15.steps-8',

        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob_in_1=0.10',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob_in_1=0.25',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob_in_1=0.5',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob_in_1=0.5.theorems=0.01',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob_in_1=0.5.theorems=0.03',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-thing.ref_prob_in_1=0.5.theorems=0.03.adjust_theorems',

        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-v2',
        # '2024-08-09.depth_fix.2024-03-29.JSAI_best.no_aug.trnsl-v3',

        # '2024-08-09.depth_fix.2024-03-29.FLD_v2',

        # '2024-08-10.rerun-4288b3b.2024-03-29.FLD_v2',
        # '2024-08-10.rerun-2cab8a2.2024-03-29.FLD_v2',
        # '2024-08-10.rerun-618e7c3.2024-03-29.FLD_v2',
        # '2024-08-10.rerun-618e7c3.2024-03-29.FLD_v2.proc-18',
        # '2024-08-10.rerun-618e7c3.2024-03-29.FLD_v2.proc-18.xcs',
        # '2024-08-10.rerun-618e7c3.2024-03-29.FLD_v2.proc-18.large',



        # =================================== 2024-08-12.neurips_camera_ready.towards_best_corpora ========================================

        # 'hf.hitachi-nlp/ruletaker',
        # 'hf.hitachi-nlp/PARARULE-Plus',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2',

        # '2024-08-12.neurips_camera_ready.towards_best_corpora',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.wo_trnsl-v2',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.wo_theorems',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.suppress_dilemma',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.suppress_dilemma.theorems-0.003',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorems-0.0001',


        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--1.0--0.1',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--1.0--0.01',

        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--0.5--0.1',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--0.5--0.01',

        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--0.25--0.1',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--0.25--0.01',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--0.25--0.1.G_MP-3',
        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--0.25--0.1.G_MP-10',

        # '2024-08-12.neurips_camera_ready.towards_best_corpora.theorem--0.10--0.1',




        # =================================== 2024-08-16.neurips_camera_ready ========================================

        # '2024-08-09.depth_fix.2024-03-29.FLD_v2',
        # '2024-08-16.neurips_camera_ready.FLD.small_vocab',


        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0',

        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.wo_suppress_if',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.wo_phrase',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.wo_clause',


        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.ref_prob-0.05',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.ref_prob-0.20',


        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.05',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.15',


        # XXX 途中で G_MPだけのコーパスと混ざってしまった．非推奨．
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.15.w_flag.super_theorems',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.15.w_flag.super_theorems.syllogism',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.15.w_flag.super_theorems.syllogism.contraposition',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.15.w_flag.super_theorems.syllogism.contraposition.and_interchangeability',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.15.w_flag',

        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0.theorem--0.15.w_flag.super_theorems.ref_prob=0.20',


        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v2',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v2.ref_prob-0.1',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v2.ref_prob-0.126',
        # '2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v2.ref_prob-0.1.theorems-0.25',



        # =================================== 2024-08-30.fix_ref_prob ========================================

        # '2024-08-30.trnsl-thing_person-v0.ref_prob-0.20',
        # '2024-08-30.trnsl-thing_person-v0.ref_prob-0.20.theorem-G_MP',
        # '2024-08-30.trnsl-thing_person-v0.ref_prob-0.20.theorem-G_MP.syllogism',
        # '2024-08-30.trnsl-thing_person-v0.ref_prob-0.20.theorem-G_MP.syllogism.contraposition',
        # '2024-08-30.trnsl-thing_person-v0.ref_prob-0.20.theorem-G_MP.syllogism.contraposition.interchangeability',
        # '2024-08-30.trnsl-thing_person-v0.ref_prob-0.20.theorem-all',



        # ====================================== ./outputs/01.train.py/2024-08-26.really_camera_ready ============================

        'hf.hitachi-nlp/ruletaker',
        'hf.hitachi-nlp/PARARULE-Plus',
        # '2024-08-16.neurips_camera_ready.FLD.small_vocab',



        # =================== ./outputs.FLD-augmentation/00.augment.py/2024-08-25 ======================

        # 'AUG__2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0__augmnttn=False__augmnttn_prb=0.5__llm_nm=hf.TechxGenus@Mistral-Large-Instruct-2407-AWQ__prf_intrmdt_stps=randomly_include',
        # 'AUG__2024-08-09.depth_fix.2024-03-29.FLD_v2.trnsl-thing_person-v0__augmnttn=True__augmnttn_prb=0.5__llm_nm=hf.TechxGenus@Mistral-Large-Instruct-2407-AWQ__prf_intrmdt_stps=randomly_include',




    ]


    proof_intermediate_steps_args = [
        'randomly_include',   # the best
        # 'include',
        # 'exclude',
    ]

    proof_intermediate_steps_prob_args = [
        None,
        # 0.33,   # 150k
        # 0.166666,   # 300k

    ]





    learnings = [
        # 'debug.ZS',
        # 'debug.micro',
        # 'debug.tiny',

        # =========================== JFLD experiments ===========================
        # 'LLM_FS.shot-5',
        # 'LLM_FS.shot-100',
        # 'LLM_FS.shot-1000',
        # 'LLM_FS.shot-10000',
        # 'LLM_FS.shot-30000',
        

        # =========================== LPT ===========================
        # 'LPT.bs-2048__step-12000.wrmp-100',
        # 'LPT.bs-2048__step-12000.wrmp-1000',
        # 'LPT.bs-2048__step-13200.wrmp-132',


        # =========================== ALPT_strong ===========================

        # 'debug.FT.bs-64__step-10.wrmp-0',

        # ------------------ 25k examples ------------------
        # 'FT.bs-256__step-98.wrmp-50',

        # ------------------ 50k examples ------------------
        # 'FT.bs-256__step-195.wrmp-100',

        # ------------------ 100k examples ------------------
        'FT.bs-256__step-390.wrmp-200',
        # 'FT.bs-256__step-379.wrmp-200',   # reduce 3%
        # 'FT.bs-256__step-390.wrmp-40',

        # ------------------ 150k examples ------------------
        # 'FT.bs-256__step-586.wrmp-200',

        # ------------------ 200k examples ------------------
        # 'FT.bs-256__step-780.wrmp-200',

        # ------------------ 300k examples ------------------
        # 'FT.bs-768__step-390.wrmp-200',

        # ------------------ 1M examples ------------------
        # 'FT.bs-256__step-3900.wrmp-200',
        # 'FT.bs-2560__step-390.wrmp-200',   # with the same step as 100k

        # ---------- 10M examples ----------
        # 'FT.bs-1024__step-9765.wrmp-300',

    ]



    lrates = [

        #  ================================ LPT ================================
        # 3e-5,    # BLPT, 3M examples
        # 1e-4,    # BLPT, 10M examples

        # 3e-4,    # LPT
        # 1e-4,    # LPT after BLPT


        #  ================================ ALPT_strong ================================
        # 3e-6,
        1e-5,     # the best on 100k examples
        # 3e-5,   # the best on 1M examples

        # 1e-4,
        # 3e-4,
    ]



    # XXX: The fisher coef MUST be tuned for each model,
    # as the optimal value differs much from model to model.
    optimizer_setings = [
        ('rec_adam', 1.0, 'auto'),

        # (None, None, None),
        # ('adamw_hf', None, None),
        # ('rec_adam', 1.0, 0),
        # ('rec_adam', 1.0, 300),
        # ('rec_adam', 1.0, 1000),
        # ('rec_adam', 1.0, 3000),
        # ('rec_adam', 1.0, 5000),

    ]

    # run_mode = 'vanilla'
    # run_mode = 'torchrun'
    run_mode = 'deepspeed'



    # ------------------------------- HAIC --------------------------------

    # engine = SubprocessEngine('haic', 'xhn_s.small', n_resource=1)
    # engine = SubprocessEngine('haic', 'xhn_s.large', n_resource=1)


    # engine = QsubEngine('haic', 'xhn_s.middle2', n_resource=1)
    # hours = 8


    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=1)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=2)
    engine = QsubEngine('haic', 'xhn_s.large', n_resource=4)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=8)
    hours = 8


    # engine = QsubEngine('haic', 'xhn_l.large', n_resource=1)
    # engine = QsubEngine('haic', 'xhn_l.large', n_resource=2)
    # engine = QsubEngine('haic', 'xhn_l.large', n_resource=4)
    # engine = QsubEngine('haic', 'xhn_l.large', n_resource=8)
    # hours = 72
























    # ------------------------------------ fixed settings -------------------------------------------

    # skip_if_exists = False
    skip_if_exists = True

    # ------------------------------- ABCI --------------------------------
    # engine = QsubEngine('ABCI', 'rt_G.small', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_G.large', n_resource=1)

    # engine = QsubEngine('ABCI', 'rt_F', n_resource=1)   # <= 10B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=2)   # >= 10B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=16)   # 70B model
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=32)   # 70B model

    dry_run = False

    weight_decay_args = [
        0.0,
        # 0.01,
        # 0.1,
    ]

    max_grad_norm_args = [
        0.5,
        # 1.0,
        # 3.0,
    ]


    max_train_samples_args = [
        None,
        # 10,
        # 100,
        # 1000,
    ]


    multitask_setting_list = [
        # [datasetライブラリで大規模データセットを扱う]($PROJECTS/NLP/LLM.md)
        # ================================================ NeurIPS 2024 ==============================================

        (
            1.0,
            [],
        ),

        # ================================================ LPT ==============================================

        # (
        #     0.03,
        #     [
        #         (1.0, 'cerebras/SlimPajama-627B', None, None, None)
        #     ],
        # ),


        # ================================================ ALPT_strong ==============================================

        # (
        #     0.5,
        #     [
        #         (1.0, 'DKYoon/SlimPajama-6B', None, None, None)
        #     ],
        # ),


    ]

    # do_sft = True
    do_sft = False





    prompt_indicate_theorems_args = [
        # False,
        True,
    ]

    prompt_emphasize_theorems_args = [
        False,
        # True,
    ]

    augmentation_args = [
        False,
        # True,
    ]

    augmentation_prob_args = [
        # 0.5,
        1.0,
    ]

    # deepspeed_stage = 'zero0'   # 1B models can use this
    # deepspeed_stage = 'zero2'   # 7B can used this, but not that much speedup
    deepspeed_stage = 'zero3'

    resume_from_checkpoint = None
    # resume_from_checkpoint = './outputs/01.train.py/checkpoint.2024-02-18'

    # take_interval_between_jobs = False
    take_interval_between_jobs = True

    from_scratch_args = [
        False,
        # True,
    ]

    lr_scheduler_type = None  # better than 'cosine' for ALPT
    # lr_scheduler_type = 'cosine'

    streaming = False

    update_parameters_args = [
        'all',
        # 'attention',
        # 'mlp',
    ]

    surface_is_formula = False

    max_eval_samples = 10000

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
        if logic_dataset_uname == 'hf.hitachi-nlp/FLD.v2__default':
            logic_dataset_config_load_type = 'concat_all'
            logic_dataset_concatenate_all_splits_into_train = True
        else:
            logic_dataset_config_load_type = None
            logic_dataset_concatenate_all_splits_into_train = False

        if logic_dataset_uname.find('AUG') >= 0:
            use_original_serial = True
            context_len = 3000
        else:
            use_original_serial = False
            context_len = 2048


        for max_train_samples in max_train_samples_args:

            region = engine.region
            n_cpus_per_node, n_gpus_per_node, n_total_gpus, gpu_name_for_batch_size = get_qsub_cpu_gpu_setting(engine, context_len, run_mode)
            is_V100 = engine.resource.find('rt_G') >= 0 or engine.resource.find('rt_F') >= 0

            for logic_dataset_prob, other_dataset_settings in multitask_setting_list:
                for learning in learnings:
                    for optimizer, rec_adam_target_task_weight, rec_adam_fisher_coef in optimizer_setings:
                        for prompt_indicate_theorems in prompt_indicate_theorems_args:
                            for prompt_emphasize_theorems in prompt_emphasize_theorems_args:
                                for sample_negative_proof in sample_negative_proof_args:
                                    for proof_intermediate_steps in proof_intermediate_steps_args:
                                        for proof_intermediate_steps_prob in proof_intermediate_steps_prob_args:
                                            for no_subproof_for_unknown in no_subproof_for_unknown_args:
                                                for seed in seeds:
                                                    for model_name, lm_type, model_name_for_batch_size in model_settings:
                                                        deepspeed_stage_org = deepspeed_stage
                                                        if model_name.find('rwkv') >= 0 or model_name.find('RWKV') >= 0:
                                                            deepspeed_stage = 'zero2'  # as zero3 somehow hangs
                                                            preprocessing_num_workers = 4
                                                        else:
                                                            # preprocessing_num_workers = 1
                                                            # larger value may lead to hangup
                                                            # preprocessing_num_workers = 32
                                                            preprocessing_num_workers = 10

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
                                                            _hours = hours or 24

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
                                                            for weight_decay in weight_decay_args:
                                                                for max_grad_norm in max_grad_norm_args:
                                                                    for update_parameters in update_parameters_args:
                                                                        for from_scratch in from_scratch_args:
                                                                            for augmentation in augmentation_args:
                                                                                for augmentation_prob in augmentation_prob_args:
                                                                                    if augmentation:
                                                                                        instruction = False
                                                                                    else:
                                                                                        instruction = True

                                                                                    setting = {}

                                                                                    setting.update(get_base_setting(base_setting_name))

                                                                                    setting.update(
                                                                                        get_learning_setting(
                                                                                            learning,
                                                                                            epoch=epoch,
                                                                                            steps_upper=steps_upper,
                                                                                            warmup_steps=warmup_steps,
                                                                                            warmup_ratio=warmup_ratio,
                                     
                                                                                            model_name=model_name,
                                                                                            optimizer=optimizer,
                                                                                            max_grad_norm=max_grad_norm,
                                                                                            rec_adam_target_task_weight=rec_adam_target_task_weight,
                                                                                            rec_adam_fisher_coef=rec_adam_fisher_coef,

                                                                                            update_parameters=update_parameters,

                                                                                            train_effective_batch_size=train_effective_batch_size,
                                                                                            num_evals=num_evals,
                                                                                            max_eval_samples=max_eval_samples,
                                                                                            logic_dataset_prob=logic_dataset_prob,

                                                                                            n_gpus=n_total_gpus,
                                                                                        )
                                                                                    )


                                                                                    other_dataset_probs = [other_dataset[0] for other_dataset in other_dataset_settings]
                                                                                    other_dataset_names = [other_dataset[1] for other_dataset in other_dataset_settings]
                                                                                    other_dataset_config_names = [other_dataset[2] for other_dataset in other_dataset_settings]
                                                                                    other_dataset_config_load_types = [other_dataset[3] for other_dataset in other_dataset_settings]
                                                                                    other_dataset_take_n_s = [other_dataset[4] for other_dataset in other_dataset_settings]
                                                                                    setting.update(
                                                                                        get_dataset_setting(
                                                                                            dataset_uname=logic_dataset_uname,
                                                                                            top_dirs=DATASETS_DIRS,
                                                                                            other_dataset_names=other_dataset_names,
                                                                                            other_dataset_config_names=other_dataset_config_names,
                                                                                            other_dataset_config_load_types=other_dataset_config_load_types,
                                                                                            other_dataset_take_n_s=other_dataset_take_n_s,
                                                                                            other_dataset_probs=other_dataset_probs,
                                                                                            use_test_as_val=setting.get('use_test_as_val', use_test_as_val),
                                                                                            use_test_as_train=setting.get('use_test_as_train', use_test_as_train),
                                                                                            streaming=streaming,
                                                                                            use_original_serial=use_original_serial,
                                                                                            surface_is_formula=surface_is_formula,
                                                                                            instruction=instruction,
                                                                                            prompt_indicate_theorems=prompt_indicate_theorems,
                                                                                            prompt_emphasize_theorems=prompt_emphasize_theorems,
                                                                                            augmentation=augmentation,
                                                                                            augmentation_prob=augmentation_prob,

                                                                                            sample_negative_proof=sample_negative_proof,
                                                                                            proof_intermediate_steps=proof_intermediate_steps,
                                                                                            proof_intermediate_steps_prob=proof_intermediate_steps_prob,
                                                                                            no_subproof_for_unknown=no_subproof_for_unknown,


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
                                                                                            # batch_size_per_gpu_factor = 1/2 if fp32 or optimizer == 'rec_adam' else 1.0,
                                                                                            batch_size_per_gpu_factor = 1/2,
                                                                                        )
                                                                                    )

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

                                                                                        'do_sft': do_sft,

                                                                                        'logic_dataset_uname': logic_dataset_uname,
                                                                                        'max_train_samples': max_train_samples,
                                                                                        # 'other_dataset_name': other_dataset_names,    # should avoid list in the setting
                                                                                        # 'other_dataset_config_name': other_dataset_config_names,

                                                                                        'logic_dataset_config_load_type': logic_dataset_config_load_type,
                                                                                        'logic_dataset_concatenate_all_splits_into_train': logic_dataset_concatenate_all_splits_into_train,

                                                                                        'resume_from_checkpoint': resume_from_checkpoint,

                                                                                        'base_setting_name': base_setting_name,

                                                                                        'lm_type': lm_type,
                                                                                        'float_precision': float_precision,
                                                                                        'fp16': fp16,
                                                                                        'bf16': bf16,
                                                                                        'deepspeed_stage': deepspeed_stage,

                                                                                        # 'save_total_limit': save_total_limit,

                                                                                        # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                                                                                        'learning': learning,

                                                                                        'learning_rate': lrate,
                                                                                        'lr_scheduler_type': lr_scheduler_type,
                                                                                        'weight_decay': weight_decay,

                                                                                        # [XXX] may hang???
                                                                                        # 'preprocessing_num_workers': max(1, max(16, int(n_cpus_per_node / n_gpus_per_node))),
                                                                                        # 'preprocessing_num_workers': max(1, n_cpus_per_node - 10),
                                                                                        # 'preprocessing_num_workers': max(1, min(32, n_cpus_per_node)),

                                                                                        # 'preprocessing_num_workers': 10,  # fixすべき．変えるとcache作り直し -> cache sizeが膨れ上がる
                                                                                        'preprocessing_num_workers': preprocessing_num_workers,  # fixすべき．変えるとcache作り直し -> cache sizeが膨れ上がる
                                                                                        'preprocess_batch_size': 500,

                                                                                        # 'dataloader_num_workers': max(1, int(n_cpus_per_node / n_gpus_per_node)),

                                                                                        'preprocess_keep_in_memory': False,

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
                                                                                                           deepspeed_stage=deepspeed_stage,
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

                                                        engine.n_resource = n_resouce_org
                                                        deepspeed_stage= deepspeed_stage_org

    logger.info('------------- ./01.train.py finished !! -----------')



if __name__ == '__main__':
    main()
