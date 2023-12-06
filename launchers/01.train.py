#!/usr/bin/env python
import logging
from pathlib import Path

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
    get_qsub_gpu_setting,
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

    # output_top_dir = Path('./outputs/01.train.py/20230807.all_at_once')

    # output_top_dir = Path('./outputs/01.train.py/20230919.jpn')
    # output_top_dir = Path('./outputs/01.train.py/20230919.jpn.seed--1')

    # output_top_dir = Path('./outputs/01.train.py/20231005.jpn.seed--0')
    # output_top_dir = Path('./outputs/01.train.py/20231008.run_causal_prover')

    # output_top_dir = Path('./outputs/01.train.py/20231008.jpn.run_causal_prover')
    # output_top_dir = Path('./outputs/01.train.py/20231009.jpn.run_causal_prover')
    # output_top_dir = Path('./outputs/01.train.py/20231009.run_causal_prover.large_models')
    # output_top_dir = Path('./outputs/01.train.py/debug')

    # output_top_dir = Path('./outputs/01.train.py/20231010.run_causal_prover.large_models')
    # output_top_dir = Path('./outputs/01.train.py/20231010.run_causal_prover.large_models.save_models')
    # output_top_dir = Path('./outputs/01.train.py/20231010.large_vocab.small')
    # output_top_dir = Path('./outputs/01.train.py/20231012.large_vocab')
    # output_top_dir = Path('./outputs/01.train.py/20231012.large_vocab.other_corpus')
    # output_top_dir = Path('./outputs/01.train.py/debug')

    # output_top_dir = Path('./outputs/01.train.py/20231012.large_vocab.other_corpus')
    # output_top_dir = Path('./outputs/01.train.py/20231012.large_vocab.other_corpus')
    # output_top_dir = Path('./outputs/01.train.py/20231103.knowledge')

    # output_top_dir = Path('./outputs/01.train.py/20231203.jpn')
    # output_top_dir = Path('./outputs/01.train.py/20231203.jpn.no_subproof_for_unknown')
    output_top_dir = Path('./outputs/01.train.py/20231206.new_models')

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
    ]

    FLD_dataset_unames = [

        # ---------------------------------- 20230729.case_study_finalize ------------------------------------
        # '20230729.case_study_finalize.D3',
        # '20230729.case_study_finalize.D8',

        # 'hf.hitachi-nlp/FLD.v2__default',
        # 'hf.hitachi-nlp/FLD.v2__star',

        # ---------------------------------- 20230826.jpn ------------------------------------
        # '20230826.jpn.D3',
        # '20230826.jpn.D8',

        # ---------------------------------- 20230916.jpn ------------------------------------
        # '20230916.jpn.D1_wo_dist',
        # '20230916.jpn.D1',
        # '20230916.jpn.D3',
        # '20230916.jpn.D5',

        # ---------------------------------- 20231010.D3.large_vocab ------------------------------------
        # '20231010.D3.large_vocab',

        # ---------------------------------- 20231012.D3.large_vocab ------------------------------------
        # '20231012.D3.large_vocab',
        # '20231012.D3.large_vocab.smpl_stncs',
        # '20231012.D3.large_vocab.smpl_stncs.cntx_shffls-3',
        # '20231012.D3.large_vocab.smpl_stncs.cntx_shffls-3.trnsl_vrnts-3',

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

        # ---------------------------------- 20231203.jpn ------------------------------------
        '20231203.jpn.D1_wo_dist',
        # '20231203.jpn.D1',
        # '20231203.jpn.D3',
        # '20231203.jpn.D8',
    ]

    # other_dataset_name = "wikitext"
    # other_dataset_config_name = "wikitext-2-raw-v1"

    # other_dataset_name = "cerebras/SlimPajama-627B"
    # other_dataset_config_name = None

    # [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B)
    # other_dataset_name = "cerebras/SlimPajama-627B"
    # other_dataset_config_name = "None"

    other_dataset_name = None
    other_dataset_config_name = None

    model_settings = [
        # ============================ english      ============================

        # # # -------------- < 1B params --------------
        # ('t5-base', 'seq2seq', 't5-base'),

        # ('gpt2-medium', 'causal', 'gpt2-medium.short_cntx'),  # XXX: context is short, only  for debug
        # ('gpt2-medium', 'causal', 'cyberagent/open-calm-medium'),


        # # # # # -------------- > 1B params --------------

        # ('PY007/TinyLlama-1.1B-intermediate-step-480k-1T', 'causal', 'cyberagent/open-calm-3b'),   # much better than "PY007/TinyLlama-1.1B-Chat-v0.3"
        # ('PY007/TinyLlama-1.1B-Chat-v0.3', 'causal', 'cyberagent/open-calm-3b'),

        # ('meta-llama/Llama-2-7b', 'causal', 'cyberagent/open-calm-1b-short-ctx')
        # ('meta-llama/Llama-2-7b-hf', 'causal', 'cyberagent/open-calm-1b-short-ctx')
        # ('meta-llama/Llama-2-7b-chat-hf', 'causal', 'cyberagent/open-calm-1b-short-ctx')



        # ============================ multilingual ============================
        # ('google/mt5-base', 'seq2seq', 'google/mt5-base'),
        # ('google/mt5-large', 'seq2seq', 'google/mt5-large'),

        # TODO: other models such as mBART



        # # # ============================ japanese     ============================

        # -- V100 x 4 x 1 nodes --

        # ('retrieva-jp/t5-xl', 'seq2seq', 'retrieva-jp/t5-xl'),

        # ('elyza/ELYZA-japanese-Llama-2-7b-fast', 'causal', 'matsuo-lab/weblab-10b'),
        # ('elyza/ELYZA-japanese-Llama-2-7b-fast-instruct', 'causal', 'matsuo-lab/weblab-10b'),

        # # ('cyberagent/open-calm-1b', 'causal', 'cyberagent/open-calm-1b'),
        # # ('cyberagent/open-calm-3b', 'causal', 'cyberagent/open-calm-3b'),
        # ('cyberagent/open-calm-7b', 'causal', 'cyberagent/open-calm-7b'),
        # ('cyberagent/calm2-7b', 'causal', 'cyberagent/open-calm-7b'),   # NEW
        # ('cyberagent/calm2-7b-chat', 'causal', 'cyberagent/open-calm-7b'),   # NEW

        # # ('line-corporation/japanese-large-lm-1.7b', 'causal', 'cyberagent/open-calm-1b'),
        # # ('line-corporation/japanese-large-lm-1.7b-instruction-sft', 'causal', 'cyberagent/open-calm-1b'),
        # ('line-corporation/japanese-large-lm-3.6b', 'causal', 'cyberagent/open-calm-3b'),
        # ('line-corporation/japanese-large-lm-3.6b-instruction-sft', 'causal', 'cyberagent/open-calm-3b'),

        # ('rinna/japanese-gpt-neox-3.6b', 'causal', 'cyberagent/open-calm-3b'),
        # # ('rinna/japanese-gpt-neox-3.6b-instruction-sft-v2', 'causal', 'cyberagent/open-calm-3b'),
        # ('rinna/japanese-gpt-neox-3.6b-instruction-ppo', 'causal', 'cyberagent/open-calm-3b'),

        # ('stabilityai/japanese-stablelm-base-alpha-7b', 'causal', 'matsuo-lab/weblab-10b'),
        ('stabilityai/japanese-stablelm-instruct-alpha-7b-v2', 'causal', 'matsuo-lab/weblab-10b'),   # NEW
        

        # -- V100 x 4 x 2 nodes --
        # ('matsuo-lab/weblab-10b', 'causal', 'matsuo-lab/weblab-10b'),
        # ('matsuo-lab/weblab-10b-instruction-sft', 'causal', 'matsuo-lab/weblab-10b'),

        # ('stockmark/stockmark-13b', 'causal', 'matsuo-lab/weblab-10b'),   # NEW
        # ('pfnet/plamo-13b', 'causal', 'matsuo-lab/weblab-10b')

        # ('llm-jp/llm-jp-13b-v1.0', 'causal', 'matsuo-lab/weblab-10b'),   # NEW
        # ('llm-jp/llm-jp-13b-instruct-full-jaster-v1.0', 'causal', 'matsuo-lab/weblab-10b'),   # NEW

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
    ]

    # script_type = 'run_prover'
    script_type = 'run_causal_prover'

    # seq2seq_proof_sampling = 'stepwise'
    seq2seq_proof_sampling = 'all_at_once'

    learnings = [
        # 'debug.ZS',
        # 'debug.step-10',
        # 'debug.micro',
        # 'debug.micro.deepspeed',
        # 'debug.tiny',
        # 'debug.middle',
        # 'debug.large',
        # 'debug.find_batch_size',
        # 'debug.20000.zero_warmup',

        # 'FS.shot-0',
        # 'FS.shot-10',
        # 'FS.shot-100',
        # 'FT.step-5000',
        # 'FT.step-10000',
        # 'FT.step-20000',
        # 'FT.step-50000',
        # 'FT.step-100000',

        # ---- JFLD experiments ----
        # 'LLM_FS.shot-10',
        'LLM_FS.shot-100',
        # 'LLM_FS.shot-1000',
        # 'LLM_FS.shot-10000',
    ]

    seeds = [
        0,
        # 1,
    ]

    lrates = [
        # ==== script_type = 'run_prover' =====

        # -- learning = 'FT' ---
        # 1e-4,

        # -- learning = 'LLM_FS' ---
        # 1e-5,
        # 3e-5,   # might be better but not checked


        # ==== script_type = 'run_causal_prover' =====

        # -- learning = 'FT' ---
        # 1e-5,
        # 1e-4,   # much better than 1e-05

        # -- learning = 'LLM_FS' ---
        # 3e-5,   # 20230919.jpn
        1e-5,   # NLP_2024
    ]

    streaming = False
    # streaming = True

    instruction_args = [
        False,       # better for chat-model?
        # True,      # better for non-chat model, somehow.
    ]

    # run_mode = 'vanilla'
    # run_mode = 'torchrun'
    run_mode = 'deepspeed'

    # engine = SubprocessEngine()
    engine = QsubEngine('ABCI', 'rt_G.large', n_resource=1)
    # engine = QsubEngine('ABCI', 'rt_F', n_resource=2)   # XXX only for weblab, plamo

    if isinstance(engine, SubprocessEngine):
        # n_gpus = 1  # debug
        n_gpus = 4
        # n_gpus = None  # specify this when running through QsubEngine

        # gpu_name_for_batch_size = 'A100_48_1'
        # gpu_name_for_batch_size = 'V100_16_1'
        # gpu_name_for_batch_size = 'V100_16_4'
        gpu_name_for_batch_size = 'V100_16_4.deepspeed'
        # gpu_name_for_batch_size = None   # specify this when running through QsubEngine

    hours = 12
    # hours = 24

    save_model = False
    # save_model = True

    # dry_run = True
    dry_run = False

    # --------------------------- fixed settings ---------------------------------
    if isinstance(engine, QsubEngine):
        n_gpus, gpu_name_for_batch_size = get_qsub_gpu_setting(engine, run_mode)

    base_setting_name = 'default'

    # slow generatoin is most likely the repetitions coming from underfitting, so we discard such generations.
    generation_timeout = 60 * 10  # For LLMs

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
    train_effective_batch_size = None
    steps_upper = None
    warmup_steps = None
    max_eval_samples = None
    num_evals = None

    for FLD_dataset_uname in FLD_dataset_unames:
        for learning in learnings:
            for sample_negative_proof in sample_negative_proof_args:
                for no_subproof_for_unknown in no_subproof_for_unknown_args:
                    for seed in seeds:
                        for model_name, lm_type, model_name_for_batch_size in model_settings:
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
                                        )
                                    )

                                    setting.update(
                                        get_dataset_setting(
                                            script_type,
                                            dataset_uname=FLD_dataset_uname,
                                            top_dirs=DATASETS_DIRS,
                                            other_dataset_name=other_dataset_name,
                                            other_dataset_config_name=other_dataset_config_name,
                                            use_test_as_val=setting.get('use_test_as_val', False),
                                            use_test_as_train=setting.get('use_test_as_train', False),
                                            streaming=streaming,
                                            instruction=instruction,
                                        )
                                    )

                                    setting.update(
                                        get_save_eval_step_setting(
                                            max_steps=setting['max_steps'],
                                            eval_steps=setting['eval_steps'],
                                            do_save_model=save_model,
                                        )
                                    )

                                    setting.update(
                                        get_batch_setting(
                                            script_type,
                                            gpu_name=gpu_name_for_batch_size,
                                            n_gpus=n_gpus,
                                            model_name=model_name_for_batch_size + '.all_at_once' if proof_sampling == 'all_at_once' else model_name_for_batch_size,
                                            train_effective_batch_size=setting.get('train_effective_batch_size', None),
                                        )
                                    )

                                    setting.update(get_model_setting(model_name))

                                    setting.update(get_tokenizer_setting(model_name))

                                    setting.update(get_generation_setting(script_type, generation_timeout=generation_timeout))

                                    setting.update({
                                        'do_train': True,
                                        # 'do_eval': True,   # automatically set by evaluation_strategy=step
                                        'do_eval_in_outerloop': False,
                                        'do_predict': False,
                                    })

                                    setting.update({
                                        'script_type': script_type,
                                        'seed': seed,

                                        'FLD_dataset_uname': FLD_dataset_uname,
                                        'other_dataset_name': other_dataset_name,
                                        'other_dataset_config_name': other_dataset_config_name,

                                        'base_setting_name': base_setting_name,

                                        'lm_type': lm_type,
                                        'fp16': model_name.find('t5-') < 0 and model_name.find('rinna/japanese-gpt2-medium') < 0,

                                        # 'save_total_limit': save_total_limit,

                                        # 'trainer_ckpt_for_resume_training': None,  # Specify if you want to resume training
                                        'proof_sampling': proof_sampling,
                                        'learning': learning,
                                        'sample_negative_proof': sample_negative_proof,
                                        'no_subproof_for_unknown': no_subproof_for_unknown,

                                        'lr_scheduler_type': 'linear',
                                        'learning_rate': lrate,
                                        'weight_decay': 0.0,

                                        # 'n_gpu': 1,
                                        'dataloader_num_workers': 0,

                                        'lora': False,

                                        'gpu_name_for_batch_size': gpu_name_for_batch_size,
                                        'use_auth_token': True,
                                        'log_examples': True,
                                    })

                                    output_dir = make_output_dir(setting, output_top_dir)
                                    command = make_command(script_type,
                                                           output_dir,
                                                           setting,
                                                           run_mode,
                                                           n_gpus=n_gpus)

                                    run_by_engine(
                                        engine,
                                        command,
                                        output_dir,
                                        hours=hours,
                                        dry_run=dry_run
                                    )

    logger.info('------------- ./01.train.py finished !! -----------')


if __name__ == '__main__':
    main()
