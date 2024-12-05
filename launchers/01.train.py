#!/usr/bin/env python
import logging
from pathlib import Path
import time
import os
import json
from itertools import product
import random


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
    
    './outputs.FLD-augmentation/00.augment.py/2024-08-25',
    # './outputs.FLD/00.create_corpus/2024-08-12.neurips_camera_ready.towards_best_corpora/dataset_name=2024-08-16.neurips_camera_ready.FLD.small_vocab',

    # './outputs.FLD/00.create_corpus/2024-08-30.fix_ref_prob',
    # './outputs.FLD/00.create_corpus/2024-09-16.fix_negation',
    './outputs.FLD/00.create_corpus/2024-09-18.fix_negation',
]


OTHER_DATASETS_DIRS = [
    './outputs.factorized_reasoning/',
]








@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)


    # ====================================== NeurIPS_2024 camera ready ========================================
    # output_top_dir = Path('./outputs/01.train.py/2024-09-18.fix_negation')


    # =================================== ./outputs.FLD-augmentation/00.augment.py/2024-08-25 ========================================
    # output_top_dir = Path('./outputs/01.train.py/00.augment.py.2024-08-25')


    # =================================== 2024-09-08.ALPT_strong ========================================
    # output_top_dir = Path('./outputs/01.train.py/2024-09-06.ABCI_debug')
    # output_top_dir = Path('./outputs/01.train.py/2024-09-08.ALPT_strong')
    # output_top_dir = Path('./outputs/01.train.py/2024-09-10.ALPT_strong.fix_rec_adam')


    # =================================== 2024-09-08.factorized_reasoning ========================================
    # output_top_dir = Path('./outputs/01.train.py/2024-09-08.factorized_reasoning')
    # output_top_dir = Path('./outputs/01.train.py/2024-09-10.factorized_reasoning.fix_rec_adam')
    # output_top_dir = Path('./outputs/01.train.py/2024-09-16.factorized_reasoning.fix_rec_adam')
    # output_top_dir = Path('./outputs/01.train.py/2024-09-18.factorized_reasoning')


    output_top_dir = Path('./outputs/01.train.py/2024-12-02.refactor_before_NeurIPS_2024_release')








    model_settings = [
        # ======================================================== NeurIPS_2024 camera ready     ========================================================

        # ('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 'causal', 'cyberagent/open-calm-3b'),

        ('meta-llama/Meta-Llama-3.1-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3.1-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),
        
        # ('mistralai/Mistral-7B-v0.1', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('mistralai/Mixtral-8x7B-v0.1', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('Qwen/Qwen2-7B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('Qwen/Qwen2-72B', 'causal', 'meta-llama/Llama-2-70b-hf'),


        # ======================================================== augment     ========================================================
        # ('meta-llama/Meta-Llama-3.1-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3.1-70B', 'causal', 'meta-llama/Llama-2-70b-hf'),

        # ('meta-llama/Meta-Llama-3.1-8B-Instruct', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'causal', 'meta-llama/Llama-2-70b-hf'),


        # ======================================================== 2024-09-06.factorized_reasoning     ========================================================
        # ('mdl_nm=meta-llama@Meta-Llama-3.1-8B__lgc_dtst_unm=2024-09-03.trnsl-thing_person-v0__optmzr=rec_adam__lrnng=FT.bs-256__step-1953.wrmp-200__lrnng_rt=1e-05__rc_adm_fshr_cf=1000__augmnttn=False__prf_intrmdt_stps=randomly_include__prf_intrmdt_stps_prb=0.5__mx_grd_nrm=0.5__prmpt_indct_thrms=True.chk-1953', 'causal', 'meta-llama/Llama-2-7b-hf'),



        # =================================== 2024-09-08.ALPT_strong ========================================
        # ('meta-llama/Meta-Llama-3.1-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('meta-llama/Meta-Llama-3.1-8B-Instruct', 'causal', 'meta-llama/Llama-2-7b-hf'),


        # =================================== 2024-09-16.factorized_reasoning.fix_rec_adam ========================================
        # ('meta-llama/Meta-Llama-3.1-8B', 'causal', 'meta-llama/Llama-2-7b-hf'),
        # ('mdl_nm=meta-llama@Meta-Llama-3.1-8B__lgc_dtst_unm=2024-09-03.trnsl-thing_person-v0__optmzr=rec_adam__lrnng=FT.bs-256__step-390.wrmp-200__lrnng_rt=1e-05__rc_adm_fshr_cf=3000__augmnttn=False__prf_intrmdt_stps=randomly_include__prf_intrmdt_stps_prb=0.5__mx_grd_nrm=1.0__prmpt_indct_thrms=True__augmnttn_prb=0.5.chk-390', 'causal', 'meta-llama/Llama-2-7b-hf'),

    ]





    logic_dataset_unames = [

        # ===================================== NeurIPS_2024 camera ready ========================================


        # 'hf.hitachi-nlp/PARARULE-Plus',
        # 'hf.hitachi-nlp/ruletaker',
        # '2024-09-18.FLD.neg-0.10.other_seed',
        '2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15=0.90',


        # '2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15.rules-G_MP=0.90',
        # '2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15.voc-100=0.90',
        # '2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15.dstrct-0=0.90',
        # '2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15.stps-3-0=0.90',
        # '2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15.stps-1-0=0.90',
        # '2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15.trnsl-small.trnsl-old=0.90',


        # 'hf.MorishT/FLD2',

    ]

    use_original_serial = True
    # use_original_serial = False



    # XXX: 1つだと，揺らぎで結果の解釈が全くできない．3つくらい回すこと．
    seeds = [
        # 0,
        # 1,
        2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
    ]


    proof_intermediate_steps_prob_args = [
        # 0.0,
        # 0.15,
        # 0.30,
        # 0.40,
        0.50,
        # 0.60,
        # 0.70,
        # 0.80,
        # 1.0,
    ]

    augmentation_prob_args = [
        0.0,
        # 0.25,
        # 0.5,
        # 1.0,
    ]




    multitask_setting_list = [

        # ================================================ NeurIPS 2024 ==============================================
        (1.0, []),


        # ================================================ ALPT_strong ==============================================
        # (0.95, [(1.0, 'hf.DKYoon/SlimPajama-6B', None, None, None)]),
        # (0.80, [(1.0, 'hf.DKYoon/SlimPajama-6B', None, None, None)]),
        # (0.60, [(1.0, 'hf.DKYoon/SlimPajama-6B', None, None, None)]),


        # ================================================ LPT ==============================================
        # (0.03, [(1.0, 'hf.cerebras/SlimPajama-627B', None, None, None)]),


        # ============================================ factorized_reasoning ========================================
        # (1.0, []),

        # (0.0, [(1.0, 'FR.2024-09-08.with_cot.generator=cot', None, None, None)]),
        # (0.0, [(1.0, 'FR.2024-09-08.with_cot.generator=factorized', None, None, None)]),

        # (0.25, [(1.0, 'FR.2024-09-08.with_cot.generator=factorized', None, None, None)]),
    ]

    # do_sft = True
    do_sft = False



    learnings = [
        # 'debug.FT.bs-64__step-10.wrmp-0',


        # 'FT.bs-264__step-380.wrmp-200',  # 100k examples, 3 nodes
        'FT.bs-256__step-390.wrmp-200',  # 100k examples, 4 nodes

        # 'FT.bs-256__step-586.wrmp-200',    # 150k examples

        # 'FT.bs-256__step-780.wrmp-200',  # 200k examples
        # 'FT.bs-512__step-390.wrmp-200',  # 200k examples with same step size as 100k examples

        # 'FT.bs-256__step-1172.wrmp-200',  # 300k examples

        # 'FT.bs-256__step-1953.wrmp-200',  # 500k examples

        # 'FT.bs-256__step-3900.wrmp-200',  # 1M examples


        # ========== factoorized reasoning ==========
        # 'FT.bs-128__step-94.wrmp-20',  # 4k examples x 3 epochs
        # 'FT.bs-128__step-117.wrmp-20',   # (4k examples x 3 epochs) * 1.25

        # 'FT.bs-256__step-390.wrmp-200',  # 100k examples
    ]







    lrates = [
        2e-5,
        # 1e-5,
        # 5e-6,
        # 3e-6,
    ]


    optimizer_setings = [
        # ('rec_adam', 0),
        # ('rec_adam', 100),
        # ('rec_adam', 300),
        # ('rec_adam', 1000),
        # ('rec_adam', 2000),
        # ('rec_adam', 3000),
        ('rec_adam', 4000),
        # ('rec_adam', 4500),
        # ('rec_adam', 5000),

        # ('rec_adam', 5000),
        # (None, None),
    ]



    # engine = SubprocessEngine('haic', 'xhn_s.middle', n_resource=1)

    # engine = QsubEngine('haic', 'xhn_s.middle2', n_resource=2)
    engine = QsubEngine('haic', 'xhn_s.large', n_resource=1)
    # engine = QsubEngine('haic', 'xhn_s.large', n_resource=4)

    # engine = QsubEngine('ABCI', 'rt_F', n_resource=8)    # ~ H100 x 8


    hours = 5






    # run_mode = 'vanilla'
    # run_mode = 'torchrun'
    run_mode = 'deepspeed'


    # skip_if_exists = False
    skip_if_exists = True





















    # ------------------------------------ fixed settings -------------------------------------------





    dry_run = True


    max_train_samples_args = [
        None,
        # 100000,
    ]


    # prompt_emphasize_theorems_args = [
    #     False,
    #     # True,
    # ]

    # deepspeed_stage = 'zero0'   # 1B models can use this
    # deepspeed_stage = 'zero2'   # 7B can used this, but not that much speedup
    deepspeed_stage = 'zero3'

    resume_from_checkpoint = None
    # resume_from_checkpoint = './outputs/01.train.py/checkpoint.2024-02-18'


    from_scratch_args = [
        False,
        # True,
    ]

    lr_scheduler_type = None  # better than 'cosine' for ALPT
    # lr_scheduler_type = 'cosine'

    streaming = False
    take_interval_between_jobs_if_streaming = True

    max_eval_samples = 10000

    save_model_on_eval = True
    save_model_at_end = False

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

    epoch = None

    context_len = None
    # context_len = 3000

    warmup_ratio = None
    warmup_steps = None
    steps_upper = None
    train_effective_batch_size = None

    if engine.region == 'ABCI':
        float_precision = 'fp16'
    elif engine.region == 'haic':
        float_precision = 'bf16'
    else:
        raise ValueError()

    fix_precision = False

    hyparas = product(
        logic_dataset_unames,
        max_train_samples_args,
        multitask_setting_list,
        learnings,
        optimizer_setings,
        # prompt_emphasize_theorems_args,
        sample_negative_proof_args,
        proof_intermediate_steps_prob_args,
        seeds,
        model_settings,
        lrates,
        from_scratch_args,
        augmentation_prob_args,
    )
         
    # iter with hyparas
    for (logic_dataset_uname,
         max_train_samples,
         (logic_dataset_prob, other_dataset_settings),
         learning,
         # (optimizer, rec_adam_target_task_weight, rec_adam_fisher_coef),
         (optimizer, rec_adam_fisher_coef),
         # prompt_emphasize_theorems,
         sample_negative_proof,
         proof_intermediate_steps_prob,
         seed,
         (model_name, lm_type, model_name_for_batch_size),
         lrate,
         from_scratch,
         augmentation_prob) in hyparas:

        logic_dataset_config_load_type = None
        # logic_dataset_concatenate_all_splits_into_train = False

        if context_len is not None:
            _context_len = context_len
        else:
            if logic_dataset_uname.find('AUG') >= 0:
                _use_original_serial = True
                _context_len = 3000
            else:
                _use_original_serial = use_original_serial if use_original_serial is not None else False
                _context_len = 2048


            region = engine.region
            n_cpus_per_node, n_gpus_per_node, n_total_gpus, gpu_name_for_batch_size = get_qsub_cpu_gpu_setting(engine, _context_len, run_mode)
            is_V100 = engine.resource.find('rt_G') >= 0 or engine.resource.find('rt_F') >= 0

            deepspeed_stage_org = deepspeed_stage
            if model_name.find('rwkv') >= 0 or model_name.find('RWKV') >= 0:
                deepspeed_stage = 'zero2'  # as zero3 somehow hangs
                preprocessing_num_workers = 4
            else:
                # preprocessing_num_workers = 1
                # larger value may lead to hangup
                # preprocessing_num_workers = 32
                preprocessing_num_workers = 1

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
            if fix_precision and should_force_fp32 and float_precision in ['fp16', 'bf16']:
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

            # if augmentation:
            if augmentation_prob > 0:
                no_instruction = True
            else:
                no_instruction = False

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
                    # max_grad_norm=max_grad_norm,
                    # rec_adam_target_task_weight=rec_adam_target_task_weight,
                    rec_adam_fisher_coef=rec_adam_fisher_coef,

                    # update_parameters=update_parameters,

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
                    other_dataset_top_dirs=OTHER_DATASETS_DIRS,
                    other_dataset_names=other_dataset_names,
                    other_dataset_config_names=other_dataset_config_names,
                    other_dataset_config_load_types=other_dataset_config_load_types,
                    other_dataset_take_n_s=other_dataset_take_n_s,
                    other_dataset_probs=other_dataset_probs,
                    do_sft=do_sft,
                    use_test_as_val=setting.get('use_test_as_val', use_test_as_val),
                    use_test_as_train=setting.get('use_test_as_train', use_test_as_train),
                    streaming=streaming,
                    use_original_serial=_use_original_serial,
                    no_instruction=no_instruction,
                    # prompt_emphasize_theorems=prompt_emphasize_theorems,
                    augmentation_prob=augmentation_prob,

                    sample_negative_proof=sample_negative_proof,
                    proof_intermediate_steps_prob=proof_intermediate_steps_prob,
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
                # 'do_eval_in_outerloop': False,
                'do_predict': False,
            })
            setting.update({
                'seed': seed,

                'logic_dataset_uname': logic_dataset_uname,
                'max_train_samples': max_train_samples,
                # 'other_dataset_name': other_dataset_names,    # should avoid list in the setting
                # 'other_dataset_config_name': other_dataset_config_names,

                'logic_dataset_config_load_type': logic_dataset_config_load_type,
                # 'logic_dataset_concatenate_all_splits_into_train': logic_dataset_concatenate_all_splits_into_train,

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

                # '_n_gpu': n_total_gpus,

                'gpu_name_for_batch_size': gpu_name_for_batch_size,
                'use_auth_token': True,
                'log_examples': True,
            })

            if seed == 0:
                setting.update({
                    'train_no_sampling': True,
                    'eval_no_sampling': True,
                    'logic_eval_no_sampling': True,
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
                                   port=random.randint(29777, 31777),
                                   n_gpus_per_node=n_gpus_per_node)

            run_by_engine(
                engine,
                command,
                output_dir,
                hours=_hours,
                force=True,  # assuming that we do not have many jobs
                dry_run=dry_run
            )
            if streaming and take_interval_between_jobs_if_streaming:
                logger.info('sleep for a wihle to avoid "Too many requests" exception for huggingface hub')
                time.sleep(60 * 10)

            engine.n_resource = n_resouce_org
            deepspeed_stage= deepspeed_stage_org

    logger.info('------------- ./01.train.py finished !! -----------')



if __name__ == '__main__':
    main()
