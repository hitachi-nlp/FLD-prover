import logging
from typing import Any, Dict, Optional, Union, List, Tuple, Set
import glob
from pathlib import Path
import json
import math
import re

from pydantic import BaseModel
from lab import build_dir
from script_engine import QsubEngine, SubprocessEngine
from script_engine.base import EngineBase
from tempfile import mktemp
import os

# from pytorch_lightning import Trainer


logger = logging.getLogger(__name__)


def maybe_option_value(option: str, value: Any) -> str:
    if value is None:
        return ''
    else:
        if isinstance(value, str) and value.find(' ') >= 0:
            return f'{option} "{value}"'
        else:
            return f'{option} {value}'


_PROVER_BATCH_SETTINGS = {

    # XXX: if you change max_source_length or max_target_length,
    # make sure that all the stuf fit into memory with padding='max_len' option.
    # The 'max_len' option guarantee that the model always use the max_len inputs without truncation
    # thus, we can measure the maxmum usage of memory.

    'V100_16_1': {

        't5-base': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1700,
            'max_target_length': 100,

            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            # 'generation_num_beams': 10,
            'generation_num_beams': 1,
        },

        't5-base.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            # 'generation_num_beams': 10,
            'generation_num_beams': 1,
        },

        # could not fit
        # 'gpt2-medium.all_at_once': {
        #     # 'padding': 'max_length',
        #     'padding': 'longest',

        #     'max_source_length': 2000,
        #     'max_target_length': 2000,

        #     'per_device_train_batch_size': 1,
        #     'per_device_eval_batch_size': 1,
        #     'gradient_checkpointing': False,

        #     'generation_num_beams': 1,
        # },

        'gpt2-medium.short_cntx.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 800,
            'max_target_length': 800,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'gpt2-large.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-small.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-1b-short-ctx.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 500,
            'max_target_length': 500,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },


    },


    'V100_16_4': {

        'google/mt5-base.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-base-long': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-base-long.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 4,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-small.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-medium.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'gpt2-medium.max_pos_1000.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 4,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

    },




    'V100_16_4.deepspeed': {

        't5-base': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1700,
            'max_target_length': 100,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            # 'generation_num_beams': 10,
            'generation_num_beams': 1,
        },

        't5-base.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            # 'generation_num_beams': 10,
            'generation_num_beams': 1,
        },
        # 'allenai/led-base-16384': {}
        # 'allenai/led-large-16384': {}
        # 'google/long-t5-tglobal-base': {}
        # 'google/long-t5-tglobal-large': {}
        # 'google/mt5-base': {}




        'google/mt5-base': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'google/mt5-base.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'google/mt5-large': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 4,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'google/mt5-large.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-base-long': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 8,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-base-long.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 8,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-large-long': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 4,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-large-long.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-small.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-medium.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-large.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'facebook/xglm-564M.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },






        'retrieva-jp/t5-xl': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-xl.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-1b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        # 'cyberagent/open-calm-1b.all_at_once': {
        #     # 'padding': 'max_length',
        #     'padding': 'longest',

        #     'max_source_length': 2000,
        #     'max_target_length': 2000,

        #     'per_device_train_batch_size': 4,
        #     'per_device_eval_batch_size': 2,
        #     'gradient_checkpointing': True,

        #     'generation_num_beams': 1,
        # },

        'cyberagent/open-calm-3b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-7b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'matsuo-lab/weblab-10b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },


    },






    'A100_48_1': {

        't5-base': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1700,
            'max_target_length': 100,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            # 'generation_num_beams': 10,
            'generation_num_beams': 1,
        },

        't5-base.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            # 'generation_num_beams': 10,
            'generation_num_beams': 1,
        },
        # 'allenai/led-base-16384': {}
        # 'allenai/led-large-16384': {}
        # 'google/long-t5-tglobal-base': {}
        # 'google/long-t5-tglobal-large': {}
        # 'google/mt5-base': {}




        'google/mt5-base': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'google/mt5-base.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'google/mt5-large': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'google/mt5-large.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-base-long': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-base-long.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 8,
            'per_device_eval_batch_size': 8,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-large-long': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-large-long.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-xl': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 100,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'retrieva-jp/t5-xl.all_at_once': {
            # # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 1000,
            'max_target_length': 1000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-small.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-medium.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': False,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-large.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-1b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-3b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'cyberagent/open-calm-7b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },

        'matsuo-lab/weblab-10b.all_at_once': {
            # 'padding': 'max_length',
            'padding': 'longest',

            'max_source_length': 2000,
            'max_target_length': 2000,

            'per_device_train_batch_size': 1,
            'per_device_eval_batch_size': 1,
            'gradient_checkpointing': True,

            'generation_num_beams': 1,
        },





    }


}


_CAUSAL_PROVER_BATCH_SETTINGS = {}


def get_batch_setting(script_type: str, gpu_name: str, model_name) -> Dict[str, Any]:
    if script_type == "run_prover":

        return _PROVER_BATCH_SETTINGS[gpu_name][model_name]

    elif script_type == "run_causal_prover":
        if gpu_name in _CAUSAL_PROVER_BATCH_SETTINGS and model_name in _CAUSAL_PROVER_BATCH_SETTINGS[gpu_name]:
            setting = _CAUSAL_PROVER_BATCH_SETTINGS[gpu_name][model_name]
        else:
            setting = _PROVER_BATCH_SETTINGS[gpu_name][model_name]
            setting["block_size"] = setting["max_target_length"]
            setting["FLD_proof_eval_generation_top_k"] = setting.pop("generation_top_k", None)
            setting["FLD_proof_eval_generation_num_return_sequences"] = setting.pop("generation_num_return_sequences", None)
            setting["FLD_proof_eval_generation_num_beams"] = setting.pop("generation_num_beams", None)
            setting["FLD_proof_eval_padding"] = setting.pop("padding", None)
            return setting
    else:
        raise ValueError()


_DATASET_PATHS = {
    '20221120.negative_tree.debug': {
        'train_file': './outputs.FLNL/10.create_FLNL_corpus/20221120.negative_tree/local_dataset_name=20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-1000/rsd_objct_nns_mx_fctr=1.0/smpl_hrd_ngtvs=True/try_ngtd_hypthss_frst=False/us_fxd_trnsltn=False/test/test.jsonl',
        'validation_file': './outputs.FLNL/10.create_FLNL_corpus/20221120.negative_tree/local_dataset_name=20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-1000/rsd_objct_nns_mx_fctr=1.0/smpl_hrd_ngtvs=True/try_ngtd_hypthss_frst=False/us_fxd_trnsltn=False/test/test.jsonl',
        'test_file': './outputs.FLNL/10.create_FLNL_corpus/20221120.negative_tree/local_dataset_name=20221120.negative_tree__arg-RT__frml-cmpl__tree-small__dist-5__transl_dist--5__transl-wide__size-1000/rsd_objct_nns_mx_fctr=1.0/smpl_hrd_ngtvs=True/try_ngtd_hypthss_frst=False/us_fxd_trnsltn=False/test/test.jsonl',
    },
}


def get_dataset_setting(script_type: str,
                        uname: str,
                        top_dirs: List[str],
                        other_dataset_name: Optional[str] = None,
                        other_dataset_config_name: Optional[str] = None,
                        use_test_as_train=False,
                        use_test_as_val=False,
                        streaming=False,
                        instruction=False) -> Dict[str, Any]:
    type_, dataset_name, dataset_config_name = _parse_dataset_name(uname)

    setting: Dict[str, Any] = {
        'predict_with_generate': True,
        'remove_unused_columns': False,
        'instruction': instruction,
        'streaming': streaming,
    }

    if script_type == "run_prover":
        FLD_option_prefix = ""

    elif script_type == "run_causal_prover":
        setting["dataset_name"] = other_dataset_name
        setting["dataset_config_name"] = other_dataset_config_name
        FLD_option_prefix = "FLD_"

    else:
        raise ValueError()

    if type_ == 'local':
        dataset_paths = get_local_dataset_paths(dataset_name,
                                                top_dirs=top_dirs,
                                                use_test_as_train=use_test_as_train,
                                                use_test_as_val=use_test_as_val)
        setting.update({f"{FLD_option_prefix}{key}": val for key, val in dataset_paths.items()})
        setting[f'{FLD_option_prefix}file_type'] = 'json'

    elif type_ == 'hf':
        if use_test_as_train:
            logger.warning('use_test_as_train=True does not work with datasets hosted on huggingface.')
        if use_test_as_val:
            logger.warning('use_test_as_val=True does not work with datasets hosted on huggingface.')
        setting[f'{FLD_option_prefix}dataset_name'] = dataset_name
        setting[f'{FLD_option_prefix}dataset_config_name'] = dataset_config_name

    else:
        raise ValueError()

    return setting


def get_local_dataset_paths(uname: str,
                            top_dirs: List[str],
                            use_test_as_train=False,
                            use_test_as_val=False,
                            allow_not_found_splits=False) -> Dict[str, Optional[str]]:
    type_, dataset_name, _ = _parse_dataset_name(uname)
    if type_ != 'local':
        raise ValueError()

    def validate(dataset_paths):

        for split_name, path in dataset_paths.items():
            if path is None or not Path(path).exists:
                msg = f'dataset="{dataset_name}", split="{split_name}" not found under {str(top_dirs)}'
                if allow_not_found_splits:
                    logger.warning(msg)
                else:
                    raise Exception(msg)

    if dataset_name in _DATASET_PATHS:
        paths = _DATASET_PATHS[dataset_name].copy()

        if use_test_as_train:
            paths['train_file'] = paths['test_file']

        if use_test_as_val:
            paths['validation_file'] = paths['test_file']

        validate(paths)
        return paths

    else:

        def get_split_jsonl(top_dir: Path, split: str) -> Optional[Path]:
            all_paths = glob.glob(str(top_dir) + '/**/*', recursive=True)
            paths = [path for path in all_paths
                     if path.endswith(f'{split}.jsonl') and str(path).find('job-') < 0]
            if len(paths) == 0:
                return None
            elif len(paths) == 1:
                return paths[0]
            else:
                raise ValueError(f'multiple dataset file found under {str(top_dir)} as:\n'
                                 '\n'.join([str(path) for path in paths]))

        def get_split_jsonl_with_warning(top_dir: Path, local_dataset_name: str, split: str) -> Optional[Path]:
            path = get_split_jsonl(top_dir, split)
            # if path is None:
            #     logger.warning('dataset split="%s" name="%s" not found under "%s"',
            #                    split,
            #                    local_dataset_name,
            #                    str(top_dir))
            return path

        # We must use glob to follow symbolic links
        found_dataset_paths = None
        found_lab_path = None
        for top_dir in top_dirs:

            for lab_path in glob.glob(top_dir + '/**/*', recursive=True):

                if not lab_path.endswith('lab.params.json'):
                    continue
                if lab_path.find('job-') >= 0:
                    continue

                setting = json.load(open(lab_path))

                if setting.get('dataset_name', None) != dataset_name:
                    continue

                if found_lab_path is not None:
                    raise Exception(f'Multiple files for dataset "{dataset_name}" are found:\n1. "{str(found_lab_path)}"\n2. "{str(lab_path)}"')

                lab_path = Path(lab_path)
                found_lab_path = lab_path

                train_path = get_split_jsonl_with_warning(lab_path.parent, dataset_name, 'train')
                valid_path = get_split_jsonl_with_warning(lab_path.parent, dataset_name, 'valid')
                test_path = get_split_jsonl_with_warning(lab_path.parent, dataset_name, 'test')

                if use_test_as_train:
                    train_path = test_path
                if use_test_as_val:
                    valid_path = test_path

                found_dataset_paths = {
                    'train_file': str(train_path) if train_path is not None else None,
                    'validation_file': str(valid_path) if valid_path is not None else None,
                    'test_file': str(test_path) if test_path is not None else None,
                }

            if found_dataset_paths is not None:
                validate(found_dataset_paths)
                return found_dataset_paths

    raise ValueError(f'Dataset {dataset_name} not found under {top_dirs}.')


def _parse_dataset_name(name: str) -> Tuple[str, str, Optional[str]]:
    if name.startswith('hf.'):
        return 'hf', *re.sub(r'^hf\.', '', name).split('__')
    else:
        return 'local', name, None


def get_config(name: str) -> Dict[str, Any]:
    if name == 'default':
        return {
            'seed': 42,
            # 'n_gpu': 1,
            'max_grad_norm': 0.5,
            'num_train_epochs': None,
            'max_steps': 10000,
            'gradient_accumulation_steps': 11,
            # 'check_val_every_n_epoch': 1,
            'max_eval_samples': 2000,
            'proof_sampling': 'stepwise',
            'max_proof_steps': 30,
            'learning_rate': 1e-4,
            'warmup_steps': 1000,
            'model_name_or_path': 't5-large',
            # 'fp16': True,

            # 'source_prefix': 'Solve FLD task: ',
            'source_prefix': None,
            'generation_num_beams': 10,
            'generation_top_k': 10,
            'generation_max_proof_steps': 20,
            # 'verifier_ckpt': "",
            # 'verifier_weight': 0,
            # 'proof_search': False,
            # 'oracle_prover': False,
            # 'oracle_verifier': False,
            'max_source_length': 512,
            # 'padding': longest,
            'max_target_length': 64,

            'logging_strategy': 'steps',
            'logging_steps': 25,
            'overwrite_output_dir': True,

            'log_generation': True,

            # 'dataset': FLNL,
            # 'path_train': specify_me,
            # 'path_val': specify_me,
            # 'path_test': specify_me,

            # 'exclude_unknown': False,
            # 'add_final_reference_to_proofs': False,
            # 'sample_goal': intermediates # hypothesis | intermediates,
            # 'subtree_proved_prob': 0.75,
            # 'subtree_proved_all_or_none': False,
            'sample_negative_proof': False,
            'no_subproof_for_unknown': False,
            'per_device_train_batch_size': 3,
            'per_device_eval_batch_size': 3,
            'dataloader_num_workers': 2,
            # 'padding': longest,
            # 'shuffle_train': True,

            'log_examples': False,
            # 'no_lower': True,

            # 'evaluation_strategy': 'steps',
            # 'save_strategy': 'steps',
            'logging_strategy': 'steps',

        }
    else:
        raise NotImplementedError()


_PROVER_LEARNING_SETTINGS: Dict[str, Dict[str, Any]] = {

    'FS.shot-0': {
        'max_train_samples': 0,
        'max_eval_samples': 101,
        'max_predict_samples': 2000,

        'warmup_steps': 0,

        'train_effective_batch_size': 64,
        'max_steps': 1,
        'eval_steps': 1,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FS.shot-10': {
        'max_train_samples': 10,
        'max_eval_samples': 101,
        'max_predict_samples': 2000,

        'warmup_steps': 500,

        'train_effective_batch_size': 64,
        'max_steps': 2000,
        'eval_steps': 100,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FS.shot-100': {
        'max_train_samples': 100,
        'max_eval_samples': 101,
        'max_predict_samples': 2000,

        'warmup_steps': 500,

        'train_effective_batch_size': 64,
        'max_steps': 2000,
        'eval_steps': 100,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    # -- pre-training: FLNL (arg-RT/arg-AA) / RuleTaker / EB
    'FT.step-5000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 101,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 5000,
        'eval_steps': 1000,
        'warmup_steps': 1000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-5000.LLM': {
        'max_train_samples': None,
        'max_eval_samples': 100,
        'max_predict_samples': 100,

        'train_effective_batch_size': 64,
        'max_steps': 5000,
        'eval_steps': 1600,
        'warmup_steps': 100,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-10000.LLM': {
        'max_train_samples': None,
        'max_eval_samples': 100,
        'max_predict_samples': 100,

        'train_effective_batch_size': 64,
        'max_steps': 10000,
        'eval_steps': 2000,
        'warmup_steps': 100,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    # -- pre-training (RT_large_steps): FLNL (arg-RT/arg-AA) / RuleTaker / EB
    # -- pre-training: FLNL (arg-FLNL)
    'FT.step-10000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 101,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 10000,
        'eval_steps': 1000,
        'warmup_steps': 1000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-10000.mx_evl-100': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 100,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 10000,
        'eval_steps': 1000,
        'warmup_steps': 1000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-10000.mx_evl-100.btch_sz-8': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 100,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 8,
        'max_steps': 10000,
        'eval_steps': 1000,
        'warmup_steps': 1000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-100000.mx_evl-100.btch_sz-8': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 100,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 8,
        'max_steps': 100000,
        'eval_steps': 10000,
        'warmup_steps': 1000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },


    'FT.step-20000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 101,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 20000,
        'eval_steps': 5000,
        'warmup_steps': 1000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-20000.max_eval_300': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 300,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 20000,
        'eval_steps': 5000,
        'warmup_steps': 1000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-50000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 101,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 50000,
        'eval_steps': 5000,
        'warmup_steps': 3000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-100000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 101,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 100000,
        'eval_steps': 10000,
        'warmup_steps': 3000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'FT.step-150000': {
        'max_train_samples': None,
        # 'max_eval_samples': 2000,
        # 'max_predict_samples': 2000,
        'max_eval_samples': 101,
        'max_predict_samples': 1000,

        'train_effective_batch_size': 64,
        'max_steps': 150000,
        'eval_steps': 10000,
        'warmup_steps': 3000,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },

    'debug.ZS': {
        'max_train_samples': 1,
        'max_eval_samples': 100,
        'max_predict_samples': 1,

        'train_effective_batch_size': 64,
        'max_steps': 1,
        'eval_steps': 1,
        'warmup_steps': 999999,

        'use_test_as_train': True,
        'use_test_as_val': True,
    },

    'debug.step-10': {
        'max_train_samples': 1,
        'max_eval_samples': 1,
        'max_predict_samples': 1,

        'train_effective_batch_size': 64,
        'max_steps': 10,
        'eval_steps': 1,
        'warmup_steps': 0,

        'use_test_as_train': True,
        'use_test_as_val': True,
    },

    'debug.micro': {
        'max_train_samples': 1,
        'max_eval_samples': 1,
        'max_predict_samples': 1,

        'train_effective_batch_size': 64,
        'max_steps': 100,
        'eval_steps': 100,
        'warmup_steps': 0,

        'use_test_as_train': True,
        'use_test_as_val': True,
    },

    'debug.micro.deepspeed': {
        'max_train_samples': 1,
        'max_eval_samples': 1,
        'max_predict_samples': 1,

        'train_effective_batch_size': 64,
        'max_steps': 3000,    # need more steps for convergence
        'eval_steps': 3000,
        'warmup_steps': 0,

        'use_test_as_train': True,
        'use_test_as_val': True,
    },

    'debug.tiny': {
        'max_train_samples': 10,
        'max_eval_samples': 10,
        'max_predict_samples': 10,

        'train_effective_batch_size': 64,
        'max_steps': 300,
        'eval_steps': 300,
        'warmup_steps': 0,

        'use_test_as_train': True,
        'use_test_as_val': True,
    },

    'debug.middle': {
        'max_train_samples': 100,
        'max_eval_samples': 100,
        'max_predict_samples': 100,

        'train_effective_batch_size': 64,
        'max_steps': 300,
        'eval_steps': 300,
        'warmup_steps': 0,

        'use_test_as_train': True,
        'use_test_as_val': True,
    },

    'debug.find_batch_size': {
        'max_train_samples': 16,
        'max_eval_samples': 16,
        'max_predict_samples': 16,

        'train_effective_batch_size': 64,
        'max_steps': 10,
        'eval_steps': 10,
        'warmup_steps': 0,

        'use_test_as_train': True,
        'use_test_as_val': True,
    },

    'debug.20000.zero_warmup': {
        'max_train_samples': None,
        'max_eval_samples': 100,
        'max_predict_samples': 100,

        'train_effective_batch_size': 64,
        'max_steps': 10000,
        'eval_steps': 2000,
        'warmup_steps': 0,

        'use_test_as_train': False,
        'use_test_as_val': True,
    },


    'push_to_hub': {
        'max_train_samples': 1,
        'max_eval_samples': 1,
        'max_predict_samples': 0,

        'train_effective_batch_size': 32,
        'warmup_steps': 0,
        'max_steps': 1,
        'eval_steps': 1,

        'learning_rate': 1e-5,

        'use_test_as_train': False,
        'use_test_as_val': False,
    },

}


_CAUSAL_PROVER_LEARNING_SETTINGS: Dict[str, Dict[str, Any]] = {
}


def get_learning_setting(script_type: str,
                         name: str,

                         epoch: Optional[int] = None,
                         steps: Optional[int] = None,

                         steps_upper: Optional[int] = None,

                         warmup_steps: Optional[int] = None,
                         warmup_ratio: Optional[float] = None,

                         train_effective_batch_size: Optional[int] = None,
                         num_evals: Optional[int] = None,
                         max_eval_samples: Optional[int] = None,
                         ) -> Dict[str, Any]:
    if name.startswith('LLM_FS.shot-'):
        # if script_type == "run_causal_prover":
        #     raise NotImplementedError()

        epoch = epoch or 50
        steps_upper = steps_upper or 300
        warmup_ratio = warmup_ratio or 0.3
        train_effective_batch_size = train_effective_batch_size or 32

        if epoch is not None and steps is not None:
            raise ValueError()
        if epoch is None and steps is None:
            raise ValueError()

        max_train_samples = int(name[len('LLM_FS.shot-'):])

        max_steps = steps\
            or int(epoch * max(1, math.floor(max_train_samples / train_effective_batch_size)))

        # see[here](https://github.com/huggingface/transformers/issues/22751)
        hf_bug_zero_lr_offset = 20
        max_steps = max_steps + hf_bug_zero_lr_offset
        if steps_upper is not None:
            max_steps = min(max_steps, steps_upper)

        warmup_steps = warmup_steps or int(warmup_ratio * max_steps)

        num_evals = num_evals or 3
        for _num_evals in range(num_evals, 0, -1):
            eval_steps = int(max_steps / _num_evals)
            if eval_steps >= 50:
                # do evaluation only after 50 steps
                # because early evaluation is extremely slow due to the repetitions
                break

        max_eval_samples = max_eval_samples or 101

        base_setting = {
            'max_train_samples': max_train_samples,
            'logging_steps': 1,
            'use_test_as_train': False,
            'use_test_as_val': True,
        }

        if script_type == "run_causal_prover":
            base_setting.update({
                'FLD_dataset_prob': 1.0,
                'FLD_max_eval_samples': max_eval_samples,
            })

    else:
        if epoch is not None:
            raise ValueError()

        if script_type == "run_prover":
            base_setting = _PROVER_LEARNING_SETTINGS[name].copy()

        elif script_type == "run_causal_prover":
            if name in _CAUSAL_PROVER_LEARNING_SETTINGS:
                base_setting = _CAUSAL_PROVER_LEARNING_SETTINGS[name].copy()
            else:
                base_setting = _PROVER_LEARNING_SETTINGS[name].copy()
                base_setting.update({
                    'FLD_dataset_prob': 1.0,
                    'FLD_max_eval_samples': base_setting["max_eval_samples"],
                })
        else:
            raise ValueError()

        max_steps = steps or base_setting['max_steps']
        if steps_upper is not None:
            max_steps = min(max_steps, steps_upper)

        warmup_steps = warmup_steps\
            or int(warmup_ratio * max_steps) if warmup_ratio is not None else base_setting['warmup_steps']

        train_effective_batch_size = train_effective_batch_size or base_setting['train_effective_batch_size']

        eval_steps = int(max_steps / num_evals) if num_evals is not None else base_setting['eval_steps']

        max_eval_samples = max_eval_samples or base_setting['max_eval_samples']

    base_setting['max_steps'] = max_steps
    base_setting['warmup_steps'] = warmup_steps
    base_setting['train_effective_batch_size'] = train_effective_batch_size
    base_setting['eval_steps'] = eval_steps
    base_setting['max_eval_samples'] = max_eval_samples

    return base_setting


class CheckpointSpec(BaseModel):

    name_or_local_dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    lrate: Optional[float] = None
    # add_final_reference_to_proofs: Optional[bool] = None


def get_checkpoints(spec: CheckpointSpec,
                    check_point_dirs: Optional[List[Union[str, Path]]] = None) -> List[Tuple[str, CheckpointSpec]]:
    if spec.name_or_local_dataset_name.startswith('hf.'):
        return [(spec.name_or_local_dataset_name, spec)]
    else:
        raise NotImplementedError()
    if spec.name_or_local_dataset_name is None:
        return [(None, CheckpointSpec())]

    name_or_local_dataset_name = spec.name_or_local_dataset_name

    if check_point_dirs is None:
        raise ValueError()

    checkpoints: List[Tuple[str, CheckpointSpec]] = []
    for check_point_dir in check_point_dirs:
        for checkpoint in Path(check_point_dir).glob('**/*ckpt'):
            lab_setting = json.load(open(checkpoint.parent.parent.parent.parent / 'lab.params.json'))

            def check_spec(name: str) -> bool:
                lab_value = lab_setting.get(name, '<<default>>')
                if getattr(spec, name) is not None and lab_value != getattr(spec, name):
                    return False
                else:
                    return True

            not_met = False
            for name in spec.dict().keys():

                if name == 'name_or_local_dataset_name':
                    continue

                if not check_spec(name):
                    not_met = True
                    break

            if not_met:
                continue

            FLD_dataset_uname = name_or_local_dataset_name

            found_local_dataset_name = lab_setting['FLD_dataset_uname']

            if found_local_dataset_name == FLD_dataset_uname:
                found_spec = CheckpointSpec(**lab_setting)
                found_spec.name_or_local_dataset_name = name_or_local_dataset_name
                checkpoints.append((str(checkpoint), found_spec))

        return checkpoints


def get_model_settings(model_name_or_path: str) -> Dict[str, Any]:
    if model_name_or_path == 'izumi-lab/stormy-7b-10ep':
        return {'model_name_or_path': model_name_or_path,
                'config_name': 'cyberagent/open-calm-7b'}
    else:
        return {'model_name_or_path': model_name_or_path}


def get_tokenizer_settings(model_name_or_path: str) -> Dict[str, Any]:
    if model_name_or_path.startswith('line-corporation'):
        return {'use_fast_tokenizer': False}
    elif model_name_or_path == 'izumi-lab/stormy-7b-10ep':
        return {'tokenizer_name': 'cyberagent/open-calm-7b'}
    else:
        return {}


def get_save_eval_step_setting(eval_steps: Optional[int] = None,
                               do_save_model=False,
                               max_steps: Optional[int] = None) -> Dict[str, Any]:
    setting = {}

    if eval_steps is not None:
        setting['eval_steps'] = eval_steps
        setting['evaluation_strategy'] = 'steps'

    if max_steps is not None and setting['eval_steps'] > max_steps:
        setting['eval_steps'] = max_steps

    if do_save_model:
        setting['save_steps'] = setting.get('eval_steps', None)
        setting['save_strategy'] = 'steps'
        setting['save_total_limit'] = 1  # XXX "0" saves all the checkpoints

    return setting


def make_val_interval_setting(all_setting: Dict[str, Any], train_file: str) -> Dict[str, Any]:

    if all_setting.get('num_train_epochs', None) not in [None, -1]:
        num_val_stage_throught_training = all_setting['num_val_stage_throught_training']
        return {
            'val_check_interval_in_batch': None,
            # 'check_val_every_n_epoch': max(1, int(all_setting['num_train_epochs'] / num_val_stage_throught_training)),
        }
    else:
        if all_setting.get('max_steps', None) is None\
                or all_setting.get('gradient_accumulation_steps', None) is None:
            raise Exception('Could not specify validation interval')
        else:
            max_steps = all_setting['max_steps']
            num_val_stage_throught_training = all_setting['num_val_stage_throught_training']
            num_batches_per_grad_step = all_setting['gradient_accumulation_steps']

            num_batches_per_val = int(max_steps * num_batches_per_grad_step / num_val_stage_throught_training)

            if all_setting['limit_train_batches'] is not None:
                num_batches_per_epoch = all_setting['limit_train_batches']
            else:
                num_train_examples = len(open(train_file).readlines())
                num_batches_per_epoch = max(1, int(num_train_examples / all_setting['per_device_train_batch_size']))

            if num_batches_per_val >= num_batches_per_epoch:
                grad_steps_per_epoch = max(math.ceil(num_batches_per_epoch / num_batches_per_grad_step), 1)
                train_epochs = max_steps / grad_steps_per_epoch

                val_check_interval_in_batch = None
                check_val_every_n_epoch = max(int(train_epochs / num_val_stage_throught_training), 1)
            else:
                val_check_interval_in_batch = max(int(max_steps * num_batches_per_grad_step / num_val_stage_throught_training), 1)
                # check_val_every_n_epoch = None

            return {
                'val_check_interval_in_batch': val_check_interval_in_batch,
                # 'check_val_every_n_epoch': check_val_every_n_epoch,
            }


def make_command(script_type: str,
                 output_dir: Union[str, Path],
                 setting: Dict,
                 run_mode: str,
                 n_gpus: Optional[int] = None) -> str:

    if script_type == 'run_prover':
        script_path = './run_prover.py'

        ignore_option_names = [
            'base_config_name',
            'checkpoint_name',
            'checkpoint_path',
            'learning',
            'train_effective_batch_size',
            'max_proof_steps',
            'FLD_dataset_uname',
            'n_proc_per_node',
            'streaming',

            'use_test_as_train',
            'use_test_as_val',

            'script_type',
            'other_dataset_name',
            'other_dataset_config_name',
        ]

    elif script_type == 'run_causal_prover':
        script_path = './run_causal_prover.py'
        ignore_option_names = [
            'base_config_name',
            'checkpoint_name',
            'checkpoint_path',
            'learning',
            'train_effective_batch_size',
            'max_proof_steps',
            'FLD_dataset_uname',
            'n_proc_per_node',

            'use_test_as_train',
            'use_test_as_val',

            'proof_sampling',
            'generation_num_beams',
            'generation_top_k',
            'generation_max_proof_steps',
            'max_source_length',

            'log_generation',
            'sample_negative_proof',
            'no_subproof_for_unknown',
            'predict_with_generate',
            'other_dataset_config_name',
            'FLD_test_file',
            'FLD_file_type',
            'do_eval_in_outerloop',
            'script_type',
            'other_dataset_name',
            'lm_type',
            'FLD_proof_eval_generation_timeout',
            'max_predict_samples',
        ]

    else:
        raise ValueError()

    commands: List[str] = []

    commands.append('source ./set-envs.sh &&')
    

    torchrun_err_file = str(output_dir / 'torchrun_err.txt')

    if run_mode == 'vanilla':
        commands.append(f'python {script_path}')

    elif run_mode == 'profile':
        commands.append(f'kernprof -lv {script_path}')

    elif run_mode == 'torchrun':
        if n_gpus is None:
            raise ValueError()
        commands.append(f'TORCHELASTIC_ERROR_FILE={torchrun_err_file}'
                        f' torchrun --nproc_per_node {n_gpus} {script_path}')

    elif run_mode == 'deepspeed':
        # For deepspeed settings, see [here](https://github.com/ohtaman/abci-examples/tree/main/202307#deepspeed-を用いた-multi-node-multi-gpu-訓練分散訓練の例)

        ds_config = 'ds_config/ds_config_zero3.json'

        # commands.append(f'TORCHELASTIC_ERROR_FILE={torchrun_err_file}'
        #                 f' torchrun --nproc_per_node {n_gpus} {script_path}'
        #                 f' --deepspeed {ds_config}')

        # """
        # MASTER_ADDR=$HOSNAME deepspeed \
        #   --master_addr $HOSTNAME \
        #   --hostfile $hostfile \
        #   --no_ssh_check \
        #   --launcher OpenMPI \
        #   --launcher_args "-mca coll ^hcoll" \
        #   src/finetune_lora_distribute.py \
        #   --model_name $MODEL \
        #   --config_file $CONFIG
        # """
        # hostfile = mktemp()
        # with open(hostfile, 'w') as f_out:
        #     for line in os.environ['SGE_JOB_HOSTLIST'].split('\n'):
        #         f_out.write(line + f' slots={n_gpus}')

        commands.append(
            ' && '.join([
                # 'module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12',
                # 'source /etc/profile.d/modules.sh && module load hpcx/2.12',

                'source /etc/profile.d/modules.sh',
                'module load python/3.11 cuda/11.7 cudnn/8.6 nccl/2.12 hpcx/2.12',

                # load open-mpi
                # 'source $PROJECTS/spack/share/spack/setup-env.sh 1>err.txt 2>&1',
                # 'echo "hoge" > hoge.txt',
                # 'spack load openmpi@4.1.4',

                f'export hostfile=$(mktemp) && for l in `cat $SGE_JOB_HOSTLIST`; do echo $l slots={n_gpus}; done > $hostfile',
            ])
        )

        commands.append(
            ' '.join([
                '&& MASTER_ADDR=$HOSTNAME',
                'deepspeed',
                '--master_addr $HOSTNAME',
                '--hostfile ${hostfile}',
                '--no_ssh_check',
                '--launcher OpenMPI',
                '--launcher_args "-mca coll ^hcoll"',
                f'{script_path}',
                f'--deepspeed {ds_config}'
            ])
        )

    else:
        ValueError()

    commands.extend([
        f'--output_dir {str(output_dir)}',
        f'--logging_dir {str(Path(output_dir) / "tensorboard_log")}',
    ])

    for name, value in setting.items():
        if name in ignore_option_names:
            continue

        option_str = maybe_option_value(f'--{name}', value)
        # if run_mode == 'deepspeed':
        #     option_str = option_str.replace('"', '\\"')
        #     option_str = option_str.replace('\'', '\\\'')
        commands.append(option_str)

    return ' '.join(commands)


def make_output_dir(setting: Dict,
                    top_dir: Union[str, Path],
                    dirname_ignore_params: Optional[List[str]] = None) -> str:
    dataset_setting_names = [key for key in setting if
                             key.startswith('dataset_setting.')]
    return build_dir(
        setting,
        top_dir=str(
            Path(top_dir)
            # / f'sstm_nm={setting.get("system_name", str(None))}'
            / f'dtst_nm={setting.get("FLD_dataset_uname", None)}'
            # / f'excld_unknwn={setting.get("exclude_unknown", None)}'
            # / f'add_fnl_rfrc_t_prfs={setting.get("add_final_reference_to_proofs", None)}'
            # / f'EB_tsk={setting.get("EB_task", None)}'
            # / f'splt={setting.get("split", str(None))}'
            / f'bs_cnfg_nm={setting.get("base_config_name", None)}'
            # / f'mdl_nm={setting["model_name"].replace("/", "@") if setting["model_name"] is not None else "None"}'
            / f'chckpnt_nm={setting.get("checkpoint_name", None)}'
        ),
        short=True,

        dirname_ignore_params=[
            # 'system_name',
            # 'EB_task',

            'tokenizer_name',
            'config_name',

            'dataset_config_name',
            'dataset_1',
            'train_file_1',

            'exclude_unknown',
            # 'add_final_reference_to_proofs',
            # 'split',

            'prover_ckpt',
            'verifier_ckpt',

            'base_config_name',
            # 'base_config_path',

            'resume_from_checkpoint',
            'checkpoint_name',
            'checkpoint_path',

            'trainer_ckpt_for_resume_training',

            # 'model_name',

            'train_file',
            'validation_file',
            'test_file',

            'train_file_1',
            'validation_file_1',
            'test_file_1',

            'do_train',
            'do_eval',
            'do_eval_in_outerloop',
            'do_predict',

            'estimated_batches_per_epoch',
            'val_check_interval_in_batch',
            # 'check_val_every_n_epoch',
            'per_device_train_batch_size',
            'gradient_accumulation_steps',
            'per_device_eval_batch_size',
            'padding',
            'limit_eval_batches',
            'num_val_stage_throught_training',

            "argument_configs",
            "translation_configs",

            'max_source_length',
            'max_target_length',
            'unknown_ratio',
            'log_examples',
            'dataloader_num_workers',

            'max_grad_norm',
            'max_eval_samples',
            'max_proof_steps',
            'fp16',
            'generation_max_proof_steps',
            'generation_num_return_sequences',
            'generation_timeout',
            'source_prefix',
            'logging_strategy',

            'overwrite_output_dir',
            'log_generation',
            'path_train_1',
            'path_val_1',
            'path_test_1',
            'evaluation_strategy',
            'save_strategy',
            'logging_strategy',
            'logging_steps',
            'predict_with_generate',
            'checkpoint_lrate',
            'checkpoint_model_name_or_path',
            'checkpoint_model_name',
            'eval_steps',
            'file_type',
            'log_examples',
            'max_predict_samples',
            'max_train_samples',
            'max_eval_samples',
            'remove_unused_columns',
            'save_steps',

            'FLD_dataset_uname',
            'FLD_file_type',
            'FLD_max_eval_samples',
            'FLD_proof_eval_dataset',
            'FLD_proof_eval_generation_top_k',
            'FLD_proof_eval_padding',
            'FLD_proof_eval_generation_timeout',
            'FLD_proof_eval_generation_num_return_sequences',
            'FLD_train_file',
            'FLD_validation_file',
            'FLD_test_file',

            'gradient_checkpointing',
            'lm_type',
            'script_type',
            'use_auth_token',

        ] + dataset_setting_names + (dirname_ignore_params or []),
        save_params=True
    )


def make_system_name(lab_setting: Dict) -> str:
    FLD_dataset_uname = lab_setting['FLD_dataset_uname']
    checkpoint_name = lab_setting['checkpoint_name']

    return f'checkpoint_name--{checkpoint_name}__local_dataset_name--{FLD_dataset_uname}'


def run_by_engine(engine: EngineBase,
                  command: str,
                  output_dir: Union[Path, str],
                  hours=24,
                  dry_run=False):
    output_dir = Path(output_dir)
    log_path = output_dir / 'log.txt'
    if isinstance(engine, SubprocessEngine):
        stdout = None
        stderr = None
        wait_until_finish = True
    else:
        command += f' 1>{str(log_path)} 2>&1'
        stdout = output_dir / 'stdout.txt'
        stderr = output_dir / 'stderr.txt'
        wait_until_finish = False

    engine.run(
        command,
        stdout=stdout,
        stderr=stderr,
        options={'l_opts': [f'h_rt={hours}:00:00']},
        dry_run=dry_run,
        wait_until_finish=wait_until_finish,
    )


_REJECTED_MODELS = [
    # ---------------------------- rejected models ----------------------------

    # ---- reason = max length < 2k ----

    # ('stabilityai/japanese-stablelm-instruct-alpha-7b', 'causal', 'matsuo-lab/weblab-10b'),

    # something wrong with tokenizer
    # ('okazaki-lab/japanese-gpt2-medium-unidic', 'causal', 'gpt2-medium.max_pos_1000'),

    # ('rinna/japanese-gpt2-xsmall', 'causal', 'cyberagent/open-calm-small'),
    # ('rinna/japanese-gpt2-small', 'causal', 'cyberagent/open-calm-small'),
    # ('rinna/japanese-gpt2-medium', 'causal', 'cyberagent/open-calm-medium'),

    # something wrong with tokenizer
    # ('ku-nlp/gpt2-small-japanese-char', 'causal', 'cyberagent/open-calm-small'),
    # ('ku-nlp/gpt2-medium-japanese-char', 'causal', 'cyberagent/open-calm-medium'),

    # ('abeja/gpt2-large-japanese', 'causal', 'cyberagent/open-calm-large'),

    # ('rinna/japanese-gpt-1b', 'causal', 'cyberagent/open-calm-1b'),  # XXX only support max_len=1000

    # ---- reason = others ----

    # somehow can not fit into memory.
    # ('abeja/gpt-neox-japanese-2.7b', 'causal', 'cyberagent/open-calm-7b'),

    # no config at hub
    # ('izumi-lab/stormy-7b-10ep', 'causal', 'cyberagent/open-calm-7b'),

    # tokenizer have too many unknowns for alphabet, e.g., "U" and "l"
    # [rejected] ('sonoisa/t5-base-japanese', 'seq2seq', 't5-base'),
    # [rejected] ('sonoisa/t5-base-japanese-v1.1', 'seq2seq', 't5-base'),
]
