#!/usr/bin/env python
import logging
from pathlib import Path

import click
from script_engine import QsubEngine, SubprocessEngine
from logger_setup import setup as setup_logger

from experimental_setting import (
    get_local_dataset_paths,
)

logger = logging.getLogger(__name__)


@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)
    # output_top_dir = Path('./outputs/00.convert_json_schema.py/2023-05-15')

    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230626.many_bugs_fixed')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230628.make_harder')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230701.finalize')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230703.test_for_release')

    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230707.finalize')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230710.update_translation')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230710.update_translation.bf51eb2')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230710.update_translation.7485fef')

    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230711.refactor_distractors')

    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230711.finalize')

    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230718.case_study')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/2023-07-27.compare_models')

    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230729.case_study_finalize')
    # output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230801.case_study_finalize.fix')
    output_top_dir = Path('./outputs/00.fix_FLD_schema.py/20230801.case_study_finalize.fix.test_large')

    dataset_unames = [
        # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000__dpth-RT.G_MP',   # sFLD-impl
        # '20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',            # FLD-impl.0
        # '20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000',                 # FLD.2

        # ---------------------------------- 20230626.many_bugs_fixed ------------------------------------
        # '20230626.many_bugs_fixed.20221203.first_exp__arg-RT__frml-cmpl__dist-20__transl-nrrw__tree-3__dataset_size-30000.G_MP',
        # '20230626.many_bugs_fixed.20221203.first_exp__arg-FLNL__frml-cmpl__dist-20__transl-wide__tree-3__dataset_size-30000.plus_quantifiers',

        # ---------------------------------- 20230628.make_harder ------------------------------------
        # '20230626.many_bugs_fixed.D3.hard',
        # '20230626.many_bugs_fixed.D3.hard.dist-trees',
        # '20230626.many_bugs_fixed.D3.hard.unk-0.1',
        # '20230626.many_bugs_fixed.D3.hard.brnch-high',
        # '20230626.many_bugs_fixed.D3.hard.dist-neg-1.0',
        # '20230626.many_bugs_fixed.D3.hard.dist-neg-0.5',
        # '20230626.many_bugs_fixed.D3.hard.dist-neg-0.0',
        # '20230626.many_bugs_fixed.D3.hard.dist-trees-only',

        # '20230626.many_bugs_fixed.D8.hard',
        # '20230626.many_bugs_fixed.D8.hard.dist-trees',

        # ---------------------------------- 20230701.finalize ------------------------------------
        # '20230701.D3.default',
        # '20230701.D3.wo_transl_dist',
        # '20230701.D3.brnch-small',
        # '20230701.D3.dist-small',

        # '20230701.D8.default',

        # ---------------------------------- 20230707.finalize ------------------------------------
        # '20230707.finalize.D3.dist-double',
        # '20230707.finalize.D3.dist-triple',
        # '20230707.finalize.D3.dist-quadruple',

        # '20230707.finalize.D8.dist-double',
        # '20230707.finalize.D8.dist-triple',
        # '20230707.finalize.D8.dist-quadruple',

        # ---------------------------------- 20230711.finalize ------------------------------------
        # '20230711.dist-fallback',
        # '20230711.finalize.D3',
        # '20230711.finalize.D8',

        # ---------------------------------- 20230718.case_study ------------------------------------
        # '20230718.case_study.D3.dist-mixture',
        # '20230718.case_study.D3.num_dist-wide',
        # '20230718.case_study.D8.dist-mixture.num_dist-wide',
        
        # '20230718.case_study.D3.dist-mixture.num_dist-wide',
        # '20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_logE',
        # '20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10',
        # '20230718.case_study.D3.dist-mixture.num_dist-wide.transl_vol_log10.adj_verb_noun_equal',

        # ---------------------------------- 20230729.case_study_finalize ------------------------------------
        '20230729.case_study_finalize.D3',
        '20230729.case_study_finalize.D8',
    ]

    DATASETS_DIRS = [
        # './outputs.FLD/10.create_FLD_corpus/20221203.first_exp',
        # './outputs.FLD/10.create_FLD_corpus/20221217.back_to_the_past',
        # './NLProofS/outputs/00.create_cc100_corpus.py/',

        # './outputs.FLD/10.create_FLD_corpus/20230626.many_bugs_fixed',
        # './outputs.FLD/10.create_FLD_corpus/20230628.make_harder',
        # './outputs.FLD/10.create_FLD_corpus/20230701.finalize',

        # './outputs.FLD/10.create_FLD_corpus/20230707.finalize',

        # './outputs.FLD/00.create_corpus/20230710.update_translation',
        # './outputs.FLD/00.create_corpus/20230710.update_translation.bf51eb2',
        # './outputs.FLD/00.create_corpus/20230710.update_translation.7485fef',

        # './outputs.FLD/00.create_corpus/20230711.refactor_distractors',
        # './outputs.FLD/00.create_corpus/20230711.finalize',

        # './outputs.FLD/00.create_corpus/20230718.case_study',
        # './outputs.FLD/00.create_corpus/2023-07-27.compare_models',

        # './outputs.FLD/00.create_corpus/20230729.case_study_finalize',
        './outputs.FLD/00.create_corpus/20230801.case_study_finalize.fix',
    ]

    engine = SubprocessEngine()
    # engine = QsubEngine('ABCI', 'rt_G.large')

    dry_run = False

    for dataset_uname in dataset_unames:
        output_dir = output_top_dir / dataset_uname
        output_dir.mkdir(exist_ok=True, parents=True)

        dataset_paths = get_local_dataset_paths(dataset_uname,
                                                DATASETS_DIRS,
                                                use_test_as_val=False,
                                                use_test_as_train=False)

        is_settings_copied = False
        for input_path_str in dataset_paths.values():
            if input_path_str is None:
                continue
            input_path = Path(input_path_str)

            output_path = output_dir / input_path.name

            engine.run(
                f'python ./convert_json_schema.py {input_path} {str(output_path)}',
                wait_until_finish=True,
                dry_run=dry_run,
            )

            setting_path = input_path.parent.parent / 'lab.params.json'
            if not is_settings_copied and setting_path.exists():
                engine.run(f'cp {str(setting_path)} {str(output_dir)}')
                is_settings_copied = True

    logger.info('------------- ./00.convert_json_schema.py finished !! -----------')


if __name__ == '__main__':
    main()
