#!/usr/bin/env python
import logging

import click
from logger_setup import setup as setup_logger
from FLD_user_shared_settings import LOCAL_MODELS
import huggingface_hub
from huggingface_hub import HfApi, upload_folder



logger = logging.getLogger(__name__)





@click.command()
def main():
    setup_logger(level=logging.INFO, clear_other_handlers=True)

    # model_name = 'mdl_nm=meta-llama@Meta-Llama-3.1-8B__lgc_dtst_unm=2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15=0.90__optmzr=rec_adam__lrnng=FT.bs-256__step-390.wrmp-200__lrnng_rt=2e-05__rc_adm_fshr_cf=4000__augmnttn=None__prf_intrmdt_stps=None__prf_intrmdt_stps_prb=0.5__mx_grd_nrm=None__sd=2__augmnttn_prb=0.0.chk-390'
    # repo_id = 'hitachi-nlp/Llama-3.1-8B-FLDx2'

    model_name = 'mdl_nm=meta-llama@Meta-Llama-3.1-70B__lgc_dtst_unm=2024-10-23.hybrid__PLD_v2.neg-0.10=0.10__2024-09-18.FLD.neg-0.10.voc-large.theorems-0.15=0.90__optmzr=rec_adam__lrnng=FT.bs-256__step-390.wrmp-200__lrnng_rt=3e-06__rc_adm_fshr_cf=2000__augmnttn=None__prf_intrmdt_stps=None__prf_intrmdt_stps_prb=0.5__mx_grd_nrm=None__sd=5__augmnttn_prb=0.0.chk-390'
    repo_id = 'hitachi-nlp/Llama-3.1-70B-FLDx2'





    api = HfApi()
    api.create_repo(repo_id=repo_id)

    model_path = LOCAL_MODELS[model_name]
    upload_folder(
        folder_path=model_path,
        path_in_repo='.',
        repo_id=repo_id,
        repo_type='model',
    )




if __name__ == '__main__':
    main()
