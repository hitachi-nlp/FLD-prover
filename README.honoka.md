# FLD-Prover for dev

## Installation
```console
$ pip install -r ./requirements/requirements.others.txt
$ git clone https://github.com/HonoMi/FLD-user-shared-settings.git
$ export PYTHONPATH=./FLD-user-shared-settings:$PYTHONPATH
```

### To use DeepSpeed on ABCI
1. Install Open MPI following [here](https://docs.abci.ai/ja/tips/spack/#software-management-operations)
1. Install mpi4py
    * maybe, you have to log in to computation nodes with cuda, on which the program runs. Then,
    ```console
    $ module load hpcx/2.12  # may be not necessary
    $ pip install mpi4py
    ```
1. Modify source code of deepspeed following [abci-examples](https://github.com/ohtaman/abci-examples/tree/main/202307).
    - deepspeed/launcher/multinode_runner.py
        ```python
             '--mca',
             'btl_tcp_if_include',
        -    'eth0',
        +    'eno1',
        ] + split(self.args.launcher_args)
        ```

### To use DeepSpeed on HAIC
1. `source ./set-envs.sh` to load modules, such as CUDA and OpenMPI.
1. Install mpi4py as `pip install mpi4py`.
1. Modify source code of deepspeed as the above. The difference here is that you should use `eno3` as:
    ```python
         '--mca',
         'btl_tcp_if_include',
    -    'eth0',
    +    'eno3',
    ] + split(self.args.launcher_args)
    ```
* references
    * [README.haic.md](https://gitlab.rdck.intra.hitachi.co.jp/industrial-fm/deepspeed-huggingface/-/blob/main/README.haic.md)

### To use deepspeed with "zero2"
Edit `transformers/integrations/deepspeed.py` as follows:
```python
if inference:
    # only Z3 makes sense for the inference
    - if not hf_deepspeed_config.is_zero3():
    -     raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")
    + # if not hf_deepspeed_config.is_zero3():
    + #     raise ValueError("ZeRO inference only makes sense with ZeRO Stage 3 - please adjust your config")
```

