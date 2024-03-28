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
1. Open MPI is already installed. So just do `module load openmpi-xx`
1. Install mpi4py as the above.
1. Modify source code of deepspeed as the above. The difference here is that you should use `eno3` as:
    ```python
         '--mca',
         'btl_tcp_if_include',
    -    'eth0',
    +    'eno3',
    ] + split(self.args.launcher_args)
    ```

* references
    * [Merge branch 'impl-haic' into main](https://gitlab.rdck.intra.hitachi.co.jp/industrial-fm/deepspeed-huggingface/-/commit/8dc7ac4e646582a0edfb959b10afbb2b546c87a2)
    * [README.haic.md](https://gitlab.rdck.intra.hitachi.co.jp/industrial-fm/deepspeed-huggingface/-/blob/main/README.haic.md)

