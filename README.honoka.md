# FLD-Prover for dev

## Installation

### To use launcher scripts
```console
$ pip install -r ./requirements/requirements.launchers.txt
$ git clone https://github.com/HonoMi/FLD-user-shared-settings.git
$ cd FLD-user-shared-settings && export PYTHONPATH=`pwd -r`:$PYTHONPATH
```

### To use DeepSpeed at ABCI
1. Install Open MPI following [here](https://docs.abci.ai/ja/tips/spack/#software-management-operations)
1. Install mpi4py
    ```console

    # may be necessary to log in to computation nodes
    $ qrsh (...)

    $ module load hpcx/2.12  # may be not necessary
    $ pip install mpi4py
    ```
1. [abci-examples](https://github.com/ohtaman/abci-examples/tree/main/202307) に従い，deepspeedのソースをいじる．
