# FLD-Prover for dev

## Installation

### To use launcher scripts
```console
$ pip install -r ./requirements/requirements.others.txt
```

### To use DeepSpeed at ABCI
1. Install Open MPI following [here](https://docs.abci.ai/ja/tips/spack/#software-management-operations)
2. Install mpi4py
    ```console

    # may be necessary to log in to computation nodes
    $ qrsh (...)

    $ module load hpcx/2.12  # may be not necessary
    $ pip install mpi4py
    ```
