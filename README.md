# genre-reveal-party

The CSV input data set can be obtained from: https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs

Note: `./configure` only needs to be ran once.

The output will be placed in output/output.csv file.

## Serial

```sh
./configure
make serial
./output/serial
```

## Shared CPU

```sh
./configure
make shared_cpu
./output/shared_cpu
```

## Shared GPU

For CHPC:
- module load cuda/12.1

Dependencies:
- cuda

```sh
./configure
make shared_gpu
./output/shared_gpu
```

## Distributed CPU

On CHPC:
- module load intel-mpi

Dependencies:
- mpi

```sh
./configure
make distributed_cpu
mpiexec -n 2 ./output/distributed_cpu
```

## Distributed GPU

On CHPC:
- module load cuda/12.1
- module load gcc/8.5
- module load intel-mpi

This requires having enough GPUs allocated.

```sh
./configure
make distributed_gpu
mpiexec -n 2 ./output/distributed_gpu
```
## Visualization

Dependencies:
- matplotlib
- pandas
- python

```sh
python main.py
```
