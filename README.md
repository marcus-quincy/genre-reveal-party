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
- module load nvhpc

Dependencies:
- nvhpc 

```sh
./configure
make shared_gpu
./output/shared_gpu
```

## Distributed CPU

Dependencies:
- mpi

```sh
./configure
make distributed_cpu
mpiexec -n 2 ./output/distributed_cpu
```
