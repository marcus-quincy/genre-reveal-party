CORE_FILES := src/csv.c src/point.c
HEADER_FILES := src/constants.h

all: serial shared_cpu distributed_cpu shared_gpu
serial: src/serial.c src/k_clustering.c $(CORE_FILES) $(HEADER_FILES)
	gcc -Wall src/serial.c src/k_clustering.c $(CORE_FILES) -o output/serial
shared_cpu: src/shared_cpu.c src/shared_cpu_k_clustering.c $(CORE_FILES) $(HEADER_FILES)
	gcc -Wall -fopenmp src/shared_cpu.c src/shared_cpu_k_clustering.c $(CORE_FILES) -o output/shared_cpu
distributed_cpu: src/distributed_cpu.c src/distributed_cpu_k_clustering.c $(CORE_FILES) $(HEADER_FILES) src/mpi_util.c src/mpi_util.h
	mpicc -g -Wall -std=c99 -o output/distributed_cpu src/distributed_cpu.c src/distributed_cpu_k_clustering.c $(CORE_FILES) src/mpi_util.c
shared_gpu: src/shared_gpu.c src/shared_gpu_k_clustering.c src/kernel.cu $(CORE_FILES) $(HEADER_FILES)
	nvcc src/shared_gpu.c src/shared_gpu_k_clustering.c $(CORE_FILES) src/kernel.cu -o output/shared_gpu
distributed_gpu: src/shared_gpu.c src/shared_gpu_k_clustering.c src/kernel.cu $(CORE_FILES) $(HEADER_FILES)
	mpicc -g -Wall -std=c99 -c src/mpi_util.c -o output/mpi_util.o
	mpicc -g -Wall -std=c99 -c src/point.c -o output/point.o
	mpicc -g -Wall -std=c99 -c src/csv.c -o output/csv.o
	mpicc -g -Wall -std=c99 -c src/distributed_gpu_k_clustering.c -o output/distributed_gpu_k_clustering.o
	mpicc -g -Wall -std=c99 -c src/distributed_gpu.c -o output/distributed_gpu_main.o
	nvcc -c src/kernel.cu -o output/kernel.o
# -lstdc++ is used so __gxx_personality_v0 is defined. otherwise linker throws error. see https://stackoverflow.com/questions/329059/what-is-gxx-personality-v0-for
	mpicc -g -Wall output/*.o -o output/distributed_gpu -lcudart -lstdc++
clean:
	rm -f output/*
