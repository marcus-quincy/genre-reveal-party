CORE_FILES := src/csv.c src/point.c

all: serial shared_cpu distributed_cpu
serial: src/serial.c src/k_clustering.c $(CORE_FILES)
	gcc -Wall src/serial.c src/k_clustering.c $(CORE_FILES) -o output/serial
shared_cpu: src/shared_cpu.c src/shared_cpu_k_clustering.c $(CORE_FILES)
	gcc -Wall -fopenmp src/shared_cpu.c src/shared_cpu_k_clustering.c $(CORE_FILES) -o output/shared_cpu
distributed_cpu: src/distributed_cpu.c src/distributed_cpu_k_clustering.c $(CORE_FILES)
	mpicc -g -Wall -std=c99 -o output/distributed_cpu src/distributed_cpu.c src/distributed_cpu_k_clustering.c $(CORE_FILES)
clean:
	rm -f output/*
