CORE_FILES := src/csv.c src/k_clustering.c src/point.c
SHARED_CPU_FILES := src/csv.c src/shared_cpu_k_clustering.c src/point.c

all: serial
serial: src/serial.c $(CORE_FILES)
	gcc -Wall src/serial.c $(CORE_FILES) -o output/serial
shared_cpu: src/shared_cpu.c $(SHARED_CPU_FILES)
	gcc -Wall -fopenmp src/shared_cpu.c $(SHARED_CPU_FILES) -o output/shared_cpu
# TODO:
# distributed_cpu: src/distributed_cpu.c $(CORE_FILES)
#	mpicc ...
clean:
	rm -f output/*
