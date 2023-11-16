CORE_FILES := src/csv.c src/k_clustering.c src/point.c

all: serial
serial: src/serial.c $(CORE_FILES)
	gcc -Wall src/serial.c $(CORE_FILES) -o output/serial
# TODO:
# distributed_cpu: src/distributed_cpu.c $(CORE_FILES)
#	mpicc ...
clean:
	rm -f output/*
