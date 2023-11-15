all: serial
serial: src/serial.c
	gcc -Wall src/*.c -o output/serial
clean:
	rm -f output/*
