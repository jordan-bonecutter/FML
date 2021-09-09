#
#
#
#

WROOT     = `git rev-parse --show-toplevel`
COMPILER  = gcc
FLAGS     = -Wall -ansi -std=c99 -DMEMTOOLS
DEBUG     = -O0 -g -DDEBUG
IMEMTOOLS = -I $(WROOT)/../memtools/
LMEMTOOLS = ../memtools/libmemtools.a
CC        = $(COMPILER) $(FLAGS) $(DEBUG) $(IMEMTOOLS)
CO        = $(CC) -c

all: fml_layer.o fml_net.o fml_layer_fully_connected.o fml_layer_convolution.o

fml_layer.o: src/fml_layer.c
	$(CO) src/fml_layer.c -o fml_layer.o

fml_layer_fully_connected.o: src/fml_layer_internal.h src/fml_layer_fully_connected.h src/fml_layer_fully_connected.c
	$(CO) src/fml_layer_fully_connected.c -o fml_layer_fully_connected.o

fml_layer_convolution.o: src/fml_layer_internal.h src/fml_layer_convolution.h src/fml_layer_convolution.c
	$(CO) src/fml_layer_convolution.c -o fml_layer_convolution.o

fml_net.o: src/fml_net.c
	$(CO) src/fml_net.c -o fml_net.o

clean:
	rm -rf *.o

