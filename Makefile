#
#
#
#

COMPILER  = gcc
FLAGS     = -Wall -ansi -std=c99 -DMEMTOOLS
DEBUG     = -O0 -g -DDEBUG
IMEMTOOLS = -I ../memtools/
LMEMTOOLS = ../memtools/libmemtools.a
CC        = $(COMPILER) $(FLAGS) $(DEBUG) $(IMEMTOOLS)
CO        = $(CC) -c

all: fml_layer.o fml_net.o fml_layer_fully_connected.o fml_layer_convolution.o

fml_layer.o: fml_layer.c
	$(CO) fml_layer.c -o fml_layer.o

fml_layer_fully_connected.o: fml_layer_internal.h fml_layer_fully_connected.h fml_layer_fully_connected.c
	$(CO) fml_layer_fully_connected.c -o fml_layer_fully_connected.o

fml_layer_convolution.o: fml_layer_internal.h fml_layer_convolution.h fml_layer_convolution.c
	$(CO) fml_layer_convolution.c -o fml_layer_convolution.o

fml_net.o: fml_net.c
	$(CO) fml_net.c -o fml_net.o

clean:
	rm -rf *.o

